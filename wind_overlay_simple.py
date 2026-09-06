"""
Wind Overlay - Fetches GFS wind data from Dynamical.org, reprojects to EPSG:3857
Supports both historical (-18h to 0) and forecast (0 to +18h) hours
Supports both 10m surface winds and 100m winds (for wind energy applications)

Uses Dynamical.org zarr endpoint (no rate limiting unlike NOMADS)
Wind speed is computed from U and V components.

Historical data is built up over time by archiving current (hour 0) images
every hour. After 18 hours of running, full -18h to 0 history is available.
"""
import logging
import numpy as np
import zarr
import fsspec
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta, timezone
import math
# Ported from Firebase: R2 stands in for google.cloud.storage. See r2_storage.py.
import r2_storage as storage

logger = logging.getLogger(__name__)

# Global bounds for wind overlay
WIND_BOUNDS = {
    'west': -180.0,
    'south': -85.0,
    'east': 180.0,
    'north': 85.0
}

# Output image dimensions - small for fast loading
OUTPUT_WIDTH = 720
OUTPUT_HEIGHT = 360

# Dynamical.org GFS endpoint
GFS_ZARR_URL = "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr"

# Storage bucket
BUCKET_NAME = 'tree-sixty.firebasestorage.app'

# Historical archive settings - separate folders for 10m and 100m
HISTORY_FOLDER_10M = 'wind_history_10m'
HISTORY_FOLDER_100M = 'wind_history_100m'
TILES_FOLDER_10M = 'wind_tiles_10m'
TILES_FOLDER_100M = 'wind_tiles_100m'
# Legacy folder for backwards compatibility
HISTORY_FOLDER = 'wind_history'
TILES_FOLDER = 'wind_tiles'
MAX_HISTORY_HOURS = 24

# Valid wind levels
VALID_LEVELS = ['10m', '100m']


def latlon_to_web_mercator(lon, lat):
    """Convert lat/lon (EPSG:4326) to Web Mercator meters (EPSG:3857)."""
    lat = max(-85.0511, min(85.0511, lat))
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return x, y


def get_folders_for_level(level='10m'):
    """Get the history and tiles folders for a wind level."""
    if level == '100m':
        return HISTORY_FOLDER_100M, TILES_FOLDER_100M
    else:
        # Default to 10m, also use for legacy
        return HISTORY_FOLDER_10M, TILES_FOLDER_10M


def get_history_blob_name(valid_time, level='10m'):
    """Get the blob name for a historical archive image."""
    history_folder, _ = get_folders_for_level(level)
    return f"{history_folder}/{valid_time.strftime('%Y%m%d_%H')}.webp"


def archive_current_wind(level='10m'):
    """
    Archive the current (hour 0) wind image to the historical archive.

    Args:
        level: '10m' or '100m' wind level
    """
    try:
        now = datetime.now(timezone.utc)
        valid_time = now.replace(minute=0, second=0, microsecond=0)

        logger.info(f"Archiving current {level} wind for valid_time: {valid_time.isoformat()}")

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        archive_blob_name = get_history_blob_name(valid_time, level)
        archive_blob = bucket.blob(archive_blob_name)

        if archive_blob.exists():
            logger.info(f"{level} wind archive already exists for {valid_time.isoformat()}, skipping")
            return {
                'status': 'already_exists',
                'valid_time': valid_time.isoformat(),
                'url': archive_blob.public_url,
                'level': level
            }

        result = generate_wind_image(forecast_hour=0, level=level)

        if not result:
            raise Exception(f"Failed to generate {level} wind image")

        source_blob_path = result.get('blob_path')
        if source_blob_path:
            source_blob = bucket.blob(source_blob_path)
            if source_blob.exists():
                bucket.copy_blob(source_blob, bucket, archive_blob_name)
                archive_blob = bucket.blob(archive_blob_name)
                archive_blob.make_public()
                logger.info(f"Archived {level} wind to: {archive_blob_name}")

        cleanup_old_archives(storage_client, bucket, level)

        return {
            'status': 'archived',
            'valid_time': valid_time.isoformat(),
            'url': archive_blob.public_url,
            'level': level
        }

    except Exception as e:
        logger.error(f"Error archiving {level} wind: {e}", exc_info=True)
        return None


def cleanup_old_archives(storage_client, bucket, level='10m'):
    """Remove archived images older than MAX_HISTORY_HOURS."""
    try:
        history_folder, _ = get_folders_for_level(level)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=MAX_HISTORY_HOURS)
        blobs = bucket.list_blobs(prefix=f"{history_folder}/")
        deleted_count = 0

        for blob in blobs:
            try:
                filename = blob.name.split('/')[-1].replace('.webp', '')
                blob_time = datetime.strptime(filename, '%Y%m%d_%H').replace(tzinfo=timezone.utc)

                if blob_time < cutoff_time:
                    blob.delete()
                    deleted_count += 1
                    logger.info(f"Deleted old {level} wind archive: {blob.name}")
            except (ValueError, IndexError):
                continue

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old {level} wind archive(s)")

    except Exception as e:
        logger.error(f"Error cleaning up {level} wind archives: {e}", exc_info=True)


def get_historical_wind(hours_ago, level='10m'):
    """Get a historical wind image from the archive."""
    try:
        hours_ago = abs(int(hours_ago))
        if hours_ago < 1 or hours_ago > 18:
            return None

        now = datetime.now(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        target_time = current_hour - timedelta(hours=hours_ago)

        logger.info(f"Looking for historical {level} wind: {hours_ago}h ago = {target_time.isoformat()}")

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        archive_blob_name = get_history_blob_name(target_time, level)
        archive_blob = bucket.blob(archive_blob_name)

        if archive_blob.exists():
            logger.info(f"Found historical {level} wind archive: {archive_blob_name}")
            return {
                'url': archive_blob.public_url,
                'valid_time': target_time.isoformat(),
                'hours_ago': hours_ago,
                'bounds': WIND_BOUNDS,
                'projection': 'EPSG:3857',
                'source': 'archive',
                'level': level
            }
        else:
            logger.info(f"No historical {level} wind archive found for {target_time.isoformat()}")
            return None

    except Exception as e:
        logger.error(f"Error getting historical {level} wind: {e}", exc_info=True)
        return None


def list_available_wind_history(level='10m'):
    """List all available historical hours for a wind level."""
    try:
        now = datetime.now(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        history_folder, _ = get_folders_for_level(level)
        available_hours = []
        blobs = bucket.list_blobs(prefix=f"{history_folder}/")

        for blob in blobs:
            try:
                filename = blob.name.split('/')[-1].replace('.webp', '')
                blob_time = datetime.strptime(filename, '%Y%m%d_%H').replace(tzinfo=timezone.utc)
                hours_ago = int((current_hour - blob_time).total_seconds() / 3600)

                if 1 <= hours_ago <= 18:
                    available_hours.append(hours_ago)
            except (ValueError, IndexError):
                continue

        return sorted(available_hours)

    except Exception as e:
        logger.error(f"Error listing {level} wind history: {e}", exc_info=True)
        return []


def generate_wind_image(forecast_hour=0, level='10m', email="tree60weather@gmail.com"):
    """
    Generate a wind overlay image for the specified forecast hour and level.

    Args:
        forecast_hour: Hour offset (0-18)
        level: '10m' for surface winds, '100m' for upper level winds
        email: Email for Dynamical.org tracking
    """
    try:
        now = datetime.now(timezone.utc)
        forecast_hour = max(0, min(18, forecast_hour))

        _, tiles_folder = get_folders_for_level(level)
        gfs_run_hour = (now.hour // 6) * 6
        cache_time = now.replace(hour=gfs_run_hour, minute=0, second=0, microsecond=0)
        cache_key = f'gfs_wind_{level}_{cache_time.strftime("%Y%m%d_%H")}_h{forecast_hour:+03d}'
        blob_path = f'{tiles_folder}/{cache_key}.webp'

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_path)

        logger.info(f"Generating {level} wind image for hour {forecast_hour}")

        url = GFS_ZARR_URL
        if email:
            url += f"?email={email}"

        store = fsspec.get_mapper(url)
        root = zarr.open_consolidated(store, mode='r')

        lat_array = root['latitude'][:]
        lon_array = root['longitude'][:]
        lead_time_array = root['lead_time'][:]
        init_time_array = root['init_time'][:]

        lead_time_hours = lead_time_array / 3600.0
        hour_idx = int(np.argmin(np.abs(lead_time_hours - forecast_hour)))
        actual_lead = int(lead_time_hours[hour_idx])

        # Load U and V wind components based on level
        if level == '100m':
            logger.info("Loading 100m wind U and V components...")
            wind_u = root['wind_u_100m'][0, hour_idx, :, :]
            wind_v = root['wind_v_100m'][0, hour_idx, :, :]
        else:
            logger.info("Loading 10m wind U and V components...")
            wind_u = root['wind_u_10m'][0, hour_idx, :, :]
            wind_v = root['wind_v_10m'][0, hour_idx, :, :]

        # Calculate wind speed
        wind_speed = np.sqrt(np.array(wind_u)**2 + np.array(wind_v)**2)

        logger.info(f"{level} wind speed range: {np.nanmin(wind_speed):.1f} to {np.nanmax(wind_speed):.1f} m/s")

        lons_2d, lats_2d = np.meshgrid(lon_array, lat_array)

        reprojected = reproject_to_web_mercator(
            wind_speed, lats_2d, lons_2d,
            WIND_BOUNDS, OUTPUT_WIDTH, OUTPUT_HEIGHT
        )

        img = create_wind_image(reprojected)

        img_bytes = BytesIO()
        img.save(img_bytes, format='WEBP', quality=40, method=6, lossless=False)
        img_bytes.seek(0)

        compressed_size = len(img_bytes.getvalue())
        logger.info(f"Compressed {level} wind to WebP: {compressed_size/1024:.1f} KB")

        blob.upload_from_file(img_bytes, content_type='image/webp')
        blob.make_public()

        init_time = datetime.fromtimestamp(int(init_time_array[0]) / 1e9, tz=timezone.utc)
        valid_time = init_time + timedelta(hours=actual_lead)

        return {
            'url': blob.public_url,
            'blob_path': blob_path,
            'forecast_hour': forecast_hour,
            'level': level,
            'actual_lead_time': actual_lead,
            'init_time': init_time.isoformat(),
            'valid_time': valid_time.isoformat(),
            'bounds': WIND_BOUNDS,
            'projection': 'EPSG:3857',
            'wind_range': {
                'min': float(np.nanmin(wind_speed)),
                'max': float(np.nanmax(wind_speed))
            }
        }

    except Exception as e:
        logger.error(f"Error generating {level} wind image: {e}", exc_info=True)
        return None


def get_gfs_wind_png(forecast_hour=0, level='10m', email="tree60weather@gmail.com"):
    """
    Get or generate wind overlay PNG/WebP for specified hour and level.

    Args:
        forecast_hour: Hour offset (-18 to +18)
        level: '10m' for surface winds, '100m' for upper level winds
        email: Email for Dynamical.org tracking
    """
    try:
        now = datetime.now(timezone.utc)
        forecast_hour = max(-18, min(18, forecast_hour))

        # Validate level
        if level not in VALID_LEVELS:
            level = '10m'

        logger.info(f"Wind overlay request: hour={forecast_hour}, level={level}")

        # HISTORICAL
        if forecast_hour < 0:
            hours_ago = abs(forecast_hour)
            historical = get_historical_wind(hours_ago, level)

            if historical:
                return {
                    'url': historical['url'],
                    'cached': True,
                    'forecast_hour': forecast_hour,
                    'level': level,
                    'valid_time': historical['valid_time'],
                    'bounds': WIND_BOUNDS,
                    'projection': 'EPSG:3857',
                    'source': 'archive'
                }
            else:
                return {
                    'status': 'unavailable',
                    'forecast_hour': forecast_hour,
                    'level': level,
                    'message': f'Historical {level} wind data for {hours_ago}h ago not yet available.',
                    'available_history': list_available_wind_history(level),
                    'bounds': WIND_BOUNDS,
                    'projection': 'EPSG:3857'
                }

        # FORECAST/CURRENT
        _, tiles_folder = get_folders_for_level(level)
        gfs_run_hour = (now.hour // 6) * 6
        cache_time = now.replace(hour=gfs_run_hour, minute=0, second=0, microsecond=0)
        cache_key = f'gfs_wind_{level}_{cache_time.strftime("%Y%m%d_%H")}_h{forecast_hour:+03d}'
        blob_path = f'{tiles_folder}/{cache_key}.webp'

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_path)

        if blob.exists():
            blob.reload()
            age_seconds = (datetime.now(timezone.utc) - blob.updated).total_seconds()
            if age_seconds < 21600:  # 6 hours
                logger.info(f"Using cached {level} wind overlay: {cache_key} (age: {age_seconds/60:.1f}min)")
                return {
                    'url': blob.public_url,
                    'cached': True,
                    'forecast_hour': forecast_hour,
                    'level': level,
                    'bounds': WIND_BOUNDS,
                    'projection': 'EPSG:3857'
                }

        # Generate new overlay
        logger.info(f"Generating {level} wind overlay from Dynamical.org for hour {forecast_hour}")

        url = GFS_ZARR_URL
        if email:
            url += f"?email={email}"

        store = fsspec.get_mapper(url)
        root = zarr.open_consolidated(store, mode='r')

        lat_array = root['latitude'][:]
        lon_array = root['longitude'][:]
        lead_time_array = root['lead_time'][:]
        init_time_array = root['init_time'][:]

        lead_time_hours = lead_time_array / 3600.0
        hour_idx = int(np.argmin(np.abs(lead_time_hours - forecast_hour)))
        actual_lead = int(lead_time_hours[hour_idx])

        logger.info(f"Requested lead time: {forecast_hour}h, using: {actual_lead}h (index {hour_idx})")

        # Load wind components based on level
        if level == '100m':
            logger.info("Loading 100m wind U and V components...")
            wind_u = root['wind_u_100m'][0, hour_idx, :, :]
            wind_v = root['wind_v_100m'][0, hour_idx, :, :]
        else:
            logger.info("Loading 10m wind U and V components...")
            wind_u = root['wind_u_10m'][0, hour_idx, :, :]
            wind_v = root['wind_v_10m'][0, hour_idx, :, :]

        wind_speed = np.sqrt(np.array(wind_u)**2 + np.array(wind_v)**2)

        logger.info(f"{level} wind speed range: {np.nanmin(wind_speed):.1f} to {np.nanmax(wind_speed):.1f} m/s")

        lons_2d, lats_2d = np.meshgrid(lon_array, lat_array)

        reprojected = reproject_to_web_mercator(
            wind_speed, lats_2d, lons_2d,
            WIND_BOUNDS, OUTPUT_WIDTH, OUTPUT_HEIGHT
        )

        img = create_wind_image(reprojected)

        img_bytes = BytesIO()
        img.save(img_bytes, format='WEBP', quality=40, method=6, lossless=False)
        img_bytes.seek(0)

        compressed_size = len(img_bytes.getvalue())
        logger.info(f"Compressed {level} wind to WebP: {compressed_size/1024:.1f} KB")

        blob.upload_from_file(img_bytes, content_type='image/webp')
        blob.make_public()

        init_time = datetime.fromtimestamp(int(init_time_array[0]) / 1e9, tz=timezone.utc)
        valid_time = init_time + timedelta(hours=actual_lead)

        return {
            'url': blob.public_url,
            'cached': False,
            'forecast_hour': forecast_hour,
            'level': level,
            'actual_lead_time': actual_lead,
            'init_time': init_time.isoformat(),
            'valid_time': valid_time.isoformat(),
            'bounds': WIND_BOUNDS,
            'projection': 'EPSG:3857',
            'wind_range': {
                'min': float(np.nanmin(wind_speed)),
                'max': float(np.nanmax(wind_speed))
            }
        }

    except Exception as e:
        logger.error(f"Error generating {level} wind overlay: {e}", exc_info=True)
        raise


def reproject_to_web_mercator(data, lats, lons, bounds, out_width, out_height):
    """Reproject wind data from EPSG:4326 to EPSG:3857."""
    from scipy import ndimage

    if lons.max() > 180:
        lons = np.where(lons > 180, lons - 360, lons)
        row_lons = lons[0, :]
        split_idx = np.argmax(row_lons > 180)
        if split_idx == 0:
            split_idx = len(row_lons) // 2
        data = np.roll(data, -split_idx, axis=1)
        lats = np.roll(lats, -split_idx, axis=1)
        lons = np.roll(lons, -split_idx, axis=1)

    if lats[0, 0] > lats[-1, 0]:
        data = np.flipud(data)
        lats = np.flipud(lats)
        lons = np.flipud(lons)

    # Add wrap column at dateline
    data = np.hstack([data, data[:, 0:1]])
    lats = np.hstack([lats, lats[:, 0:1]])
    last_lon = lons[:, -1:] + (lons[:, 1:2] - lons[:, 0:1])
    lons = np.hstack([lons, last_lon])

    src_lat_min = float(lats.min())
    src_lat_max = float(lats.max())
    src_lon_min = float(lons.min())
    src_lon_max = float(lons.max())

    x_min, y_min = latlon_to_web_mercator(bounds['west'], bounds['south'])
    x_max, y_max = latlon_to_web_mercator(bounds['east'], bounds['north'])

    x_out = np.linspace(x_min, x_max, out_width, dtype=np.float32)
    y_out = np.linspace(y_max, y_min, out_height, dtype=np.float32)

    X_out, Y_out = np.meshgrid(x_out, y_out)

    out_lons = X_out * 180.0 / 20037508.34
    out_lats = np.arctan(np.sinh(Y_out * np.pi / 20037508.34)) * 180.0 / np.pi

    lat_frac = (out_lats - src_lat_min) / (src_lat_max - src_lat_min)
    lon_frac = (out_lons - src_lon_min) / (src_lon_max - src_lon_min)

    src_rows = lat_frac * (data.shape[0] - 1)
    src_cols = lon_frac * (data.shape[1] - 1)

    valid_mask = (
        (out_lats >= src_lat_min) & (out_lats <= src_lat_max) &
        (out_lons >= -180) & (out_lons <= 180)
    )

    src_rows = np.clip(src_rows, 0, data.shape[0] - 1)
    src_cols = np.clip(src_cols, 0, data.shape[1] - 1)

    coordinates = np.array([src_rows.ravel(), src_cols.ravel()])

    output_flat = ndimage.map_coordinates(
        data.astype(np.float32),
        coordinates,
        order=1,
        mode='nearest',
        prefilter=False
    )

    output = output_flat.reshape((out_height, out_width))
    output = np.where(valid_mask, output, np.nan)

    return output


def create_wind_image(wind_array):
    """
    Create RGBA image from wind speed array with Windy.com-style color mapping.
    Wind speed in m/s mapped to colors.
    """
    height, width = wind_array.shape
    img_array = np.zeros((height, width, 4), dtype=np.uint8)

    # Windy.com-inspired wind speed color scale (m/s)
    # Calm (0-2): Light blue/gray
    # Light (2-5): Green
    # Moderate (5-10): Yellow
    # Fresh (10-15): Orange
    # Strong (15-25): Red
    # Gale (25-35): Magenta
    # Storm (35+): Purple/Black
    speed_breaks = np.array([0, 2, 5, 10, 15, 25, 35, 50], dtype=np.float32)
    colors = np.array([
        [200, 230, 255],  # 0: Very light blue (calm)
        [100, 200, 100],  # 2: Green (light)
        [255, 255, 100],  # 5: Yellow (moderate)
        [255, 180, 50],   # 10: Orange (fresh)
        [255, 80, 80],    # 15: Red (strong)
        [200, 50, 200],   # 25: Magenta (gale)
        [150, 30, 150],   # 35: Dark magenta (storm)
        [80, 0, 80]       # 50: Dark purple (hurricane)
    ], dtype=np.float32)

    valid_mask = ~np.isnan(wind_array)
    speed_safe = np.where(valid_mask, wind_array, 0)
    speed_clamped = np.clip(speed_safe, speed_breaks[0], speed_breaks[-1])

    segment_indices = np.searchsorted(speed_breaks, speed_clamped, side='right') - 1
    segment_indices = np.clip(segment_indices, 0, len(speed_breaks) - 2)

    lower_speeds = speed_breaks[segment_indices]
    upper_speeds = speed_breaks[segment_indices + 1]
    t = (speed_clamped - lower_speeds) / (upper_speeds - lower_speeds)
    t = np.clip(t, 0, 1)

    lower_colors = colors[segment_indices]
    upper_colors = colors[segment_indices + 1]

    r = (lower_colors[:, :, 0] + t * (upper_colors[:, :, 0] - lower_colors[:, :, 0])).astype(np.uint8)
    g = (lower_colors[:, :, 1] + t * (upper_colors[:, :, 1] - lower_colors[:, :, 1])).astype(np.uint8)
    b = (lower_colors[:, :, 2] + t * (upper_colors[:, :, 2] - lower_colors[:, :, 2])).astype(np.uint8)

    img_array[:, :, 0] = np.where(valid_mask, r, 0)
    img_array[:, :, 1] = np.where(valid_mask, g, 0)
    img_array[:, :, 2] = np.where(valid_mask, b, 0)
    img_array[:, :, 3] = np.where(valid_mask, 160, 0)  # Slightly more transparent than temp

    return Image.fromarray(img_array, 'RGBA')


def regenerate_all_wind_tiles():
    """Regenerate all wind tiles (-18 to +18 hours)."""
    results = []
    errors = []

    for hour in range(-18, 19):
        try:
            if hour < 0:
                # Historical - skip, these come from archive
                continue
            result = generate_wind_image(forecast_hour=hour)
            if result:
                results.append({
                    'hour': hour,
                    'url': result['url'],
                    'status': 'success'
                })
                logger.info(f"Generated wind tile for hour {hour}")
            else:
                errors.append({'hour': hour, 'error': 'Generation returned None'})
        except Exception as e:
            logger.error(f"Error generating wind tile for hour {hour}: {e}")
            errors.append({'hour': hour, 'error': str(e)})

    return {
        'success': len(results),
        'errors': len(errors),
        'results': results,
        'error_details': errors
    }
