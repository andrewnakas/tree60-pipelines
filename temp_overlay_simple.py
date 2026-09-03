"""
Temperature Overlay - Fetches GFS data from Dynamical.org, reprojects to EPSG:3857
Supports both historical (-18h to 0) and forecast (0 to +18h) hours

Uses Dynamical.org zarr endpoint (no rate limiting unlike NOMADS)

Historical data is built up over time by archiving current (hour 0) images
every hour. After 18 hours of running, full -18h to 0 history is available.

CRITICAL ALIGNMENT NOTES:
- Images are generated in EPSG:3857 (Web Mercator) to match Leaflet basemap
- Frontend MUST use convertLatLngBoundsForEPSG3857Image() for proper alignment
- This matches the HRRR radar overlay alignment approach
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

# Global bounds for temperature overlay
TEMP_BOUNDS = {
    'west': -180.0,
    'south': -85.0,  # Web Mercator limit is ~85.06 degrees
    'east': 180.0,
    'north': 85.0
}

# Output image dimensions - optimized for fast loading
# 1000x500 is smaller and loads faster (~30-40KB at quality 50)
OUTPUT_WIDTH = 1000
OUTPUT_HEIGHT = 500

# Dynamical.org endpoints
GFS_ZARR_URL = "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr"
GEFS_ANALYSIS_URL = "https://data.dynamical.org/noaa/gefs/analysis/latest.zarr"  # Current to present

# Storage bucket
BUCKET_NAME = 'tree-sixty.firebasestorage.app'

# Historical archive settings
HISTORY_FOLDER = 'temperature_history'
MAX_HISTORY_HOURS = 24  # Keep 24 hours of history (cleanup older)


def latlon_to_web_mercator(lon, lat):
    """Convert lat/lon (EPSG:4326) to Web Mercator meters (EPSG:3857)."""
    lat = max(-85.0511, min(85.0511, lat))
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return x, y


def get_history_blob_name(valid_time):
    """Get the blob name for a historical archive image."""
    return f"{HISTORY_FOLDER}/{valid_time.strftime('%Y%m%d_%H')}.webp"


def fetch_gefs_analysis_temperature(target_time):
    """
    Fetch temperature data from GEFS analysis for a specific historical time.

    GEFS analysis provides actual analyzed weather conditions (current to present).
    Data is available at 3-hour intervals. Uses same color mapping as GFS forecast.

    Args:
        target_time: datetime object for the desired hour (will round to nearest 3-hour interval)

    Returns:
        tuple: (temps_c, lat_array, lon_array, actual_time)
            temps_c: 2D numpy array of temperature in Celsius
            lat_array, lon_array: 1D coordinate arrays
            actual_time: datetime of the actual data time used
    """
    logger.info(f"Fetching GEFS analysis data for {target_time.isoformat()}")

    url = GEFS_ANALYSIS_URL + "?email=tree60weather@gmail.com"
    store = fsspec.get_mapper(url)
    root = zarr.open_consolidated(store, mode='r')

    # Get coordinate arrays
    lat_array = root['latitude'][:]
    lon_array = root['longitude'][:]
    time_array = root['time'][:]  # Unix timestamps (format TBD - check first value)

    # Convert target_time to Unix timestamp (seconds)
    target_timestamp_sec = target_time.timestamp()

    # Find nearest time in the dataset (GEFS analysis is 3-hourly)
    # First, determine if time_array is in seconds or nanoseconds by checking magnitude
    sample_time = float(time_array[0])
    if sample_time > 1e12:  # Nanoseconds (e.g., 1.7e18 for year 2024)
        # Convert to seconds for comparison
        time_array_sec = time_array / 1e9
        target_timestamp = target_timestamp_sec
        is_nanoseconds = True
    else:  # Already in seconds
        time_array_sec = time_array
        target_timestamp = target_timestamp_sec
        is_nanoseconds = False

    time_idx = int(np.argmin(np.abs(time_array_sec - target_timestamp)))
    actual_timestamp_sec = float(time_array_sec[time_idx])
    actual_time = datetime.fromtimestamp(actual_timestamp_sec, tz=timezone.utc)

    logger.info(f"Time format detected: {'nanoseconds' if is_nanoseconds else 'seconds'}, sample={sample_time}")

    logger.info(f"Using GEFS analysis time: {actual_time.isoformat()} (requested: {target_time.isoformat()})")

    # Load temperature data - GEFS analysis has simple structure: (time, latitude, longitude)
    temps_c = root['temperature_2m'][time_idx, :, :]

    return np.array(temps_c), lat_array, lon_array, actual_time


def archive_current_temperature():
    """
    Archive the current temperature image by copying the existing forecast tile.
    This should be called every hour to build up historical data.

    IMPORTANT: Copies the actual forecast tile to ensure PERFECT visual consistency.
    What you see as hour 0 will look exactly the same when it becomes hour -1.

    Returns:
        dict with archive info or None if failed
    """
    try:
        now = datetime.now(timezone.utc)
        # Round to nearest hour for archive naming
        valid_time = now.replace(minute=0, second=0, microsecond=0)

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        # Check if already archived for this hour
        archive_blob_name = get_history_blob_name(valid_time)
        archive_blob = bucket.blob(archive_blob_name)

        if archive_blob.exists():
            logger.info(f"Archive already exists for {valid_time.isoformat()}, skipping")
            return {
                'status': 'already_exists',
                'valid_time': valid_time.isoformat(),
                'url': archive_blob.public_url
            }

        # Find the GFS forecast tile that corresponds to this valid_time
        # GFS runs every 6 hours at 00Z, 06Z, 12Z, 18Z
        gfs_run_hour = (valid_time.hour // 6) * 6
        gfs_run_time = valid_time.replace(hour=gfs_run_hour, minute=0, second=0, microsecond=0)

        # If valid_time is before the most recent GFS run, use previous run
        if valid_time < gfs_run_time:
            gfs_run_time -= timedelta(hours=6)

        # Calculate forecast hour for this valid_time
        forecast_hour = int((valid_time - gfs_run_time).total_seconds() / 3600)

        # Look for existing forecast tile
        forecast_cache_key = f'gfs_temp_{gfs_run_time.strftime("%Y%m%d_%H")}_h{forecast_hour:+03d}'
        forecast_blob_path = f'temperature_tiles/{forecast_cache_key}.webp'
        forecast_blob = bucket.blob(forecast_blob_path)

        if forecast_blob.exists():
            # Copy the existing forecast tile to historical archive
            logger.info(f"Copying forecast tile {forecast_blob_path} to historical archive")

            # Download and re-upload to historical archive
            img_bytes = forecast_blob.download_as_bytes()
            archive_blob.upload_from_string(img_bytes, content_type='image/webp')
            archive_blob.make_public()

            logger.info(f"Archived by copying forecast tile: {archive_blob_name}")
            data_source = 'copied_from_forecast'

        else:
            # Forecast tile doesn't exist, generate from GFS data
            logger.info(f"Forecast tile not found, generating from GFS data")

            url = GFS_ZARR_URL + "?email=tree60weather@gmail.com"
            store = fsspec.get_mapper(url)
            root = zarr.open_consolidated(store, mode='r')

            lat_array = root['latitude'][:]
            lon_array = root['longitude'][:]
            lead_time_array = root['lead_time'][:]

            lead_time_hours = lead_time_array / 3600.0
            hour_idx = int(np.argmin(np.abs(lead_time_hours - forecast_hour)))

            # Use index 0 to match forecast generation (not init_idx)
            temps_c = root['temperature_2m'][0, hour_idx, :, :]

            # Create 2D coordinate grids
            lons_2d, lats_2d = np.meshgrid(lon_array, lat_array)

            # Reproject to EPSG:3857
            reprojected = reproject_to_web_mercator(
                np.array(temps_c), lats_2d, lons_2d,
                TEMP_BOUNDS, OUTPUT_WIDTH, OUTPUT_HEIGHT
            )

            # Create image
            img = create_temperature_image(reprojected)

            # Compress to WebP
            img_bytes_io = BytesIO()
            img.save(img_bytes_io, format='WEBP', quality=50, method=6, lossless=False)
            img_bytes = img_bytes_io.getvalue()

            # Upload to archive
            archive_blob.upload_from_string(img_bytes, content_type='image/webp')
            archive_blob.make_public()

            logger.info(f"Archived from GFS generation: {archive_blob_name}")
            data_source = 'generated_from_gfs'

        # Cleanup old archives
        cleanup_old_archives(storage_client, bucket)

        return {
            'status': 'archived',
            'valid_time': valid_time.isoformat(),
            'data_source': data_source,
            'url': archive_blob.public_url
        }

    except Exception as e:
        logger.error(f"Error archiving temperature: {e}", exc_info=True)
        return None


def cleanup_old_archives(storage_client, bucket):
    """Remove archived images older than MAX_HISTORY_HOURS."""
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=MAX_HISTORY_HOURS)

        # List all blobs in history folder
        blobs = bucket.list_blobs(prefix=f"{HISTORY_FOLDER}/")
        deleted_count = 0

        for blob in blobs:
            # Parse the datetime from blob name
            # Format: temperature_history/YYYYMMDD_HH.webp
            try:
                filename = blob.name.split('/')[-1].replace('.webp', '')
                blob_time = datetime.strptime(filename, '%Y%m%d_%H').replace(tzinfo=timezone.utc)

                if blob_time < cutoff_time:
                    blob.delete()
                    deleted_count += 1
                    logger.info(f"Deleted old archive: {blob.name}")
            except (ValueError, IndexError):
                continue

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old archive(s)")

    except Exception as e:
        logger.error(f"Error cleaning up archives: {e}", exc_info=True)


def get_historical_temperature(hours_ago):
    """
    Get a historical temperature image from the archive.

    Args:
        hours_ago: Positive integer, how many hours in the past (1-18)

    Returns:
        dict with url and metadata, or None if not available
    """
    try:
        hours_ago = abs(int(hours_ago))
        if hours_ago < 1 or hours_ago > 18:
            return None

        now = datetime.now(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        target_time = current_hour - timedelta(hours=hours_ago)

        logger.info(f"Looking for historical temperature: {hours_ago}h ago = {target_time.isoformat()}")

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        archive_blob_name = get_history_blob_name(target_time)
        archive_blob = bucket.blob(archive_blob_name)

        if archive_blob.exists():
            logger.info(f"Found historical archive: {archive_blob_name}")
            return {
                'url': archive_blob.public_url,
                'valid_time': target_time.isoformat(),
                'hours_ago': hours_ago,
                'bounds': TEMP_BOUNDS,
                'projection': 'EPSG:3857',
                'source': 'archive'
            }
        else:
            logger.info(f"No historical archive found for {target_time.isoformat()}")
            return None

    except Exception as e:
        logger.error(f"Error getting historical temperature: {e}", exc_info=True)
        return None


def list_available_history():
    """
    List all available historical hours.

    Returns:
        List of available hours_ago values (e.g., [1, 2, 3, ...])
    """
    try:
        now = datetime.now(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        available_hours = []
        blobs = bucket.list_blobs(prefix=f"{HISTORY_FOLDER}/")

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
        logger.error(f"Error listing history: {e}", exc_info=True)
        return []


def generate_temperature_image(forecast_hour=0, email="tree60weather@gmail.com"):
    """
    Generate a temperature overlay image for the specified forecast hour.
    This is the core generation function, separated from caching logic.

    Returns:
        dict with image data and blob_path, or None on failure
    """
    try:
        now = datetime.now(timezone.utc)

        # Clamp to valid forecast range (0-18)
        forecast_hour = max(0, min(18, forecast_hour))

        # Create cache key using GFS run time (00Z, 06Z, 12Z, 18Z)
        gfs_run_hour = (now.hour // 6) * 6
        cache_time = now.replace(hour=gfs_run_hour, minute=0, second=0, microsecond=0)
        cache_key = f'gfs_temp_{cache_time.strftime("%Y%m%d_%H")}_h{forecast_hour:+03d}'
        blob_path = f'temperature_tiles/{cache_key}.webp'

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_path)

        # Generate from Dynamical.org
        logger.info(f"Generating temperature image for hour {forecast_hour}")

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

        # Find closest lead time
        hour_idx = int(np.argmin(np.abs(lead_time_hours - forecast_hour)))
        actual_lead = int(lead_time_hours[hour_idx])

        # Load temperature data
        temp_data = root['temperature_2m']
        temps_c = temp_data[0, hour_idx, :, :]

        # Create 2D coordinate grids
        lons_2d, lats_2d = np.meshgrid(lon_array, lat_array)

        # Reproject to EPSG:3857
        reprojected = reproject_to_web_mercator(
            np.array(temps_c), lats_2d, lons_2d,
            TEMP_BOUNDS, OUTPUT_WIDTH, OUTPUT_HEIGHT
        )

        # Create image
        img = create_temperature_image(reprojected)

        # Compress to WebP
        img_bytes = BytesIO()
        img.save(img_bytes, format='WEBP', quality=50, method=6, lossless=False)
        img_bytes.seek(0)

        # Upload to storage
        blob.upload_from_file(img_bytes, content_type='image/webp')
        blob.make_public()

        # Get metadata
        init_time = datetime.fromtimestamp(int(init_time_array[-1]) / 1e9, tz=timezone.utc)
        valid_time = init_time + timedelta(hours=actual_lead)

        return {
            'url': blob.public_url,
            'blob_path': blob_path,
            'forecast_hour': forecast_hour,
            'actual_lead_time': actual_lead,
            'init_time': init_time.isoformat(),
            'valid_time': valid_time.isoformat(),
            'bounds': TEMP_BOUNDS,
            'projection': 'EPSG:3857',
            'temp_range': {
                'min': float(np.nanmin(temps_c)),
                'max': float(np.nanmax(temps_c))
            }
        }

    except Exception as e:
        logger.error(f"Error generating temperature image: {e}", exc_info=True)
        return None


def get_gfs_temperature_png(forecast_hour=0, email="tree60weather@gmail.com"):
    """
    Get or generate temperature overlay PNG/WebP for specified hour.

    For forecast hours (0 to +18): Uses Dynamical.org GFS zarr data
    For historical hours (-18 to -1): Uses archived images from rolling history

    Args:
        forecast_hour: int - Hour offset from current time (-18 to +18)
        email: str - Email for Dynamical.org usage tracking

    Returns:
        dict with url, bounds, projection info, etc.
    """
    try:
        now = datetime.now(timezone.utc)

        # Clamp to valid range
        forecast_hour = max(-18, min(18, forecast_hour))

        logger.info(f"Temperature overlay request: hour={forecast_hour}")

        # HISTORICAL: Look up from archive
        if forecast_hour < 0:
            hours_ago = abs(forecast_hour)
            historical = get_historical_temperature(hours_ago)

            if historical:
                return {
                    'url': historical['url'],
                    'cached': True,
                    'forecast_hour': forecast_hour,
                    'valid_time': historical['valid_time'],
                    'bounds': TEMP_BOUNDS,
                    'projection': 'EPSG:3857',
                    'source': 'archive'
                }
            else:
                # No historical data available yet
                logger.info(f"No historical data for hour {forecast_hour}, returning unavailable")
                return {
                    'status': 'unavailable',
                    'forecast_hour': forecast_hour,
                    'message': f'Historical data for {hours_ago}h ago not yet available. History builds up over time.',
                    'available_history': list_available_history(),
                    'bounds': TEMP_BOUNDS,
                    'projection': 'EPSG:3857'
                }

        # FORECAST/CURRENT: Generate or use cache
        # Cache by GFS init_time (00Z, 06Z, 12Z, 18Z) so tiles stay valid until next GFS run
        # GFS runs at 00, 06, 12, 18 UTC - cache for 6 hours
        gfs_run_hour = (now.hour // 6) * 6  # Round down to nearest GFS run
        cache_time = now.replace(hour=gfs_run_hour, minute=0, second=0, microsecond=0)
        cache_key = f'gfs_temp_{cache_time.strftime("%Y%m%d_%H")}_h{forecast_hour:+03d}'
        blob_path = f'temperature_tiles/{cache_key}.webp'

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_path)

        if blob.exists():
            blob.reload()
            age_seconds = (datetime.now(timezone.utc) - blob.updated).total_seconds()
            # Cache for 6 hours (matches GFS update cycle)
            if age_seconds < 21600:  # 6 hours
                logger.info(f"Using cached temperature overlay: {cache_key} (age: {age_seconds/60:.1f}min)")
                return {
                    'url': blob.public_url,
                    'cached': True,
                    'forecast_hour': forecast_hour,
                    'bounds': TEMP_BOUNDS,
                    'projection': 'EPSG:3857'
                }

        # Generate new overlay from Dynamical.org
        logger.info(f"Generating temperature overlay from Dynamical.org for hour {forecast_hour}")

        url = GFS_ZARR_URL
        if email:
            url += f"?email={email}"

        # Open zarr store
        logger.info(f"Opening zarr store: {url}")
        store = fsspec.get_mapper(url)
        root = zarr.open_consolidated(store, mode='r')

        # Get coordinate arrays
        lat_array = root['latitude'][:]
        lon_array = root['longitude'][:]
        lead_time_array = root['lead_time'][:]  # in seconds (not hours!)
        init_time_array = root['init_time'][:]

        # Convert lead times from seconds to hours
        lead_time_hours = lead_time_array / 3600.0

        logger.info(f"GFS grid: {len(lat_array)} lats x {len(lon_array)} lons")
        logger.info(f"Lead times (hours): {lead_time_hours[:5]}...{lead_time_hours[-3:]} ({len(lead_time_hours)} total)")

        # Find closest available lead time
        hour_idx = int(np.argmin(np.abs(lead_time_hours - forecast_hour)))
        actual_lead = int(lead_time_hours[hour_idx])

        logger.info(f"Requested lead time: {forecast_hour}h, using: {actual_lead}h (index {hour_idx})")

        # Load global temperature data for this time step
        # Shape: (init_time, lead_time, latitude, longitude)
        temp_data = root['temperature_2m']

        # Get the full grid for the selected time
        logger.info("Loading temperature grid...")
        temps_c = temp_data[0, hour_idx, :, :]  # First init time, selected lead time

        logger.info(f"Raw temperature shape: {temps_c.shape}")
        logger.info(f"Temperature range: {np.nanmin(temps_c):.1f}°C to {np.nanmax(temps_c):.1f}°C")

        # Create 2D coordinate grids
        lons_2d, lats_2d = np.meshgrid(lon_array, lat_array)

        # Reproject to EPSG:3857
        logger.info("Reprojecting to EPSG:3857...")
        reprojected = reproject_to_web_mercator(
            np.array(temps_c), lats_2d, lons_2d,
            TEMP_BOUNDS, OUTPUT_WIDTH, OUTPUT_HEIGHT
        )

        logger.info(f"Reprojected shape: {reprojected.shape}")

        # Create PNG with color mapping
        img = create_temperature_image(reprojected)
        logger.info(f"Image created: {img.size[0]}x{img.size[1]}")

        # Compress to WebP with aggressive settings for small file size
        img_bytes = BytesIO()
        img.save(img_bytes, format='WEBP', quality=50, method=6, lossless=False)
        img_bytes.seek(0)

        compressed_size = len(img_bytes.getvalue())
        logger.info(f"Compressed to WebP: {compressed_size/1024:.1f} KB")

        # Upload to storage
        blob.upload_from_file(img_bytes, content_type='image/webp')
        blob.make_public()

        logger.info(f"Uploaded to: {blob.public_url}")

        # Get init time for metadata
        init_time = datetime.fromtimestamp(int(init_time_array[0]) / 1e9, tz=timezone.utc)
        valid_time = init_time + timedelta(hours=actual_lead)

        return {
            'url': blob.public_url,
            'cached': False,
            'forecast_hour': forecast_hour,
            'actual_lead_time': actual_lead,
            'init_time': init_time.isoformat(),
            'valid_time': valid_time.isoformat(),
            'bounds': TEMP_BOUNDS,
            'projection': 'EPSG:3857',
            'temp_range': {
                'min': float(np.nanmin(temps_c)),
                'max': float(np.nanmax(temps_c))
            }
        }

    except Exception as e:
        logger.error(f"Error generating temperature overlay: {e}", exc_info=True)
        raise


def reproject_to_web_mercator(data, lats, lons, bounds, out_width, out_height):
    """
    Reproject temperature data from EPSG:4326 to EPSG:3857 using vectorized operations.
    Handles dateline wrapping seamlessly.
    """
    from scipy import ndimage

    # Handle longitude normalization (Dynamical uses -180 to 180)
    # Check if we need to handle 0-360 format
    if lons.max() > 180:
        lons = np.where(lons > 180, lons - 360, lons)
        # Roll array to put -180 on left
        row_lons = lons[0, :]
        split_idx = np.argmax(row_lons > 180)
        if split_idx == 0:
            split_idx = len(row_lons) // 2
        data = np.roll(data, -split_idx, axis=1)
        lats = np.roll(lats, -split_idx, axis=1)
        lons = np.roll(lons, -split_idx, axis=1)

    # Flip if needed (ensure south to north)
    if lats[0, 0] > lats[-1, 0]:
        data = np.flipud(data)
        lats = np.flipud(lats)
        lons = np.flipud(lons)

    # CRITICAL: Add wrap column at dateline for seamless tiling
    # Copy the first column to the end to close the gap at 180/-180
    data = np.hstack([data, data[:, 0:1]])
    lats = np.hstack([lats, lats[:, 0:1]])
    # Extend longitude to 180 (or slightly beyond)
    last_lon = lons[:, -1:] + (lons[:, 1:2] - lons[:, 0:1])  # Add one grid spacing
    lons = np.hstack([lons, last_lon])

    src_lat_min = float(lats.min())
    src_lat_max = float(lats.max())
    src_lon_min = float(lons.min())
    src_lon_max = float(lons.max())

    logger.info(f"Source grid (with wrap): lat [{src_lat_min:.2f}, {src_lat_max:.2f}], lon [{src_lon_min:.2f}, {src_lon_max:.2f}]")

    # Create output grid in Web Mercator
    x_min, y_min = latlon_to_web_mercator(bounds['west'], bounds['south'])
    x_max, y_max = latlon_to_web_mercator(bounds['east'], bounds['north'])

    x_out = np.linspace(x_min, x_max, out_width, dtype=np.float32)
    y_out = np.linspace(y_max, y_min, out_height, dtype=np.float32)

    X_out, Y_out = np.meshgrid(x_out, y_out)

    # Convert output coords to lat/lon
    out_lons = X_out * 180.0 / 20037508.34
    out_lats = np.arctan(np.sinh(Y_out * np.pi / 20037508.34)) * 180.0 / np.pi

    # Calculate source coordinates
    lat_frac = (out_lats - src_lat_min) / (src_lat_max - src_lat_min)
    lon_frac = (out_lons - src_lon_min) / (src_lon_max - src_lon_min)

    src_rows = lat_frac * (data.shape[0] - 1)
    src_cols = lon_frac * (data.shape[1] - 1)

    # Validity mask - now covers full -180 to 180 range
    valid_mask = (
        (out_lats >= src_lat_min) & (out_lats <= src_lat_max) &
        (out_lons >= -180) & (out_lons <= 180)
    )

    src_rows = np.clip(src_rows, 0, data.shape[0] - 1)
    src_cols = np.clip(src_cols, 0, data.shape[1] - 1)

    # Bilinear interpolation using scipy
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


def create_temperature_image(temp_array):
    """Create RGBA image from temperature array with color mapping."""
    height, width = temp_array.shape
    img_array = np.zeros((height, width, 4), dtype=np.uint8)

    temp_breaks = np.array([-40, -30, -20, -10, 0, 10, 20, 25, 30, 35, 40], dtype=np.float32)
    colors = np.array([
        [0, 0, 139], [0, 0, 255], [0, 127, 255], [0, 255, 255],
        [0, 255, 127], [0, 255, 0], [127, 255, 0], [255, 255, 0],
        [255, 127, 0], [255, 0, 0], [139, 0, 0]
    ], dtype=np.float32)

    valid_mask = ~np.isnan(temp_array)
    temp_safe = np.where(valid_mask, temp_array, 0)
    temp_clamped = np.clip(temp_safe, temp_breaks[0], temp_breaks[-1])

    segment_indices = np.searchsorted(temp_breaks, temp_clamped, side='right') - 1
    segment_indices = np.clip(segment_indices, 0, len(temp_breaks) - 2)

    lower_temps = temp_breaks[segment_indices]
    upper_temps = temp_breaks[segment_indices + 1]
    t = (temp_clamped - lower_temps) / (upper_temps - lower_temps)
    t = np.clip(t, 0, 1)

    lower_colors = colors[segment_indices]
    upper_colors = colors[segment_indices + 1]

    r = (lower_colors[:, :, 0] + t * (upper_colors[:, :, 0] - lower_colors[:, :, 0])).astype(np.uint8)
    g = (lower_colors[:, :, 1] + t * (upper_colors[:, :, 1] - lower_colors[:, :, 1])).astype(np.uint8)
    b = (lower_colors[:, :, 2] + t * (upper_colors[:, :, 2] - lower_colors[:, :, 2])).astype(np.uint8)

    img_array[:, :, 0] = np.where(valid_mask, r, 0)
    img_array[:, :, 1] = np.where(valid_mask, g, 0)
    img_array[:, :, 2] = np.where(valid_mask, b, 0)
    img_array[:, :, 3] = np.where(valid_mask, 180, 0)

    return Image.fromarray(img_array, 'RGBA')


def temp_to_color(temp_c):
    """Convert temperature to RGB color (kept for compatibility)."""
    colors = [
        (-40, (0, 0, 139)), (-30, (0, 0, 255)), (-20, (0, 127, 255)),
        (-10, (0, 255, 255)), (0, (0, 255, 127)), (10, (0, 255, 0)),
        (20, (127, 255, 0)), (25, (255, 255, 0)), (30, (255, 127, 0)),
        (35, (255, 0, 0)), (40, (139, 0, 0))
    ]
    for i in range(len(colors) - 1):
        t1, c1 = colors[i]
        t2, c2 = colors[i + 1]
        if t1 <= temp_c <= t2:
            ratio = (temp_c - t1) / (t2 - t1)
            return tuple(int(c1[j] + ratio * (c2[j] - c1[j])) for j in range(3))
    return colors[0][1] if temp_c < -40 else colors[-1][1]
