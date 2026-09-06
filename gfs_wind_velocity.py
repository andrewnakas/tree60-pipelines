"""
GFS Wind Velocity API - Returns wind data in leaflet-velocity JSON format
Supports both historical (-18h to 0) and forecast (0 to +18h) hours
Supports 10m surface winds and 100m upper-level winds

Data is fetched from Dynamical.org GFS zarr endpoint and formatted for
the leaflet-velocity library which expects u/v component arrays.

Output is compressed JSON with reduced precision for fast loading.
"""
import logging
import numpy as np
import zarr
import fsspec
import json
import gzip
from io import BytesIO
from datetime import datetime, timedelta, timezone
# Ported from Firebase: R2 stands in for google.cloud.storage. See r2_storage.py.
import r2_storage as storage

logger = logging.getLogger(__name__)

# Dynamical.org GFS endpoint
GFS_ZARR_URL = "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr"

# Storage bucket
BUCKET_NAME = 'tree-sixty.firebasestorage.app'

# Velocity data folders
VELOCITY_FOLDER_10M = 'wind_velocity_10m'
VELOCITY_FOLDER_100M = 'wind_velocity_100m'
VELOCITY_HISTORY_10M = 'wind_velocity_history_10m'
VELOCITY_HISTORY_100M = 'wind_velocity_history_100m'
MAX_HISTORY_HOURS = 24

# Output resolution - lower = smaller files, faster loading
# GFS native is 0.25 deg, we'll use 1.0 deg for fast loading
OUTPUT_RESOLUTION = 1.0  # degrees


def get_folders_for_level(level='10m'):
    """Get the velocity and history folders for a wind level."""
    if level == '100m':
        return VELOCITY_FOLDER_100M, VELOCITY_HISTORY_100M
    else:
        return VELOCITY_FOLDER_10M, VELOCITY_HISTORY_10M


def get_history_blob_name(valid_time, level='10m'):
    """Get the blob name for a historical archive."""
    _, history_folder = get_folders_for_level(level)
    return f"{history_folder}/{valid_time.strftime('%Y%m%d_%H')}.json.gz"


def downsample_data(data, lat_array, lon_array, target_resolution=1.0):
    """
    Downsample wind data to target resolution for smaller file size.

    Args:
        data: 2D numpy array (lat x lon)
        lat_array: 1D array of latitudes
        lon_array: 1D array of longitudes
        target_resolution: target grid spacing in degrees

    Returns:
        downsampled_data, new_lat_array, new_lon_array
    """
    # Calculate stride based on current and target resolution
    current_lat_res = abs(lat_array[1] - lat_array[0]) if len(lat_array) > 1 else 0.25
    current_lon_res = abs(lon_array[1] - lon_array[0]) if len(lon_array) > 1 else 0.25

    lat_stride = max(1, int(round(target_resolution / current_lat_res)))
    lon_stride = max(1, int(round(target_resolution / current_lon_res)))

    # Downsample
    new_data = data[::lat_stride, ::lon_stride]
    new_lat = lat_array[::lat_stride]
    new_lon = lon_array[::lon_stride]

    return new_data, new_lat, new_lon


def format_velocity_json(u_data, v_data, lat_array, lon_array, ref_time, level='10m'):
    """
    Format wind data in leaflet-velocity JSON format.

    The format expected by leaflet-velocity is:
    [
        {
            "header": {
                "parameterCategory": 2,
                "parameterNumber": 2,  // U component
                "lo1": first_lon,
                "la1": first_lat,
                "lo2": last_lon,
                "la2": last_lat,
                "dx": lon_spacing,
                "dy": lat_spacing,
                "nx": num_lon,
                "ny": num_lat,
                "refTime": "ISO timestamp"
            },
            "data": [flattened u values...]
        },
        {
            "header": { ... parameterNumber: 3 for V ... },
            "data": [flattened v values...]
        }
    ]

    Note: leaflet-velocity expects longitude in 0-360 format and data ordered
    west-to-east starting from lon 0.
    """
    # Convert longitude from -180,180 to 0,360 format if needed
    lon_360 = lon_array.copy()
    if lon_array[0] < 0:
        # Need to convert -180,180 to 0,360
        lon_360 = np.where(lon_array < 0, lon_array + 360, lon_array)

        # Find index where original array crossed 0 (becomes 360 after transform)
        # We need to reorder: shift data so lon 0 is first instead of lon 180
        split_idx = np.searchsorted(lon_array, 0)

        # Reorder longitude array to 0,360 (ascending)
        lon_360 = np.concatenate([lon_360[split_idx:], lon_360[:split_idx]])

        # Reorder data arrays along longitude axis (axis=1)
        u_data = np.concatenate([u_data[:, split_idx:], u_data[:, :split_idx]], axis=1)
        v_data = np.concatenate([v_data[:, split_idx:], v_data[:, :split_idx]], axis=1)

        logger.info(f"Converted lon from -180,180 to 0,360. New range: {lon_360[0]:.1f} to {lon_360[-1]:.1f}")

    # Get grid parameters
    ny, nx = u_data.shape
    lo1 = float(lon_360[0])
    lo2 = float(lon_360[-1])
    la1 = float(lat_array[0])  # Usually 90 (north)
    la2 = float(lat_array[-1])  # Usually -90 (south)

    # Calculate spacing
    dx = abs(lon_360[1] - lon_360[0]) if len(lon_360) > 1 else 1.0
    dy = abs(lat_array[1] - lat_array[0]) if len(lat_array) > 1 else 1.0

    # Handle NaN values - replace with 0
    u_clean = np.nan_to_num(u_data, nan=0.0)
    v_clean = np.nan_to_num(v_data, nan=0.0)

    # Round to 2 decimal places for smaller file size
    u_rounded = np.round(u_clean, 2)
    v_rounded = np.round(v_clean, 2)

    # Flatten to 1D array (row-major order: west-to-east, north-to-south)
    u_flat = u_rounded.flatten().tolist()
    v_flat = v_rounded.flatten().tolist()

    # Create header
    base_header = {
        "lo1": round(lo1, 2),
        "la1": round(la1, 2),
        "lo2": round(lo2, 2),
        "la2": round(la2, 2),
        "dx": round(dx, 4),
        "dy": round(dy, 4),
        "nx": nx,
        "ny": ny,
        "refTime": ref_time.isoformat() if ref_time else datetime.now(timezone.utc).isoformat(),
        "parameterUnit": "m/s",
        "parameterNumberName": f"wind_{level}"
    }

    # U component
    u_header = base_header.copy()
    u_header["parameterCategory"] = 2
    u_header["parameterNumber"] = 2
    u_header["parameterNumberName"] = f"eastward_wind_{level}"

    # V component
    v_header = base_header.copy()
    v_header["parameterCategory"] = 2
    v_header["parameterNumber"] = 3
    v_header["parameterNumberName"] = f"northward_wind_{level}"

    return [
        {"header": u_header, "data": u_flat},
        {"header": v_header, "data": v_flat}
    ]


def fetch_gfs_wind_velocity(forecast_hour=0, level='10m', email="tree60weather@gmail.com"):
    """
    Fetch GFS wind velocity data from Dynamical.org.

    Args:
        forecast_hour: Hour offset (-18 to +18)
        level: '10m' or '100m'
        email: Email for Dynamical.org usage tracking

    Returns:
        dict with velocity JSON data, or None on failure
    """
    try:
        now = datetime.now(timezone.utc)

        # Clamp to valid range
        forecast_hour = max(-18, min(18, forecast_hour))

        logger.info(f"Fetching GFS {level} wind velocity for hour {forecast_hour}")

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

        # For historical hours, we need archived data
        # For now, use closest available forecast hour (may not be exact for history)
        if forecast_hour < 0:
            # Try to get archived data first
            archived = get_archived_velocity(abs(forecast_hour), level)
            if archived:
                return archived
            # Fall back to hour 0 if no archive
            hour_idx = 0
            logger.info(f"No archived data for hour {forecast_hour}, using hour 0")
        else:
            hour_idx = int(np.argmin(np.abs(lead_time_hours - forecast_hour)))

        actual_lead = int(lead_time_hours[hour_idx])

        # Load U and V wind components based on level
        if level == '100m':
            wind_u = np.array(root['wind_u_100m'][0, hour_idx, :, :])
            wind_v = np.array(root['wind_v_100m'][0, hour_idx, :, :])
        else:
            wind_u = np.array(root['wind_u_10m'][0, hour_idx, :, :])
            wind_v = np.array(root['wind_v_10m'][0, hour_idx, :, :])

        # Downsample for smaller file size
        u_down, lat_down, lon_down = downsample_data(wind_u, lat_array, lon_array, OUTPUT_RESOLUTION)
        v_down, _, _ = downsample_data(wind_v, lat_array, lon_array, OUTPUT_RESOLUTION)

        # Get reference time
        init_time = datetime.fromtimestamp(int(init_time_array[-1]), tz=timezone.utc)
        valid_time = init_time + timedelta(hours=actual_lead)

        # Format for leaflet-velocity
        velocity_data = format_velocity_json(u_down, v_down, lat_down, lon_down, valid_time, level)

        return {
            'data': velocity_data,
            'forecast_hour': forecast_hour,
            'actual_lead_time': actual_lead,
            'init_time': init_time.isoformat(),
            'valid_time': valid_time.isoformat(),
            'level': level,
            'resolution': OUTPUT_RESOLUTION,
            'grid_size': {'nx': len(lon_down), 'ny': len(lat_down)}
        }

    except Exception as e:
        logger.error(f"Error fetching GFS wind velocity: {e}", exc_info=True)
        return None


def get_archived_velocity(hours_ago, level='10m'):
    """
    Get archived velocity data from storage.

    Args:
        hours_ago: Positive integer (1-18)
        level: '10m' or '100m'

    Returns:
        dict with velocity data, or None if not available
    """
    try:
        hours_ago = abs(int(hours_ago))
        if hours_ago < 1 or hours_ago > 18:
            return None

        now = datetime.now(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        target_time = current_hour - timedelta(hours=hours_ago)

        logger.info(f"Looking for archived {level} velocity: {hours_ago}h ago = {target_time.isoformat()}")

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        archive_blob_name = get_history_blob_name(target_time, level)
        archive_blob = bucket.blob(archive_blob_name)

        if archive_blob.exists():
            logger.info(f"Found archived velocity: {archive_blob_name}")

            # Download and decompress
            compressed_data = archive_blob.download_as_bytes()
            json_data = gzip.decompress(compressed_data).decode('utf-8')
            velocity_data = json.loads(json_data)

            return {
                'data': velocity_data,
                'forecast_hour': -hours_ago,
                'valid_time': target_time.isoformat(),
                'level': level,
                'source': 'archive',
                'cached': True
            }
        else:
            logger.info(f"No archived velocity for {target_time.isoformat()}")
            return None

    except Exception as e:
        logger.error(f"Error getting archived velocity: {e}", exc_info=True)
        return None


def archive_current_velocity(level='10m'):
    """
    Archive the current (hour 0) velocity data.
    Called hourly to build up historical data.

    Args:
        level: '10m' or '100m'

    Returns:
        dict with archive info, or None if failed
    """
    try:
        now = datetime.now(timezone.utc)
        valid_time = now.replace(minute=0, second=0, microsecond=0)

        logger.info(f"Archiving current {level} velocity for {valid_time.isoformat()}")

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        # Check if already archived
        archive_blob_name = get_history_blob_name(valid_time, level)
        archive_blob = bucket.blob(archive_blob_name)

        if archive_blob.exists():
            logger.info(f"{level} velocity archive already exists for {valid_time.isoformat()}")
            return {
                'status': 'already_exists',
                'valid_time': valid_time.isoformat(),
                'level': level
            }

        # Fetch current velocity data
        result = fetch_gfs_wind_velocity(forecast_hour=0, level=level)

        if not result or not result.get('data'):
            raise Exception("Failed to fetch velocity data")

        # Compress and upload
        json_str = json.dumps(result['data'], separators=(',', ':'))  # Compact JSON
        compressed = gzip.compress(json_str.encode('utf-8'))

        archive_blob.upload_from_string(compressed, content_type='application/gzip')
        archive_blob.make_public()

        # Cleanup old archives
        cleanup_old_velocity_archives(storage_client, bucket, level)

        logger.info(f"Archived {level} velocity: {archive_blob_name} ({len(compressed)} bytes)")

        return {
            'status': 'archived',
            'valid_time': valid_time.isoformat(),
            'level': level,
            'size_bytes': len(compressed)
        }

    except Exception as e:
        logger.error(f"Error archiving velocity: {e}", exc_info=True)
        return None


def cleanup_old_velocity_archives(storage_client, bucket, level='10m'):
    """Remove archived velocity data older than MAX_HISTORY_HOURS."""
    try:
        _, history_folder = get_folders_for_level(level)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=MAX_HISTORY_HOURS)

        blobs = bucket.list_blobs(prefix=f"{history_folder}/")
        deleted_count = 0

        for blob in blobs:
            try:
                filename = blob.name.split('/')[-1].replace('.json.gz', '')
                blob_time = datetime.strptime(filename, '%Y%m%d_%H').replace(tzinfo=timezone.utc)

                if blob_time < cutoff_time:
                    blob.delete()
                    deleted_count += 1
                    logger.info(f"Deleted old velocity archive: {blob.name}")
            except (ValueError, IndexError):
                continue

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old {level} velocity archive(s)")

    except Exception as e:
        logger.error(f"Error cleaning up velocity archives: {e}", exc_info=True)


def list_available_velocity_history(level='10m'):
    """
    List all available historical hours for velocity data.

    Returns:
        List of available hours_ago values (e.g., [1, 2, 3, ...])
    """
    try:
        now = datetime.now(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        _, history_folder = get_folders_for_level(level)

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        available_hours = []
        blobs = bucket.list_blobs(prefix=f"{history_folder}/")

        for blob in blobs:
            try:
                filename = blob.name.split('/')[-1].replace('.json.gz', '')
                blob_time = datetime.strptime(filename, '%Y%m%d_%H').replace(tzinfo=timezone.utc)
                hours_ago = int((current_hour - blob_time).total_seconds() / 3600)

                if 1 <= hours_ago <= 18:
                    available_hours.append(hours_ago)
            except (ValueError, IndexError):
                continue

        return sorted(available_hours)

    except Exception as e:
        logger.error(f"Error listing velocity history: {e}", exc_info=True)
        return []


def get_gfs_wind_velocity(forecast_hour=0, level='10m', email="tree60weather@gmail.com"):
    """
    Main entry point - get wind velocity data for specified hour.

    For forecast hours (0 to +18): Fetches from GFS
    For historical hours (-18 to -1): Uses archived data

    Args:
        forecast_hour: Hour offset (-18 to +18)
        level: '10m' or '100m'
        email: Email for Dynamical.org

    Returns:
        dict with velocity data and metadata
    """
    try:
        forecast_hour = max(-18, min(18, forecast_hour))

        logger.info(f"Wind velocity request: hour={forecast_hour}, level={level}")

        # Historical: check archive first
        if forecast_hour < 0:
            hours_ago = abs(forecast_hour)
            archived = get_archived_velocity(hours_ago, level)

            if archived:
                return archived
            else:
                # No archived data - return unavailable
                logger.info(f"No archived velocity for hour {forecast_hour}")
                return {
                    'status': 'unavailable',
                    'forecast_hour': forecast_hour,
                    'level': level,
                    'message': f'Historical data for {hours_ago}h ago not yet available',
                    'available_history': list_available_velocity_history(level)
                }

        # Forecast/current: fetch from GFS
        result = fetch_gfs_wind_velocity(forecast_hour, level, email)

        if result:
            return result
        else:
            return {
                'status': 'error',
                'forecast_hour': forecast_hour,
                'level': level,
                'message': 'Failed to fetch GFS wind data'
            }

    except Exception as e:
        logger.error(f"Error in get_gfs_wind_velocity: {e}", exc_info=True)
        return {
            'status': 'error',
            'forecast_hour': forecast_hour,
            'level': level,
            'message': str(e)
        }
