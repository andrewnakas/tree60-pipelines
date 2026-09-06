"""
Legacy HRRR smoke source loaders (fallback path).

Ported from functions/hrrr_smoke.py: reads Utah's hrrrzarr S3 archive, with a
GRIB2 idx byte-range fallback from noaa-hrrr-bdp-pds. Used only when the
dynamical.org virtual store is unreachable or lagging. Smoke layers only —
the hrrrzarr sfc archive doesn't carry the reflectivity variable we render
from the virtual store, so the reflectivity layer is absent in fallback runs.
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

from render_common import SCALE_MAX_INT, SCALE_MAX_SFC

logger = logging.getLogger(__name__)


def zarr_run_exists(run_time, fs=None):
    """Check whether hrrrzarr MASSDEN data exists for a specific run."""
    date_str = run_time.strftime('%Y%m%d')
    hour_str = run_time.strftime('%H')
    check_url = (
        f"https://hrrrzarr.s3.amazonaws.com/sfc/{date_str}/"
        f"{date_str}_{hour_str}z_fcst.zarr/8m_above_ground/MASSDEN/"
        f"8m_above_ground/MASSDEN/0.0.0"
    )
    try:
        resp = requests.head(check_url, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def find_latest_zarr_run():
    """Latest HRRR-Smoke run available in the hrrrzarr S3 archive."""
    now = datetime.now(timezone.utc)
    for hours_back in range(2, 48):
        candidate = now - timedelta(hours=hours_back)
        run_time = candidate.replace(minute=0, second=0, microsecond=0)
        if zarr_run_exists(run_time):
            logger.info(f"Found legacy zarr run: {run_time.isoformat()}")
            return run_time

    logger.warning("No hrrrzarr run found, trying GRIB idx approach")
    return _find_latest_grib_run()


def _find_latest_grib_run():
    now = datetime.now(timezone.utc)
    for hours_back in range(2, 30):
        candidate = now - timedelta(hours=hours_back)
        run_time = candidate.replace(minute=0, second=0, microsecond=0)
        date_str = run_time.strftime('%Y%m%d')
        cycle = run_time.strftime('%H')
        idx_url = (
            f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
            f"hrrr.{date_str}/conus/hrrr.t{cycle}z.wrfsfcf00.grib2.idx"
        )
        try:
            resp = requests.head(idx_url, timeout=5)
            if resp.status_code == 200:
                return run_time
        except Exception:
            continue
    return (now - timedelta(hours=12)).replace(minute=0, second=0, microsecond=0)


def load_zarr_frame(run_time, fhour, layer_type='surface', fs=None, need_coords=True):
    """Load one smoke frame from hrrrzarr. Returns (data, x_coords, y_coords)
    with data flipped to north-first row order (hrrrzarr y is ascending)."""
    import s3fs
    import zarr

    if fs is None:
        fs = s3fs.S3FileSystem(anon=True)

    date_str = run_time.strftime('%Y%m%d')
    hour_str = run_time.strftime('%H')
    zarr_base = f"s3://hrrrzarr/sfc/{date_str}/{date_str}_{hour_str}z_fcst.zarr"

    if layer_type == 'integrated':
        level = 'entire_atmosphere_single_layer'
        variable = 'COLMD'
    else:
        level = '8m_above_ground'
        variable = 'MASSDEN'

    group = f"{level}/{variable}"
    data_path = f"{zarr_base}/{group}/{group}"

    try:
        x_coords = None
        y_coords = None
        if need_coords:
            x_path = f"{zarr_base}/{group}/projection_x_coordinate"
            y_path = f"{zarr_base}/{group}/projection_y_coordinate"
            try:
                x_coords = np.asarray(zarr.open(s3fs.S3Map(root=x_path, s3=fs), mode='r'))
                y_coords = np.asarray(zarr.open(s3fs.S3Map(root=y_path, s3=fs), mode='r'))
            except Exception as ce:
                logger.debug(f"coord read failed for {group}: {ce}")

        store = s3fs.S3Map(root=data_path, s3=fs)
        z = zarr.open(store, mode='r')
        data = np.array(z[fhour, :, :])
        return np.flipud(data), x_coords, y_coords
    except Exception as e:
        logger.debug(f"legacy zarr {layer_type} load failed for f{fhour:02d}: {e}")
    return None, None, None


def load_grib_frame(run_time, fhour):
    """GRIB byte-range fallback for the surface smoke layer (north-first)."""
    date_str = run_time.strftime('%Y%m%d')
    cycle = run_time.strftime('%H')
    grib_url = (
        f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
        f"hrrr.{date_str}/conus/hrrr.t{cycle}z.wrfsfcf{fhour:02d}.grib2"
    )
    idx_url = grib_url + ".idx"

    try:
        idx_resp = requests.get(idx_url, timeout=15)
        if idx_resp.status_code != 200:
            return None
    except Exception:
        return None

    lines = idx_resp.text.strip().split('\n')
    for i, line in enumerate(lines):
        parts = line.split(':')
        if len(parts) >= 5 and parts[3] == 'MASSDEN' and '8 m' in parts[4]:
            start_byte = int(parts[1])
            end_byte = int(lines[i + 1].split(':')[1]) - 1 if i + 1 < len(lines) else ''
            range_header = f"bytes={start_byte}-{end_byte}" if end_byte else f"bytes={start_byte}-"

            data_resp = requests.get(grib_url, headers={'Range': range_header}, timeout=30)
            if data_resp.status_code in (200, 206):
                data = _decode_grib_message(data_resp.content)
                if data is not None:
                    return data
            break
    return None


def _decode_grib_message(raw):
    """Decode a single GRIB2 message with gribberish (no eccodes/cfgrib needed
    in this codebase). HRRR messages scan south-to-north, so flip to
    north-first to match the rest of the pipeline."""
    try:
        import gribberish

        mapping = gribberish.parse_grib_mapping(raw)
        offset = next(iter(mapping.values()))[2] if mapping else 0
        message = gribberish.parse_grib_message(raw, offset)
        grid_shape = message.metadata.grid_shape
        data = np.asarray(message.data(), dtype='float64').reshape(grid_shape)
        return np.flipud(data)
    except Exception as e:
        logger.warning(f"gribberish decode failed: {e}")
        return None


LEGACY_SCALES = {'surface': SCALE_MAX_SFC, 'integrated': SCALE_MAX_INT}
