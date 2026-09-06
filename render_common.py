"""
Rendering primitives shared by the virtual and legacy HRRR overlay pipelines.

Ported from functions/hrrr_smoke.py (default codebase) so the smoke frames this
codebase produces are pixel-compatible with the existing ones: same colormaps,
same LCC->EPSG:4326 warp, same Storage paths and Firestore frame docs. Adds a
fixed NWS-style colormap for the new forecast composite-reflectivity layer.
"""

import io
import logging
from datetime import timedelta

import numpy as np

logger = logging.getLogger(__name__)

# hrrrzarr's projection (legacy source path). The virtual store carries its own
# WKT + GeoTransform which differ slightly (sphere radius 6371229 vs 6371200);
# each transform must be built from the dataset it is paired with.
HRRR_PROJ4 = (
    '+proj=lcc +a=6371200.0 +b=6371200.0 '
    '+lon_0=262.5 +lat_0=38.5 +lat_1=38.5 +lat_2=38.5'
)
HRRR_NX = 1799
HRRR_NY = 1059
HRRR_DX = 3000.0
HRRR_DY = 3000.0

SCALE_MAX_SFC = 250e-9   # near-surface smoke (kg/m³)
SCALE_MAX_INT = 0.6      # vertically integrated (kg/m²)

OUT_RES = 0.05  # output resolution in degrees

# layer_type -> render parameters. 'var' is the virtual-store variable name.
# Smoke layers keep the adaptive p99.5 scaling from the legacy pipeline;
# reflectivity uses a fixed dBZ palette (adaptive scaling would make frame
# colors incomparable across an animation).
LAYER_CONFIGS = {
    'surface': {
        'var': 'mass_density_8m',
        'default_scale': SCALE_MAX_SFC,
        'noise_floor': 1e-10,
        'adaptive_min': 50e-9,
    },
    'integrated': {
        'var': 'column_integrated_mass_density_atmosphere',
        'default_scale': SCALE_MAX_INT,
        'noise_floor': 1e-6,
        'adaptive_min': 0.0005,
    },
    'reflectivity': {
        'var': 'composite_reflectivity',
        'default_scale': None,
    },
}


def smoke_rgba(data, scale_max):
    """Smoke concentration -> RGBA. Grey colormap matching HMS plume styling."""
    arr = np.nan_to_num(data.astype('float64'), nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0
    norm = np.clip(arr / scale_max, 0, 1)

    alpha = np.clip(np.power(norm, 0.65) * 255, 0, 255).astype('uint8')
    grey = np.interp(norm, [0, 0.1, 0.3, 0.6, 1.0],
                           [0, 187,  153,  119,  85]).astype('uint8')

    rgba = np.dstack([grey, grey, grey, alpha])
    rgba[alpha == 0] = 0
    return rgba


# Classic NEXRAD reflectivity palette, 5 dBZ steps from 5 to 75.
_REFL_LEVELS = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
_REFL_COLORS = np.array([
    (4, 233, 231),    # 5
    (1, 159, 244),    # 10
    (3, 0, 244),      # 15
    (2, 253, 2),      # 20
    (1, 197, 1),      # 25
    (0, 142, 0),      # 30
    (253, 248, 2),    # 35
    (229, 188, 0),    # 40
    (253, 149, 0),    # 45
    (253, 0, 0),      # 50
    (212, 0, 0),      # 55
    (188, 0, 0),      # 60
    (248, 0, 253),    # 65
    (152, 84, 198),   # 70
    (253, 253, 253),  # 75
], dtype='uint8')


def refl_rgba(data):
    """Composite reflectivity (dBZ) -> RGBA with the classic NEXRAD palette.
    Transparent below 5 dBZ (the store encodes clear sky as -10)."""
    arr = np.nan_to_num(data.astype('float64'), nan=-999.0)
    idx = np.digitize(arr, _REFL_LEVELS) - 1  # -1 => below 5 dBZ
    visible = idx >= 0
    idx_clipped = np.clip(idx, 0, len(_REFL_COLORS) - 1)

    rgba = np.zeros(arr.shape + (4,), dtype='uint8')
    rgba[..., :3] = _REFL_COLORS[idx_clipped]
    alpha = np.interp(arr, [5, 20, 50], [120, 190, 230]).astype('uint8')
    rgba[..., 3] = np.where(visible, alpha, 0)
    rgba[~visible] = 0
    return rgba


def build_transform_from_grid(grid):
    """Source transform from the virtual store's own CRS WKT + GeoTransform."""
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    gt = grid['geotransform']
    src_crs = CRS.from_wkt(grid['crs_wkt'])
    west, north = gt[0], gt[3]
    dx, dy = gt[1], abs(gt[5])
    east = west + grid['nx'] * dx
    south = north - grid['ny'] * dy
    transform = from_origin(west, north, dx, dy)
    return src_crs, transform, (west, south, east, north)


def build_source_transform(x_coords, y_coords):
    """Legacy path: transform from the hrrrzarr projection coordinate arrays
    (cell centers). See functions/hrrr_smoke.py for the alignment history."""
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    src_crs = CRS.from_proj4(HRRR_PROJ4)
    x = np.asarray(x_coords, dtype='float64')
    y = np.asarray(y_coords, dtype='float64')
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    west = float(x.min() - abs(dx) / 2.0)
    east = float(x.max() + abs(dx) / 2.0)
    south = float(y.min() - abs(dy) / 2.0)
    north = float(y.max() + abs(dy) / 2.0)
    transform = from_origin(west, north, abs(dx), abs(dy))
    return src_crs, transform, (west, south, east, north)


def build_static_source_transform():
    """Legacy GRIB-only fallback: reconstruct the grid extent from the
    canonical hrrrzarr lower-left cell center (lat 21.138, lon -122.72)."""
    from pyproj import CRS as PyCRS, Transformer
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    src_crs = CRS.from_proj4(HRRR_PROJ4)
    transformer = Transformer.from_crs('EPSG:4326', PyCRS.from_proj4(HRRR_PROJ4), always_xy=True)
    x_ll, y_ll = transformer.transform(-122.72, 21.138)
    west = x_ll - HRRR_DX / 2.0
    south = y_ll - HRRR_DY / 2.0
    east = west + HRRR_NX * HRRR_DX
    north = south + HRRR_NY * HRRR_DY
    transform = from_origin(west, north, HRRR_DX, HRRR_DY)
    return src_crs, transform, (west, south, east, north)


def warp_rgba(rgba, src_transform, src_crs, src_bounds):
    """Reproject RGBA from LCC to EPSG:4326. Returns (warped, leaflet_bounds)."""
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    height, width = rgba.shape[:2]
    left, bottom, right, top = src_bounds

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, 'EPSG:4326', width, height,
        left, bottom, right, top,
        resolution=OUT_RES,
    )

    dst = np.zeros((4, dst_height, dst_width), dtype='uint8')
    for band_idx in range(4):
        reproject(
            source=rgba[:, :, band_idx],
            destination=dst[band_idx],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs='EPSG:4326',
            resampling=Resampling.bilinear,
            src_nodata=0,
            dst_nodata=0,
        )

    import rasterio.transform
    bounds = rasterio.transform.array_bounds(dst_height, dst_width, dst_transform)
    leaflet_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
    return np.moveaxis(dst, 0, -1), leaflet_bounds


def rgba_for_layer(data, layer_type):
    """Apply the layer's colormap; smoke layers get adaptive scaling."""
    cfg = LAYER_CONFIGS[layer_type]
    if layer_type == 'reflectivity':
        return refl_rgba(data)

    scale_max = cfg['default_scale']
    significant = data[data > cfg['noise_floor']]
    if len(significant) > 100:
        p995 = np.percentile(significant, 99.5)
        scale_max = float(np.clip(p995 * 2, cfg['adaptive_min'], cfg['default_scale']))
    return smoke_rgba(data, scale_max)


def render_and_upload_frame(data, layer_type, timeline_fhour, run_time, src_fhour,
                            run_key, bucket, transform_ctx, dry_run=False):
    """Colormap + warp + PNG upload for one frame; returns the frame dict.

    transform_ctx caches {'crs','transform','bounds','leaflet'} across frames so
    every overlay in a run shares one georeference. Storage path and frame-dict
    schema are byte-identical to the legacy pipeline.
    """
    from PIL import Image

    if data.ndim > 2:
        data = data.squeeze()
    if data.ndim != 2:
        raise ValueError(f"unexpected data shape {data.shape}")

    rgba = rgba_for_layer(data, layer_type)
    warped, bounds = warp_rgba(
        rgba, transform_ctx['transform'], transform_ctx['crs'], transform_ctx['bounds']
    )
    if transform_ctx.get('leaflet') is None:
        transform_ctx['leaflet'] = bounds

    img = Image.fromarray(warped, 'RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    buf.seek(0)

    valid_time = run_time + timedelta(hours=src_fhour)
    tag = f"h{timeline_fhour:+03d}"
    blob_path = f"smoke/hrrr/{run_key}/{layer_type}/{tag}.png"

    frame = {
        'layer_type': layer_type,
        'fhour': timeline_fhour,
        'valid_time': valid_time.isoformat(),
        'url': None,
    }
    if dry_run:
        logger.info(f"[dry-run] would upload {blob_path} ({len(buf.getvalue())} bytes)")
        frame['url'] = f"dry-run://{blob_path}"
        return frame

    blob = bucket.blob(blob_path)
    blob.upload_from_file(buf, content_type='image/png')
    blob.make_public()
    frame['url'] = blob.public_url
    logger.info(f"Uploaded {layer_type} {tag} (run {run_key}, src f{src_fhour:02d})")
    return frame
