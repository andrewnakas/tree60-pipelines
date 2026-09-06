"""
Ocean wave forecast rendering module (GFS-Wave / WW3).

Reads NOAA GFS-Wave `global.0p25` (regular lat/lon, EPSG:4326) via Herbie from
the free NOAA AWS bucket, renders significant-wave-height RGBA PNG overlays,
uploads to Firebase Cloud Storage, and writes a Firestore manifest — mirroring
the HRRR smoke pipeline (hrrr_smoke.py) so the frontend layer/timeline code is
identical in shape.

Also emits a leaflet-velocity JSON field (a swell-direction/height visual proxy)
gzipped to Cloud Storage, mirroring gfs_wind_velocity.py, for animated swell
particles.

Data source (probed 2026-07-01):
  Herbie model="gfs_wave", product="global.0p25"
  -> gfs.{YYYYMMDD}/{HH}/wave/gridded/gfswave.t{HH}z.global.0p25.f{FFF}.grib2
  Grid: 721x1440 @ 0.25 deg, lat[-90,90], lon[0,359.75] (0-360; rolled to -180/180)
  Vars (cfgrib): HTSGW->swh (m), DIRPW->dirpw (deg), PERPW->perpw (s)
"""

import io
import gzip
import json
import logging
import math
import numpy as np
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

WAVE_MODEL = "gfs_wave"
WAVE_PRODUCT = "global.0p25"

# GFS-Wave runs every 6h; render 3-hourly out to +120h.
FORECAST_STEP = 3
FORECAST_MAX_HOURS = 120

# Significant wave height (m) colormap scale. ~11 m is a strong storm sea.
WAVE_SCALE_MAX = 8.0

# Downsample factor for the leaflet-velocity swell field (0.25 deg -> 1.0 deg).
VELOCITY_TARGET_RES = 1.0

# GCS bucket (Firebase default). Matches gfs_wind_velocity.py.
BUCKET_NAME = 'tree-sixty.firebasestorage.app'


# ─────────────────────────────────────────────────────────────────────────────
# Colormap / RGBA rendering
# ─────────────────────────────────────────────────────────────────────────────

def wave_rgba(height_m, scale_max=WAVE_SCALE_MAX):
    """
    Convert significant wave height (metres) to an RGBA image.

    Colormap blue -> cyan -> green -> yellow -> orange -> red (calm -> huge).
    Power-curve alpha so flat/calm seas are near-transparent and big seas opaque,
    matching the smoke module's alpha treatment.
    """
    arr = np.nan_to_num(np.asarray(height_m, dtype='float64'), nan=0.0,
                        posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0
    norm = np.clip(arr / scale_max, 0, 1)

    # Anchor points along 0..1 for a perceptual blue->red ramp.
    stops = [0.0, 0.15, 0.35, 0.55, 0.75, 1.0]
    r_pts = [30,  40,  60, 240, 250, 200]
    g_pts = [90, 180, 200, 230, 130,  30]
    b_pts = [200, 230, 120,  70,  40,  40]

    r = np.interp(norm, stops, r_pts).astype('uint8')
    g = np.interp(norm, stops, g_pts).astype('uint8')
    b = np.interp(norm, stops, b_pts).astype('uint8')

    # Alpha: any real ocean cell should be CLEARLY visible (calm seas included),
    # not near-transparent. Small waves (e.g. 0.5-1.5 m at high latitudes) were
    # rendering at ~24% alpha, which — combined with the 0.75 layer opacity and
    # canvas upscaling that blends toward alpha-0 land pixels — made them
    # invisible against the basemap. Floor the alpha at a solid value for any
    # cell with wave data and ramp up to fully opaque for big seas.
    ALPHA_FLOOR = 170          # min opacity for the smallest real waves
    alpha = ALPHA_FLOOR + (255 - ALPHA_FLOOR) * np.power(norm, 0.5)
    alpha = np.clip(alpha, 0, 255).astype('uint8')
    # Land / no-wave cells (~0 after nan_to_num) => fully transparent.
    alpha[arr <= 0.05] = 0

    rgba = np.dstack([r, g, b, alpha])
    rgba[alpha == 0] = 0
    return rgba


# Web Mercator (EPSG:3857) valid latitude extent.
MERC_LAT_LIMIT = 85.05112878


def _warp_to_mercator(rgba, lat, lon):
    """
    Reproject an equirectangular (EPSG:4326) RGBA array to Web Mercator
    (EPSG:3857) so the PNG aligns pixel-for-pixel with the Leaflet basemap —
    the same convention the temperature/HRRR overlays use. Placing a plain
    equirectangular image via L.imageOverlay looks right near the equator but
    stretches badly toward the poles because Mercator is non-linear in latitude.

    Input rows are north-first, lon ascending -180..180. Returns
    (merc_rgba, leaflet_bounds) where leaflet_bounds = [[south,west],[north,east]]
    in degrees at the Mercator image's true edges (lat clipped to ±85.051).
    """
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS

    h, w = rgba.shape[:2]
    # Source (equirectangular) edges span the full sphere.
    src_transform = from_bounds(-180.0, -90.0, 180.0, 90.0, w, h)
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3857)

    # Destination: the canonical *square* global Web Mercator extent
    # ±20037508.342789244 m on both axes (the standard XYZ/Leaflet world). Using
    # this exact, symmetric extent — rather than a computed one — guarantees the
    # image lines up pixel-perfect with the basemap and wraps cleanly at ±180.
    MERC_MAX = 20037508.342789244
    dst_w = w
    dst_h = w  # square world
    dst_transform = from_bounds(-MERC_MAX, -MERC_MAX, MERC_MAX, MERC_MAX, dst_w, dst_h)

    dst = np.zeros((4, dst_h, dst_w), dtype='uint8')
    for band in range(4):
        reproject(
            source=rgba[:, :, band],
            destination=dst[band],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=0,
            dst_nodata=0,
        )

    # The square Mercator world maps exactly to lat ±85.0511288, lon ±180.
    leaflet_bounds = [[-MERC_LAT_LIMIT, -180.0], [MERC_LAT_LIMIT, 180.0]]

    return np.moveaxis(dst, 0, -1), leaflet_bounds


# ─────────────────────────────────────────────────────────────────────────────
# Data access (Herbie -> NOAA AWS GRIB)
# ─────────────────────────────────────────────────────────────────────────────

def _latest_run_time():
    """Most recent GFS-Wave cycle (00/06/12/18z) likely to be published.

    GFS-Wave posts a few hours after the cycle; back off 5h before rounding
    down to the cycle hour.
    """
    now = datetime.now(timezone.utc) - timedelta(hours=5)
    run_hour = (now.hour // 6) * 6
    return now.replace(hour=run_hour, minute=0, second=0, microsecond=0)


def _load_wave_field(run_time, fhour, search):
    """
    Load a single GFS-Wave field for one forecast hour via Herbie.

    Returns (data2d, lat1d, lon1d) north-first / EPSG:4326, or (None, None, None).
    cfgrib decodes this grid north-first with lon in 0-360.
    """
    try:
        from herbie import Herbie
        H = Herbie(
            run_time.strftime("%Y-%m-%d %H:00"),
            model=WAVE_MODEL,
            product=WAVE_PRODUCT,
            fxx=fhour,
        )
        ds = H.xarray(search, remove_grib=True)
        # Herbie may return a list if the search matched multiple messages.
        if isinstance(ds, list):
            ds = ds[0]
        var = list(ds.data_vars)[0]
        da = ds[var]
        data = np.asarray(da.values, dtype='float64')
        lat = np.asarray(ds['latitude'].values, dtype='float64')
        lon = np.asarray(ds['longitude'].values, dtype='float64')
        if data.ndim != 2:
            data = np.squeeze(data)
        return data, lat, lon
    except Exception as e:
        logger.warning(f"Wave field load failed f{fhour:03d} ({search}): {e}")
        return None, None, None


def _find_latest_available_run():
    """Return the newest run whose f000 HTSGW is actually retrievable.

    Tries the latest cycle, then steps back one cycle (6h) up to 3 times, so a
    just-issued but not-yet-uploaded cycle doesn't fail the whole render.
    """
    run = _latest_run_time()
    for _ in range(4):
        data, _, _ = _load_wave_field(run, 0, ":HTSGW:surface:")
        if data is not None:
            return run
        logger.info(f"Wave run {run.isoformat()} not available yet, stepping back 6h")
        run = run - timedelta(hours=6)
    return None


def _roll_to_180(data, lon):
    """Roll a 0-360 lon grid to -180..180 ascending (same as wind pipeline)."""
    if lon is None or lon.size == 0:
        return data, lon
    # Already in -180..180? nothing to do.
    if lon.max() <= 180.0:
        return data, lon
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    order = np.argsort(lon180)
    return data[:, order], lon180[order]


# ─────────────────────────────────────────────────────────────────────────────
# Height overlay frames
# ─────────────────────────────────────────────────────────────────────────────

def _render_height_frame(run_time, fhour, run_key, bucket):
    """Render + upload one wave-height PNG frame. Returns a frame dict or None."""
    from PIL import Image

    data, lat, lon = _load_wave_field(run_time, fhour, ":HTSGW:surface:")
    if data is None:
        return None

    data, lon = _roll_to_180(data, lon)

    # Ensure north-first rows so the image isn't upside down. cfgrib for this
    # product decodes lat descending (90 -> -90) already; guard anyway.
    if lat is not None and lat.size > 1 and lat[0] < lat[-1]:
        data = np.flipud(data)
        lat = lat[::-1]

    rgba = wave_rgba(data, WAVE_SCALE_MAX)

    # Reproject to Web Mercator (EPSG:3857) so the PNG aligns perfectly with the
    # Leaflet basemap — same convention as the temperature/HRRR overlays. This is
    # the fix for the "waves don't line up / drift toward the poles" issue: a raw
    # equirectangular image placed via imageOverlay is only correct at the equator.
    merc_rgba, leaflet_bounds = _warp_to_mercator(rgba, lat, lon)

    img = Image.fromarray(merc_rgba, 'RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    buf.seek(0)

    valid_time = run_time + timedelta(hours=fhour)
    tag = f"h{fhour:+04d}"
    blob_path = f"waves/{run_key}/height/{tag}.png"
    blob = bucket.blob(blob_path)
    blob.upload_from_file(buf, content_type='image/png')
    blob.make_public()

    logger.info(f"Uploaded wave height {tag} (run {run_key})")
    return {
        'layer_type': 'height',
        'fhour': fhour,
        'valid_time': valid_time.isoformat(),
        'url': blob.public_url,
    }, leaflet_bounds


# ─────────────────────────────────────────────────────────────────────────────
# Swell velocity field (leaflet-velocity JSON, for animated particles)
# ─────────────────────────────────────────────────────────────────────────────

def _downsample(data, lat, lon, target_res):
    step_lat = max(1, int(round(target_res / max(abs(lat[1] - lat[0]), 1e-6))))
    step_lon = max(1, int(round(target_res / max(abs(lon[1] - lon[0]), 1e-6))))
    return data[::step_lat, ::step_lon], lat[::step_lat], lon[::step_lon]


def _build_swell_velocity_json(run_time, fhour):
    """
    Build a leaflet-velocity [{header,data(U)},{header,data(V)}] payload from
    wave height + primary wave direction. This is a VISUAL proxy field (not a
    physical velocity): magnitude = wave height, bearing = DIRPW.

    Meteorological direction convention: DIRPW is "direction from" in degrees.
    Convert to a "toward" vector so particles flow the way the swell travels.
    """
    height, lat, native_lon = _load_wave_field(run_time, fhour, ":HTSGW:surface:")
    direction, _, _ = _load_wave_field(run_time, fhour, ":DIRPW:surface:")
    if height is None or direction is None:
        return None

    # height and direction share the identical native grid, so roll both with
    # the same native lon to keep them aligned.
    height, lon = _roll_to_180(height, native_lon)
    direction, _ = _roll_to_180(direction, native_lon)

    height = np.nan_to_num(height, nan=0.0)
    direction = np.nan_to_num(direction, nan=0.0)

    # North-first
    if lat is not None and lat.size > 1 and lat[0] < lat[-1]:
        height = np.flipud(height)
        direction = np.flipud(direction)
        lat = lat[::-1]

    height, lat_d, lon_d = _downsample(height, lat, lon, VELOCITY_TARGET_RES)
    direction, _, _ = _downsample(direction, lat, lon, VELOCITY_TARGET_RES)

    # "from" -> "toward": toward = from + 180. Vector components (met convention):
    #   u (eastward) = mag * sin(toward_rad); v (northward) = mag * cos(toward_rad)
    toward = np.deg2rad((direction + 180.0) % 360.0)
    u = height * np.sin(toward)
    v = height * np.cos(toward)

    ny, nx = u.shape
    la1, la2 = float(lat_d[0]), float(lat_d[-1])
    lo1, lo2 = float(lon_d[0]), float(lon_d[-1])
    dx = abs(lon_d[1] - lon_d[0]) if len(lon_d) > 1 else VELOCITY_TARGET_RES
    dy = abs(lat_d[1] - lat_d[0]) if len(lat_d) > 1 else VELOCITY_TARGET_RES

    base = {
        "lo1": round(lo1, 2), "la1": round(la1, 2),
        "lo2": round(lo2, 2), "la2": round(la2, 2),
        "dx": round(dx, 4), "dy": round(dy, 4),
        "nx": nx, "ny": ny,
        "refTime": run_time.isoformat(),
        "parameterUnit": "m",
        "parameterCategory": 2,
    }
    u_header = {**base, "parameterNumber": 2, "parameterNumberName": "swell_u"}
    v_header = {**base, "parameterNumber": 3, "parameterNumberName": "swell_v"}
    return [
        {"header": u_header, "data": np.round(u, 2).flatten().tolist()},
        {"header": v_header, "data": np.round(v, 2).flatten().tolist()},
    ]


def _upload_swell_velocity(run_time, fhour, bucket):
    """Build + gzip-upload the swell velocity JSON for one fhour. Returns url|None."""
    payload = _build_swell_velocity_json(run_time, fhour)
    if payload is None:
        return None
    raw = json.dumps(payload).encode('utf-8')
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode='wb') as f:
        f.write(raw)
    gz.seek(0)
    valid_time = run_time + timedelta(hours=fhour)
    blob_path = f"wave_velocity/{valid_time.strftime('%Y%m%d_%H')}.json.gz"
    blob = bucket.blob(blob_path)
    blob.content_encoding = 'gzip'
    blob.upload_from_file(gz, content_type='application/json')
    blob.make_public()
    return blob.public_url


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def render_wave_frames(max_hours=FORECAST_MAX_HOURS, step=FORECAST_STEP,
                       velocity_step=6):
    """
    Render GFS-Wave significant-height PNG overlays + swell velocity fields.

    Writes:
      GCS  waves/{run_key}/height/h{fhour:+04d}.png
      GCS  wave_velocity/{valid_time}.json.gz
      Firestore  wave_runs/{run_key} (+ frames subcollection + 'latest' pointer)
    """
    import r2_storage as fb_storage

    run_time = _find_latest_available_run()
    if run_time is None:
        logger.error("No available GFS-Wave run found")
        return None
    logger.info(f"Rendering GFS-Wave for run {run_time.isoformat()}")

    bucket = fb_storage.bucket()
    run_key = run_time.strftime('%Y%m%d_%H%M')

    fhours = list(range(0, max_hours + 1, step))
    frames = []
    leaflet_bounds = None

    for fhour in fhours:
        result = _render_height_frame(run_time, fhour, run_key, bucket)
        if not result:
            continue
        frame, bounds = result
        frames.append(frame)
        if leaflet_bounds is None:
            leaflet_bounds = bounds

    if not frames:
        logger.error("No wave frames rendered")
        return None

    # Swell velocity fields at a coarser cadence (heavier payloads).
    velocity_frames = []
    for fhour in range(0, max_hours + 1, velocity_step):
        try:
            url = _upload_swell_velocity(run_time, fhour, bucket)
            if url:
                velocity_frames.append({
                    'fhour': fhour,
                    'valid_time': (run_time + timedelta(hours=fhour)).isoformat(),
                    'url': url,
                })
        except Exception as e:
            logger.warning(f"Swell velocity f{fhour:03d} failed: {e}")

    if leaflet_bounds is None:
        leaflet_bounds = [[-90.0, -180.0], [90.0, 180.0]]

    manifest = {
        'run_time': run_time.isoformat(),
        'run_key': run_key,
        'bounds_south': leaflet_bounds[0][0],
        'bounds_west': leaflet_bounds[0][1],
        'bounds_north': leaflet_bounds[1][0],
        'bounds_east': leaflet_bounds[1][1],
        'frame_count': len(frames),
        'velocity_count': len(velocity_frames),
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    manifest_with_frames = {
        **manifest,
        'bounds': leaflet_bounds,
        'frames': frames,
        'velocity_frames': velocity_frames,
    }

    try:
        import r2_firestore as fb_firestore
        db = fb_firestore.client()

        db.collection('wave_runs').document(run_key).set(manifest)

        frames_ref = db.collection('wave_runs').document(run_key).collection('frames')
        for frame in frames:
            doc_id = f"{frame['layer_type']}_h{frame['fhour']:+04d}"
            frames_ref.document(doc_id).set(frame)

        vel_ref = db.collection('wave_runs').document(run_key).collection('velocity')
        for vf in velocity_frames:
            vel_ref.document(f"h{vf['fhour']:+04d}").set(vf)

        db.collection('wave_runs').document('latest').set({
            'run_key': run_key,
            'run_time': run_time.isoformat(),
            'updated': datetime.now(timezone.utc).isoformat(),
        })
        logger.info(f"Saved wave manifest: {len(frames)} frames, {len(velocity_frames)} velocity")
    except Exception as e:
        logger.error(f"Failed to save wave manifest: {e}")

    return manifest_with_frames


def get_wave_point(lat, lon):
    """
    Current + forecast wave conditions at a single point via the free Open-Meteo
    Marine API (no key). Returns a dict with `current` and `hourly` (trimmed), or
    None over land / on error (Open-Meteo returns no marine data on land).

    Used by the "click the ocean" popup.
    """
    import urllib.request
    import urllib.parse

    params = {
        "latitude": round(float(lat), 3),
        "longitude": round(float(lon), 3),
        "current": "wave_height,wave_direction,wave_period,"
                   "swell_wave_height,swell_wave_direction,swell_wave_period,"
                   "wind_wave_height",
        "hourly": "wave_height,wave_direction,wave_period",
        "forecast_days": 5,
        "timezone": "UTC",
    }
    url = "https://marine-api.open-meteo.com/v1/marine?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tree60weather/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"Open-Meteo marine fetch failed ({lat},{lon}): {e}")
        return None

    cur = data.get("current") or {}
    # No wave height => land (or no marine coverage). Signal "not ocean".
    if cur.get("wave_height") is None:
        return {"ocean": False}

    hourly = data.get("hourly") or {}
    times = hourly.get("time", []) or []
    wh = hourly.get("wave_height", []) or []
    wd = hourly.get("wave_direction", []) or []
    wp = hourly.get("wave_period", []) or []

    # Downsample to 3-hourly to keep the payload light for the popup chart.
    step = 3
    fc = []
    for i in range(0, len(times), step):
        if i < len(wh) and wh[i] is not None:
            fc.append({
                "time": times[i],
                "height_m": wh[i],
                "direction_deg": wd[i] if i < len(wd) else None,
                "period_s": wp[i] if i < len(wp) else None,
            })

    units = data.get("current_units", {})
    return {
        "ocean": True,
        "current": {
            "wave_height_m": cur.get("wave_height"),
            "wave_direction_deg": cur.get("wave_direction"),
            "wave_period_s": cur.get("wave_period"),
            "swell_height_m": cur.get("swell_wave_height"),
            "swell_direction_deg": cur.get("swell_wave_direction"),
            "swell_period_s": cur.get("swell_wave_period"),
            "wind_wave_height_m": cur.get("wind_wave_height"),
            "time": cur.get("time"),
        },
        "forecast": fc,
        "units": {"height": units.get("wave_height", "m"), "period": "s", "direction": "°"},
    }


def get_latest_wave_manifest():
    """Retrieve the latest wave manifest from Firestore (mirror smoke reader)."""
    import r2_firestore as fb_firestore

    db = fb_firestore.client()
    latest_doc = db.collection('wave_runs').document('latest').get()
    if not latest_doc.exists:
        return None
    run_key = latest_doc.to_dict().get('run_key')
    if not run_key:
        return None

    manifest_doc = db.collection('wave_runs').document(run_key).get()
    if not manifest_doc.exists:
        return None
    manifest = manifest_doc.to_dict()

    manifest['bounds'] = [
        [manifest.get('bounds_south', -90.0), manifest.get('bounds_west', -180.0)],
        [manifest.get('bounds_north', 90.0), manifest.get('bounds_east', 180.0)],
    ]

    frames_ref = db.collection('wave_runs').document(run_key).collection('frames')
    manifest['frames'] = [d.to_dict() for d in frames_ref.order_by('fhour').stream()]

    vel_ref = db.collection('wave_runs').document(run_key).collection('velocity')
    manifest['velocity_frames'] = [d.to_dict() for d in vel_ref.order_by('fhour').stream()]

    return manifest
