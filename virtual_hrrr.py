"""
Client for dynamical.org's low-latency virtual HRRR dataset
(noaa-hrrr-forecast-48-hour-virtual).

The dataset is an icechunk repo whose chunks are byte-range references into
NOAA's original GRIB2 files on s3://noaa-hrrr-bdp-pds, decoded on the fly by
gribberish. New forecast hours appear minutes after NOAA publishes them, so
leads for the newest init stream in progressively and must be detected.

Chunk layout is (1 init, 1 lead, full 1059x1799 CONUS field): reading any
single point still decodes a full field, so point extraction is only done
against pre-extracted caches (see point_virtual.py) — never per-request.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np

logger = logging.getLogger(__name__)

EMAIL = "treesixtyweather@gmail.com"
DATASET_ID = "noaa-hrrr-forecast-48-hour-virtual"
CANARY_VAR = "temperature_2m"  # present in every HRRR GRIB file
# Mid-CONUS grid point (Kansas-ish) used to probe whether a lead's chunk exists.
CANARY_Y, CANARY_X = 500, 900

# Snapshot of the catalog-resolved icechunk config (2026-07-19). Used as the
# last-resort open path when both the live STAC catalog and the Firestore-cached
# copy are unavailable. May go stale on a dataset version bump — the Firestore
# copy (refreshed on every successful catalog resolution) covers that case.
FALLBACK_ENTRY = {
    "id": DATASET_ID,
    "icechunk": {
        "bucket": "dynamical-noaa-hrrr",
        "prefix": "noaa-hrrr-forecast-48-hour-virtual/v0.5.0.icechunk/",
        "region": "us-west-2",
    },
    "virtual_chunk_containers": ["s3://noaa-hrrr-bdp-pds/"],
}

CONFIG_COLLECTION = "hrrr_virtual_status"
CONFIG_DOC = "config"

# HRRR runs every 6 hours; the earliest GRIB files land ~35-45 min after init.
INIT_INTERVAL_H = 6
EARLIEST_NEW_DATA_MIN = 30

_lock = threading.Lock()
_STATE = {
    "identified": False,
    "ds": None,
    "ds_opened_at": 0.0,
    "opened_via": None,       # 'catalog' | 'firestore_config' | 'fallback_entry'
    "grid": None,             # dict with crs_wkt, geotransform, ny, nx
    "leads_cache": {},        # init_iso -> (sorted lead list, checked_at)
}


def _identify_once():
    import dynamical_catalog

    if not _STATE["identified"]:
        try:
            dynamical_catalog.identify(EMAIL)
        except Exception:
            pass
        _STATE["identified"] = True


def _persist_entry(entry):
    """Best-effort: mirror the catalog-resolved icechunk config into Firestore
    so a catalog outage doesn't take down reads."""
    try:
        import r2_firestore as fb_firestore

        slim = {k: entry[k] for k in ("id", "icechunk", "virtual_chunk_containers") if k in entry}
        db = fb_firestore.client()
        doc = db.collection(CONFIG_COLLECTION).document(CONFIG_DOC)
        existing = doc.get()
        if not existing.exists or (existing.to_dict() or {}).get("entry") != slim:
            doc.set({"entry": slim, "updated": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        logger.debug(f"could not persist catalog entry: {e}")


def _load_persisted_entry():
    try:
        import r2_firestore as fb_firestore

        db = fb_firestore.client()
        doc = db.collection(CONFIG_COLLECTION).document(CONFIG_DOC).get()
        if doc.exists:
            return (doc.to_dict() or {}).get("entry")
    except Exception as e:
        logger.debug(f"could not load persisted catalog entry: {e}")
    return None


def _open_uncached():
    """Open the dataset: live catalog first, then Firestore-cached icechunk
    config, then the compiled-in fallback."""
    import dynamical_catalog
    from dynamical_catalog._open import _open_dataset

    _identify_once()
    try:
        entry = dynamical_catalog._resolve(DATASET_ID)
        ds = _open_dataset(entry, chunks=None)
        _persist_entry(entry)
        return ds, "catalog"
    except Exception as e:
        logger.warning(f"catalog open failed ({e}); trying cached icechunk config")

    for entry, via in ((_load_persisted_entry(), "firestore_config"),
                       (FALLBACK_ENTRY, "fallback_entry")):
        if not entry:
            continue
        try:
            return _open_dataset(entry, chunks=None), via
        except Exception as e:
            logger.warning(f"open via {via} failed: {e}")

    raise RuntimeError("virtual HRRR store unreachable via catalog, cached config, and fallback")


def _provably_current(ds):
    """True if no newer init can exist yet, so a re-open is pointless.
    A new 6-hourly init's first files land no sooner than ~EARLIEST_NEW_DATA_MIN
    after its cycle time."""
    try:
        last_init = ds.init_time.values[-1]
        last_init_dt = datetime.fromtimestamp(
            last_init.astype("datetime64[s]").astype(int), tz=timezone.utc
        )
        next_visible = last_init_dt + timedelta(
            hours=INIT_INTERVAL_H, minutes=EARLIEST_NEW_DATA_MIN
        )
        return datetime.now(timezone.utc) < next_visible
    except Exception:
        return False


def open_dataset(max_age=60):
    """Cached open. icechunk sessions pin a snapshot, so a cached handle never
    sees new inits/leads — re-open when stale, but skip the re-open entirely
    while the cached snapshot is provably current."""
    with _lock:
        ds = _STATE["ds"]
        age = time.time() - _STATE["ds_opened_at"]
        if ds is not None and (age < max_age or _provably_current(ds)):
            return ds
        ds, via = _open_uncached()
        _STATE["ds"] = ds
        _STATE["ds_opened_at"] = time.time()
        _STATE["opened_via"] = via
        _STATE["leads_cache"] = {}
        logger.info(f"virtual HRRR opened via {via}; latest init {ds.init_time.values[-1]}")
        return ds


def opened_via():
    return _STATE["opened_via"]


def latest_init(ds=None):
    """Return (init_dt_utc, init_idx) of the newest init_time entry."""
    if ds is None:
        ds = open_dataset()
    it = ds.init_time.values
    idx = len(it) - 1
    dt = datetime.fromtimestamp(it[idx].astype("datetime64[s]").astype(int), tz=timezone.utc)
    return dt, idx


def init_index_for(ds, init_dt):
    """Index of an exact init_time, or None."""
    target = np.datetime64(init_dt.replace(tzinfo=None), "ns")
    it = ds.init_time.values
    matches = np.where(it == target)[0]
    return int(matches[-1]) if len(matches) else None


def _lead_populated(ds, init_idx, lead):
    """Probe one lead with a single-element read. A missing chunk resolves from
    the icechunk manifest alone (no NOAA fetch) and returns NaN; a present
    chunk costs one GRIB decode."""
    try:
        v = float(
            ds[CANARY_VAR]
            .isel(init_time=init_idx, lead_time=int(lead), y=CANARY_Y, x=CANARY_X)
            .values
        )
        return not np.isnan(v)
    except Exception as e:
        logger.debug(f"lead probe failed init={init_idx} lead={lead}: {e}")
        return False


def populated_leads(ds, init_idx, max_lead=48, cache_key=None, max_age=60):
    """Sorted list of populated lead hours for an init.

    Leads publish in order, so find the frontier F = max populated lead by
    binary search (~6 probes), then report 0..F. The endpoints are verified
    directly so a fully-populated or fully-empty init costs 1-2 probes.
    """
    cache = _STATE["leads_cache"]
    if cache_key:
        hit = cache.get(cache_key)
        if hit and time.time() - hit[1] < max_age:
            return hit[0]

    if not _lead_populated(ds, init_idx, 0):
        leads = []
    elif _lead_populated(ds, init_idx, max_lead):
        leads = list(range(0, max_lead + 1))
    else:
        lo, hi = 0, max_lead  # lo populated, hi not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if _lead_populated(ds, init_idx, mid):
                lo = mid
            else:
                hi = mid
        leads = list(range(0, lo + 1))

    if cache_key:
        cache[cache_key] = (leads, time.time())
    return leads


def read_field(ds, var, init_idx, lead):
    """One chunk read -> 2D (y, x) float array, north-first row order.

    The store's y coordinate is descending (north first), verified at probe
    time; assert rather than silently rendering upside down if that changes.
    """
    y = ds.y.values
    if y[1] > y[0]:
        raise RuntimeError("virtual HRRR y-axis became ascending; orientation handling needs review")
    data = ds[var].isel(init_time=init_idx, lead_time=int(lead)).values
    return np.asarray(data)


def grid_info(ds=None):
    """CRS + affine parameters from the dataset's own spatial_ref variable."""
    if _STATE["grid"] is not None:
        return _STATE["grid"]
    if ds is None:
        ds = open_dataset()
    attrs = ds["spatial_ref"].attrs
    crs_wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
    gt = [float(v) for v in str(attrs["GeoTransform"]).split()]
    info = {
        "crs_wkt": crs_wkt,
        "geotransform": gt,  # [x_origin, dx, 0, y_origin, 0, dy] (corner-based, dy<0)
        "ny": int(ds.sizes["y"]),
        "nx": int(ds.sizes["x"]),
    }
    _STATE["grid"] = info
    return info


def latlon_to_index(lat, lon, grid):
    """(lat, lon) -> (y_idx, x_idx) on the native LCC grid via the dataset's
    own CRS; returns None if outside the grid."""
    from pyproj import CRS, Transformer

    transformer = Transformer.from_crs(
        "EPSG:4326", CRS.from_wkt(grid["crs_wkt"]), always_xy=True
    )
    x_m, y_m = transformer.transform(lon, lat)
    gt = grid["geotransform"]
    x_idx = int(round((x_m - gt[0]) / gt[1] - 0.5))
    y_idx = int(round((y_m - gt[3]) / gt[5] - 0.5))
    if 0 <= y_idx < grid["ny"] and 0 <= x_idx < grid["nx"]:
        return y_idx, x_idx
    return None
