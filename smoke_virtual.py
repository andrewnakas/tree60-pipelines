"""
Progressive HRRR overlay pipeline (smoke surface/integrated + forecast
reflectivity) sourced from the dynamical.org virtual store.

Runs 3x/hour. Each pass is idempotent: the Firestore run manifest tracks which
leads are already rendered per layer, so a pass renders only what's new —
frames appear ~35-60 min after each 00/06/12/18Z init and backfill over the
next ticks. Falls back to the legacy hrrrzarr/GRIB source when the virtual
store is unreachable or stale.

Storage paths and the Firestore manifest/frames schema are identical to the
legacy pipeline in functions/hrrr_smoke.py, so the read endpoints in the
default codebase and the frontend animation work unchanged.
"""

import io
import logging
import time
from datetime import datetime, timedelta, timezone

import numpy as np

import legacy_smoke
import render_common
import virtual_hrrr

logger = logging.getLogger(__name__)

RUNS_COLLECTION = 'hrrr_smoke_runs'
POINT_CACHE_COLLECTION = 'hrrr_virtual_point_cache'
POINT_CACHE_PREFIX = 'hrrr_virtual/point_cache'

LAYERS = ['surface', 'integrated', 'reflectivity']

# Variables pre-extracted per lead for the merged point endpoint, with decimal
# rounding applied before compression (pressure stays in Pa; precip rate keeps
# full precision because typical values are ~1e-4 mm/s).
POINT_VARS = {
    'temperature_2m': 2,
    'precipitation_rate_surface': None,
    'total_cloud_cover_atmosphere': 1,
    'relative_humidity_2m': 1,
    'pressure_surface': 0,
    'wind_u_10m': 2,
    'wind_v_10m': 2,
    'composite_reflectivity': 1,
}
POINT_CACHE_MAX_LEAD = 6

LEASE_MINUTES = 25
# Stop starting new frames this many seconds into a pass; the next tick
# resumes from the manifest.
PASS_TIME_BUDGET = 1500
# If the virtual store's newest init is older than this, something upstream is
# wrong (normal worst-case gap is ~6h50m just before a new cycle lands).
VIRTUAL_STALE_AFTER = timedelta(hours=9, minutes=30)
KEEP_RUNS = 3


def run_render_pass(force=False, source_pref='auto', max_hours=18,
                    history_hours=18, dry_run=False):
    """One scheduler tick. source_pref: 'auto' | 'virtual' | 'legacy'."""
    import r2_firestore as fb_firestore
    import r2_storage as fb_storage

    db = fb_firestore.client()
    bucket = fb_storage.bucket()
    started = time.time()

    ds = None
    init_dt = init_idx = None
    use_virtual = source_pref != 'legacy'
    if use_virtual:
        try:
            ds = virtual_hrrr.open_dataset()
            init_dt, init_idx = virtual_hrrr.latest_init(ds)
            if datetime.now(timezone.utc) - init_dt > VIRTUAL_STALE_AFTER:
                logger.warning(f"virtual store stale (latest init {init_dt.isoformat()})")
                if source_pref != 'virtual':
                    use_virtual = False
        except Exception as e:
            logger.error(f"virtual store unavailable: {e}")
            if source_pref == 'virtual':
                raise
            use_virtual = False

    if use_virtual:
        result = _virtual_pass(db, bucket, ds, init_dt, init_idx, force,
                               max_hours, history_hours, dry_run, started)
        if result is not None:
            return result
        if source_pref == 'virtual':
            return {'status': 'error', 'error': 'no populated leads in virtual store'}
        logger.warning("virtual pass found no data; falling back to legacy source")

    return _legacy_pass(db, bucket, max_hours, history_hours, dry_run)


# ── virtual source ───────────────────────────────────────────────────────────

def _virtual_pass(db, bucket, ds, init_dt, init_idx, force, max_hours,
                  history_hours, dry_run, started):
    populated = virtual_hrrr.populated_leads(
        ds, init_idx, max_lead=48, cache_key=init_dt.isoformat())
    if not populated and init_idx > 0:
        # Newest init committed coordinates before any chunks; the previous
        # 6-hourly init is complete by now.
        init_idx -= 1
        init_dt = init_dt - timedelta(hours=virtual_hrrr.INIT_INTERVAL_H)
        populated = virtual_hrrr.populated_leads(
            ds, init_idx, max_lead=48, cache_key=init_dt.isoformat())
    if not populated:
        return None

    run_key = init_dt.strftime('%Y%m%d_%H%M')
    ref = db.collection(RUNS_COLLECTION).document(run_key)
    manifest = (ref.get().to_dict() or {}) if not dry_run else {}

    if force:
        for layer in LAYERS:
            manifest[f'rendered_leads_{layer}'] = []
        manifest['history_done'] = False
        manifest['history_frame_count'] = 0
        manifest['point_cache_leads'] = []

    if not dry_run and not _acquire_lease(ref, manifest):
        return {'status': 'skipped', 'reason': 'lease held', 'run_key': run_key}

    grid = virtual_hrrr.grid_info(ds)
    crs, transform, bounds = render_common.build_transform_from_grid(grid)
    transform_ctx = {'crs': crs, 'transform': transform, 'bounds': bounds,
                     'leaflet': None}

    rendered = {layer: set(manifest.get(f'rendered_leads_{layer}') or []) for layer in LAYERS}
    history_count = int(manifest.get('history_frame_count') or 0)
    new_frames = []

    def out_of_time():
        return time.time() - started > PASS_TIME_BUDGET

    # History first on a fresh run so the timeline is complete before the
    # 'latest' pointer flips. With 6-hourly inits, timeline hour -k comes from
    # the newest prior init at or before that valid time (lead 0-5).
    if not manifest.get('history_done') and not out_of_time():
        history_count = 0
        for k in range(1, max(history_hours, 0) + 1):
            if out_of_time():
                break
            block = (k + virtual_hrrr.INIT_INTERVAL_H - 1) // virtual_hrrr.INIT_INTERVAL_H
            src_init_dt = init_dt - timedelta(hours=virtual_hrrr.INIT_INTERVAL_H * block)
            src_lead = virtual_hrrr.INIT_INTERVAL_H * block - k
            src_idx = virtual_hrrr.init_index_for(ds, src_init_dt)
            if src_idx is None:
                continue
            for layer in LAYERS:
                frame = _render_virtual_frame(
                    ds, layer, src_idx, src_lead, -k, src_init_dt,
                    run_key, bucket, transform_ctx, dry_run)
                if frame:
                    new_frames.append(frame)
                    history_count += 1
        manifest['history_done'] = True
        manifest['history_frame_count'] = history_count

    # Forecast: only leads not yet rendered for each layer.
    for layer in LAYERS:
        todo = [l for l in populated if l <= max_hours and l not in rendered[layer]]
        for lead in todo:
            if out_of_time():
                break
            frame = _render_virtual_frame(
                ds, layer, init_idx, lead, lead, init_dt,
                run_key, bucket, transform_ctx, dry_run)
            if frame:
                new_frames.append(frame)
                rendered[layer].add(lead)

    point_cached = _update_point_cache(
        ds, init_dt, init_idx, run_key, populated, manifest, grid,
        db, bucket, dry_run, out_of_time)

    forecast_count = sum(len(v) for v in rendered.values())
    leaflet = transform_ctx.get('leaflet') or [
        [manifest.get('bounds_south', 21.5), manifest.get('bounds_west', -129.5)],
        [manifest.get('bounds_north', 52.5), manifest.get('bounds_east', -61.0)],
    ]

    manifest_update = {
        'run_time': init_dt.isoformat(),
        'run_key': run_key,
        'source': 'virtual',
        'bounds_south': leaflet[0][0],
        'bounds_west': leaflet[0][1],
        'bounds_north': leaflet[1][0],
        'bounds_east': leaflet[1][1],
        'frame_count': forecast_count + int(manifest.get('history_frame_count') or 0),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'history_done': bool(manifest.get('history_done')),
        'history_frame_count': int(manifest.get('history_frame_count') or 0),
        'point_cache_leads': point_cached,
        'lease_until': datetime.now(timezone.utc).isoformat(),  # release
    }
    for layer in LAYERS:
        manifest_update[f'rendered_leads_{layer}'] = sorted(rendered[layer])

    if not dry_run:
        ref.set(manifest_update, merge=True)
        for frame in new_frames:
            doc_id = f"{frame['layer_type']}_h{frame['fhour']:+03d}"
            ref.collection('frames').document(doc_id).set(frame)

        # Never expose a sparse timeline: flip the pointer only once history is
        # in and at least a couple of forecast hours exist.
        if manifest_update['history_done'] and len(rendered['surface']) >= 2:
            if _flip_pointer(db, run_key, init_dt):
                _cleanup_old_runs(db, bucket)

    return {
        'status': 'completed',
        'source': 'virtual',
        'run_key': run_key,
        'run_time': init_dt.isoformat(),
        'populated_leads': populated,
        'new_frames': len(new_frames),
        'frame_count': manifest_update['frame_count'],
        'point_cache_leads': point_cached,
        'elapsed_sec': round(time.time() - started, 1),
    }


def _render_virtual_frame(ds, layer, init_idx, src_lead, timeline_fhour,
                          run_time, run_key, bucket, transform_ctx, dry_run):
    var = render_common.LAYER_CONFIGS[layer]['var']
    try:
        data = virtual_hrrr.read_field(ds, var, init_idx, src_lead)
        if np.isnan(data).mean() > 0.5:
            logger.info(f"{layer} f{src_lead:02d}: chunk not decodable yet, skipping")
            return None
        return render_common.render_and_upload_frame(
            data, layer, timeline_fhour, run_time, src_lead,
            run_key, bucket, transform_ctx, dry_run=dry_run)
    except Exception as e:
        logger.warning(f"failed {layer} f{src_lead:02d} (timeline {timeline_fhour:+d}): {e}")
        return None


def _update_point_cache(ds, init_dt, init_idx, run_key, populated, manifest,
                        grid, db, bucket, dry_run, out_of_time):
    """Pre-extract point-endpoint variables per lead into compressed npz blobs.
    The point function must never decode GRIB chunks per-request (Cloud
    Functions throttle CPU outside requests, and a decode costs ~1s/field)."""
    done = set(manifest.get('point_cache_leads') or [])
    todo = [l for l in populated if l <= POINT_CACHE_MAX_LEAD and l not in done]

    for lead in todo:
        if out_of_time():
            break
        try:
            fields = {}
            for var, decimals in POINT_VARS.items():
                data = virtual_hrrr.read_field(ds, var, init_idx, lead).astype('float32')
                if np.isnan(data).mean() > 0.5:
                    # Chunk not decodable yet (e.g. mid-publish rewrite): don't
                    # cache a NaN field or the lead would never be re-extracted.
                    raise RuntimeError(f"{var} field is NaN-full")
                if decimals is not None:
                    data = np.round(data, decimals)
                fields[var] = data
            buf = io.BytesIO()
            np.savez_compressed(buf, **fields)
            buf.seek(0)
            if dry_run:
                logger.info(f"[dry-run] point cache lead {lead}: {len(buf.getvalue())} bytes")
            else:
                blob = bucket.blob(f"{POINT_CACHE_PREFIX}/{run_key}/lead{lead:02d}.npz")
                blob.upload_from_file(buf, content_type='application/octet-stream')
            done.add(lead)
        except Exception as e:
            logger.warning(f"point cache lead {lead} failed: {e}")

    if done and not dry_run:
        db.collection(POINT_CACHE_COLLECTION).document('latest').set({
            'run_key': run_key,
            'init_time': init_dt.isoformat(),
            'leads': sorted(done),
            'crs_wkt': grid['crs_wkt'],
            'geotransform': grid['geotransform'],
            'ny': grid['ny'],
            'nx': grid['nx'],
            'updated': datetime.now(timezone.utc).isoformat(),
        })
    return sorted(done)


# ── legacy source (fallback) ─────────────────────────────────────────────────

def _legacy_pass(db, bucket, max_hours, history_hours, dry_run):
    """Full render from hrrrzarr/GRIB, mirroring the old 6-hourly job. Only
    smoke layers (hrrrzarr has no composite reflectivity)."""
    import s3fs

    run_time = legacy_smoke.find_latest_zarr_run()
    run_key = run_time.strftime('%Y%m%d_%H%M')
    logger.info(f"legacy render for run {run_key}")

    try:
        fs = s3fs.S3FileSystem(anon=True)
    except Exception:
        fs = None

    transform_ctx = {'crs': None, 'transform': None, 'bounds': None, 'leaflet': None}
    frames = []
    rendered = {'surface': set(), 'integrated': set()}

    def render_one(src_run, src_fhour, timeline_fhour, layer):
        data, x_coords, y_coords = legacy_smoke.load_zarr_frame(
            src_run, src_fhour, layer_type=layer, fs=fs,
            need_coords=transform_ctx['transform'] is None)
        if data is None and layer == 'surface':
            data = legacy_smoke.load_grib_frame(src_run, src_fhour)
        if data is None:
            return None
        if transform_ctx['transform'] is None:
            if x_coords is not None and y_coords is not None:
                crs, transform, bounds = render_common.build_source_transform(x_coords, y_coords)
            else:
                crs, transform, bounds = render_common.build_static_source_transform()
            transform_ctx.update({'crs': crs, 'transform': transform, 'bounds': bounds})
        try:
            return render_common.render_and_upload_frame(
                data, layer, timeline_fhour, src_run, src_fhour,
                run_key, bucket, transform_ctx, dry_run=dry_run)
        except Exception as e:
            logger.warning(f"legacy frame {layer} f{src_fhour:02d} failed: {e}")
            return None

    for layer in ('surface', 'integrated'):
        for fhour in range(0, min(max_hours + 1, 49)):
            frame = render_one(run_time, fhour, fhour, layer)
            if frame:
                frames.append(frame)
                rendered[layer].add(fhour)

    history_count = 0
    for k in range(1, max(history_hours, 0) + 1):
        hist_run = run_time - timedelta(hours=k)
        if not legacy_smoke.zarr_run_exists(hist_run, fs):
            continue
        for layer in ('surface', 'integrated'):
            frame = render_one(hist_run, 0, -k, layer)
            if frame:
                frames.append(frame)
                history_count += 1

    if not frames:
        return {'status': 'error', 'source': 'legacy', 'error': 'no frames rendered'}

    leaflet = transform_ctx.get('leaflet') or [[21.5, -129.5], [52.5, -61.0]]
    if not dry_run:
        ref = db.collection(RUNS_COLLECTION).document(run_key)
        ref.set({
            'run_time': run_time.isoformat(),
            'run_key': run_key,
            'source': 'legacy',
            'bounds_south': leaflet[0][0],
            'bounds_west': leaflet[0][1],
            'bounds_north': leaflet[1][0],
            'bounds_east': leaflet[1][1],
            'frame_count': len(frames),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'history_done': True,
            'history_frame_count': history_count,
            'rendered_leads_surface': sorted(rendered['surface']),
            'rendered_leads_integrated': sorted(rendered['integrated']),
            'rendered_leads_reflectivity': [],
        }, merge=True)
        for frame in frames:
            doc_id = f"{frame['layer_type']}_h{frame['fhour']:+03d}"
            ref.collection('frames').document(doc_id).set(frame)
        if _flip_pointer(db, run_key, run_time):
            _cleanup_old_runs(db, bucket)

    return {'status': 'completed', 'source': 'legacy', 'run_key': run_key,
            'frame_count': len(frames)}


# ── shared helpers ───────────────────────────────────────────────────────────

def _flip_pointer(db, run_key, run_time):
    """Point 'latest' at run_key unless a fresher run is already active.

    While the old 6-hourly scheduler coexists it writes hourly-keyed runs
    (e.g. 22Z) that can postdate our 6-hourly init; never move the timeline
    backward. Stale runs (>7h old) lose this protection so the pipeline can
    always recover after the old scheduler is retired.
    """
    latest_ref = db.collection(RUNS_COLLECTION).document('latest')
    try:
        current = latest_ref.get().to_dict() or {}
        cur_time = current.get('run_time')
        cur_updated = current.get('updated')
        if cur_time and cur_updated and current.get('run_key') != run_key:
            cur_dt = datetime.fromisoformat(cur_time)
            if cur_dt.tzinfo is None:
                cur_dt = cur_dt.replace(tzinfo=timezone.utc)
            updated_dt = datetime.fromisoformat(cur_updated)
            if updated_dt.tzinfo is None:
                updated_dt = updated_dt.replace(tzinfo=timezone.utc)
            fresh = datetime.now(timezone.utc) - updated_dt < timedelta(hours=7)
            if fresh and cur_dt > run_time:
                logger.info(
                    f"not flipping pointer: active run {current.get('run_key')} "
                    f"({cur_time}) is newer than {run_key}")
                return False
    except Exception as e:
        logger.warning(f"pointer check failed (flipping anyway): {e}")

    latest_ref.set({
        'run_key': run_key,
        'run_time': run_time.isoformat(),
        'updated': datetime.now(timezone.utc).isoformat(),
    })
    return True


def _acquire_lease(ref, manifest):
    now = datetime.now(timezone.utc)
    lease = manifest.get('lease_until')
    if lease:
        try:
            if datetime.fromisoformat(lease) > now:
                logger.info(f"lease held until {lease}; skipping pass")
                return False
        except ValueError:
            pass
    ref.set({'lease_until': (now + timedelta(minutes=LEASE_MINUTES)).isoformat()},
            merge=True)
    return True


def _cleanup_old_runs(db, bucket):
    """Keep the newest KEEP_RUNS runs; delete older frames, point caches and
    manifest docs."""
    try:
        docs = [d for d in db.collection(RUNS_COLLECTION).stream() if d.id != 'latest']
        run_keys = sorted((d.id for d in docs), reverse=True)
        for old_key in run_keys[KEEP_RUNS:]:
            for prefix in (f"smoke/hrrr/{old_key}/", f"{POINT_CACHE_PREFIX}/{old_key}/"):
                for blob in bucket.list_blobs(prefix=prefix):
                    try:
                        blob.delete()
                    except Exception:
                        pass
            old_ref = db.collection(RUNS_COLLECTION).document(old_key)
            for frame_doc in old_ref.collection('frames').stream():
                frame_doc.reference.delete()
            old_ref.delete()
            logger.info(f"cleaned up run {old_key}")
    except Exception as e:
        logger.warning(f"cleanup failed: {e}")
