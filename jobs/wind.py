#!/usr/bin/env python3
"""
Wind overlay pipelines.

  tiles     regenerate the wind-speed tile set after each GFS run
  archive   copy the current hour into the rolling wind history
  velocity  archive the leaflet-velocity particle field

All three ran on Cloud Run at 2 GiB (about $5.20/mo between them) doing the same
GFS Zarr read the temperature job does. Both levels (10 m and 100 m) are handled
in one invocation, as the schedulers did.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("wind")

LEVELS = ("10m", "100m")


def _render_tiles():
    """
    Render forecast tiles (hours 0..18) for both levels.

    The Cloud Run job only did 10 m; 100 m was rendered on demand by
    get_wind_png, which the edge cannot do. Pre-rendering both means the 100 m
    overlay is warm instead of paying a cold GFS read on first view.
    Negative hours are not rendered here — they come from the archive task.
    """
    from wind_overlay_simple import generate_wind_image

    by_level = {}
    ok = bad = 0
    for level in LEVELS:
        hours = {}
        for hour in range(0, 19):
            try:
                result = generate_wind_image(forecast_hour=hour, level=level)
                if result and result.get("url"):
                    hours[str(hour)] = {"url": result["url"]}
                    ok += 1
                else:
                    bad += 1
                    log.error("  %s hour %+d: generation returned nothing", level, hour)
            except Exception as e:
                bad += 1
                log.error("  %s hour %+d: %s", level, hour, e)
        by_level[level] = hours
        log.info("tiles %s: %s hours", level, len(hours))
    return by_level, ok, bad


def _write_tile_manifest(by_level):
    """Publish level -> hour -> tile URL for the edge to serve."""
    import json
    from datetime import datetime, timezone

    import r2_storage as storage

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "levels": by_level,
        "count": sum(len(v) for v in by_level.values()),
    }
    blob = storage.Client().bucket().blob("wind_tiles/manifest.json")
    blob.cache_control = "public, max-age=300"
    blob.upload_from_string(json.dumps(manifest), content_type="application/json")
    log.info("manifest: %s tiles -> %s", manifest["count"], blob.public_url)


def _run_levels(fn, label):
    """Run fn for both levels, tolerating one level failing."""
    results, failed = {}, []
    for level in LEVELS:
        try:
            results[level] = fn(level=level)
            log.info("%s %s: %s", label, level, results[level])
        except Exception as e:
            log.error("%s %s FAILED: %s", label, level, e, exc_info=True)
            failed.append(level)
    if len(failed) == len(LEVELS):
        raise SystemExit(f"{label}: both levels failed")
    if failed:
        log.warning("%s: %s failed but %s succeeded", label, failed,
                    [l for l in LEVELS if l not in failed])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["tiles", "archive", "velocity"])
    args = ap.parse_args()

    if args.task == "tiles":
        by_level, ok, bad = _render_tiles()
        log.info("tiles: %s ok, %s failed", ok, bad)
        _write_tile_manifest(by_level)
        if ok + bad and ok < (ok + bad) * 0.75:
            raise SystemExit(f"too many wind tile failures: {bad}/{ok + bad}")

    elif args.task == "archive":
        from wind_overlay_simple import archive_current_wind

        _run_levels(archive_current_wind, "wind archive")

    elif args.task == "velocity":
        from gfs_wind_velocity import archive_current_velocity

        _run_levels(archive_current_velocity, "velocity archive")


if __name__ == "__main__":
    main()
