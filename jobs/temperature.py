#!/usr/bin/env python3
"""
Temperature overlay pipeline.

  tiles    regenerate the -18..+18 h tile set (after each GFS run)
  archive  copy the current hour into the rolling history

Ran on Cloud Run at 2 GiB with a 3600 s timeout; runs here on a GitHub Actions
runner (4 vCPU / 16 GB) and writes to R2.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("temperature")


def _write_manifest(result):
    """
    Publish hour -> tile URL so the edge can serve the overlay without knowing how
    tile keys are derived from GFS run times.
    """
    import json
    from datetime import datetime, timezone

    import r2_storage as storage

    hours = {
        str(item["hour"]): {"url": item["url"], "valid_time": item.get("valid_time")}
        for item in result["results"]["success"]
        if item.get("url")
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hours": hours,
        "count": len(hours),
    }
    blob = storage.Client().bucket().blob("temperature_tiles/manifest.json")
    blob.cache_control = "public, max-age=300"
    blob.upload_from_string(json.dumps(manifest), content_type="application/json")
    log.info("manifest: %s hours -> %s", len(hours), blob.public_url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["tiles", "archive"])
    args = ap.parse_args()

    if args.task == "tiles":
        from temp_tile_scheduler import regenerate_all_temperature_tiles

        result = regenerate_all_temperature_tiles()
        _write_manifest(result)
        log.info(
            "tiles: %s ok, %s failed of %s",
            result["success_count"],
            result["failed_count"],
            result["total"],
        )
        for f in result["results"]["failed"]:
            log.error("  hour %+d: %s", f["hour"], f["error"])
        # A couple of gaps are tolerable; a mostly-empty run is a real failure and
        # should fail the workflow loudly rather than leave a half-updated overlay.
        if result["success_count"] < result["total"] * 0.75:
            raise SystemExit(
                f"too many tile failures: {result['failed_count']}/{result['total']}"
            )

    elif args.task == "archive":
        from temp_overlay_simple import archive_current_temperature

        result = archive_current_temperature()
        log.info("archive: %s", result)
        if isinstance(result, dict) and result.get("error"):
            raise SystemExit(result["error"])


if __name__ == "__main__":
    main()
