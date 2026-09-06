#!/usr/bin/env python3
"""
HRRR virtual smoke + forecast-reflectivity render pass.

This was the most expensive service left on Cloud Run (~$22.56/mo): hourly on
2 vCPU / 4 GiB, about eleven minutes per run. It is a progressive render — frames
appear as the virtual store's leads stream in after each 6-hourly HRRR init — so
it keeps running until a run is complete rather than doing one big pass.

Manifests that used to live in Firestore are now JSON objects in R2 (see
r2_firestore.py), which also means the runner needs no Firebase credentials.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("hrrr_smoke")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-render frames that already exist")
    ap.add_argument("--source", default="auto", choices=["auto", "virtual", "legacy"])
    ap.add_argument("--max-hours", type=int, default=18)
    ap.add_argument("--history-hours", type=int, default=18)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import smoke_virtual

    result = smoke_virtual.run_render_pass(
        force=args.force,
        source_pref=args.source,
        max_hours=args.max_hours,
        history_hours=args.history_hours,
        dry_run=args.dry_run,
    )
    log.info("render pass: %s", result)

    if isinstance(result, dict) and result.get("status") == "error":
        raise SystemExit(result.get("error", "render pass failed"))


if __name__ == "__main__":
    main()
