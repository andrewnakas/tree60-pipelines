#!/usr/bin/env python3
"""
Ocean wave forecast render (GFS-Wave via Herbie -> NOAA AWS GRIB).

Ran 6-hourly on Cloud Run at 8 GiB (~$1.95/mo). Produces the wave-height frames
and the swell velocity field, plus the run manifest that the map's timeline reads.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("waves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=int, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--velocity-step", type=int, default=6)
    args = ap.parse_args()

    from wave_forecast import render_wave_frames

    kwargs = {"velocity_step": args.velocity_step}
    if args.max_hours is not None:
        kwargs["max_hours"] = args.max_hours
    if args.step is not None:
        kwargs["step"] = args.step

    manifest = render_wave_frames(**kwargs)
    log.info("wave render: %s", manifest)

    if not manifest:
        raise SystemExit("wave render produced no manifest")
    if isinstance(manifest, dict) and manifest.get("status") == "error":
        raise SystemExit(manifest.get("error", "wave render failed"))


if __name__ == "__main__":
    main()
