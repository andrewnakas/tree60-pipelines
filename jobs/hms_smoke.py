#!/usr/bin/env python3
"""
NOAA HMS smoke plume ingest.

Replaces scheduled_scrape_smoke_data_unified (~$0.94/mo hourly on Cloud Run).
Fetches the HMS smoke KMZ, stores the plumes, and publishes the consolidated
map_cache/smoke_map document that /api/smoke/map serves in a single read.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("hms_smoke")


def build_map_cache(plumes, timestamp):
    """
    Consolidate plumes into the single FeatureCollection document the map reads.

    Mirrors the block in functions/main.py that wrote map_cache/smoke_map, so the
    served GeoJSON is unchanged. Keeping this as one document matters: the edge
    reads it in one request instead of streaming every plume.
    """
    features = []
    for plume in plumes:
        coords = []
        flat = plume.get("coordinates_flat") or []
        for i in range(0, len(flat), 2):
            if i + 1 < len(flat):
                # GeoJSON expects [lon, lat]; HMS gives lat,lon pairs.
                coords.append([flat[i + 1], flat[i]])
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "id": plume.get("id"),
                "density_category": plume.get("density_category", "Unspecified"),
                "numeric_density": plume.get("numeric_density"),
                "opacity": plume.get("opacity", 0.15),
                "description": plume.get("description", ""),
            },
        })
    return {
        "feature_collection": {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {"count": len(features), "timestamp": timestamp},
        },
        "timestamp": timestamp,
        "ttl_seconds": 1200,
    }, len(features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import hms_smoke
    import r2_firestore as firestore_shim

    plumes = hms_smoke.fetch_smoke_data()
    log.info("fetched %s smoke plumes", len(plumes))
    if not plumes:
        # HMS publishes a few times a day and can legitimately return nothing;
        # overwriting a good snapshot with an empty one would blank the layer.
        log.warning("no plumes returned — leaving the existing snapshot in place")
        return

    if args.dry_run:
        _, n = build_map_cache(plumes, datetime.now(timezone.utc).isoformat())
        log.info("[dry-run] would publish %s features", n)
        return

    hms_smoke.save_smoke_data_to_firestore_unified(plumes)

    timestamp = datetime.now(timezone.utc).isoformat()
    cache, n = build_map_cache(plumes, timestamp)
    firestore_shim.client().collection("map_cache").document("smoke_map").set(cache)
    log.info("published map_cache/smoke_map with %s features", n)


if __name__ == "__main__":
    main()
