#!/usr/bin/env python3
"""
Global wildfire clustering from NASA FIRMS.

Replaces scheduled_global_fire_clustering (~$0.46/mo, 6-hourly on 4 GiB Cloud Run).
`global_fire_clustering.py` is copied unmodified — it has no storage dependency at
all, it just returns clusters — so only the persistence side lives here.

Publishes ONE consolidated document rather than ~200 individual ones. The old
/api/global-wildfires streamed every cluster per request and took ~15 s cold;
the edge reads a single object instead.
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
log = logging.getLogger("global_fires")

SNAPSHOT_KEY = "wildfires/global_clusters.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import json

    import r2_storage as storage
    from global_fire_clustering import fetch_and_cluster_global_fires

    clusters = fetch_and_cluster_global_fires()
    if not clusters:
        # FIRMS occasionally serves an empty or partial CSV; publishing that would
        # empty the layer, so keep whatever is already there.
        log.warning("no clusters returned — leaving the existing snapshot in place")
        return
    log.info("clustered %s global fires", len(clusters))

    now = datetime.now(timezone.utc).isoformat()
    for c in clusters:
        c["updated"] = now
        c["is_active"] = True

    if args.dry_run:
        log.info("[dry-run] would publish %s clusters", len(clusters))
        return

    payload = {
        "status": "success",
        "count": len(clusters),
        "data": clusters,
        "timestamp": now,
    }
    blob = storage.Client().bucket().blob(SNAPSHOT_KEY)
    blob.cache_control = "public, max-age=900"
    blob.upload_from_string(json.dumps(payload, default=str), content_type="application/json")
    log.info("published %s clusters -> %s", len(clusters), blob.public_url)


if __name__ == "__main__":
    main()
