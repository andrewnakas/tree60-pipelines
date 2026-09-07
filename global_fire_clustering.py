"""
Global fire clustering using DBSCAN for fast spatial clustering.
Clusters NASA FIRMS VIIRS hotspots worldwide, excluding continental US
(which is handled by InciWeb + FIRMS fusion system).
"""

import math
import logging
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Continental US bounding box (excluded — handled by InciWeb)
US_BBOX = (24.5, -125.0, 49.0, -66.0)  # (south, west, north, east)

# Region bounding boxes for labeling
REGIONS = [
    ("Alaska", 54.0, -180.0, 72.0, -130.0),
    ("Canada", 49.0, -141.0, 83.5, -52.3),
    ("Mexico", 14.5, -118.5, 32.7, -86.7),
    ("Central America", 7.0, -92.0, 18.5, -77.0),
    ("South America", -56.0, -82.0, 13.0, -34.0),
    ("Europe", 35.0, -25.0, 71.0, 45.0),
    ("Northern Africa", 0.0, -18.0, 37.0, 55.0),
    ("Southern Africa", -35.0, 8.0, 0.0, 52.0),
    ("Central Africa", -15.0, -18.0, 15.0, 52.0),
    ("Middle East", 12.0, 25.0, 42.0, 63.0),
    ("Central Asia", 35.0, 45.0, 55.0, 90.0),
    ("South Asia", 5.0, 60.0, 37.0, 98.0),
    ("Southeast Asia", -11.0, 92.0, 28.5, 142.0),
    ("East Asia", 18.0, 73.0, 54.0, 146.0),
    ("Russia", 41.0, 19.0, 82.0, 180.0),
    ("Australia", -44.0, 112.0, -10.0, 154.0),
    ("Oceania", -47.0, 154.0, 0.0, 180.0),
]

FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/viirs/csv/J1_VIIRS_C2_Global_24h.csv"

# Earth radius in km
R_EARTH = 6371.0

# 5 km clustering radius converted to radians for haversine DBSCAN
EPS_KM = 5.0
EPS_RAD = EPS_KM / R_EARTH


def _in_us_bbox(lat, lon):
    """Check if a point falls within the continental US bounding box."""
    return (US_BBOX[0] <= lat <= US_BBOX[2] and
            US_BBOX[1] <= lon <= US_BBOX[3])


def _get_region(lat, lon):
    """Determine region label from lat/lon using bounding boxes."""
    for name, south, west, north, east in REGIONS:
        if south <= lat <= north and west <= lon <= east:
            return name
    return "Other"


def cluster_global_hotspots(firms_df, max_distance_km=5, min_samples=4):
    """
    Fast global clustering of FIRMS hotspots using scipy/sklearn DBSCAN
    with haversine metric on (lat, lon) in radians.

    1. Exclude continental US (handled by InciWeb).
    2. Run DBSCAN with eps=5km, min_samples=4.
    3. Assign region labels.
    """
    # --- vectorized filtering: exclude continental US ---
    mask = ~(
        (firms_df['latitude'] >= US_BBOX[0]) & (firms_df['latitude'] <= US_BBOX[2]) &
        (firms_df['longitude'] >= US_BBOX[1]) & (firms_df['longitude'] <= US_BBOX[3])
    )
    df = firms_df[mask].copy()
    logger.info(f"Global clustering: {len(df)} hotspots after excluding continental US")

    if df.empty:
        return []

    # --- clean numeric columns vectorized ---
    for col, default in [('frp', 0), ('confidence', 50)]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('nominal', '', regex=False)
            df[col] = df[col].str.replace(r'[^\d\.]', '', regex=True).replace('', str(default))
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)

    if 'acq_time' in df.columns:
        df['acq_time'] = df['acq_time'].astype(str).str.zfill(4)

    # --- DBSCAN clustering with haversine metric ---
    from sklearn.cluster import DBSCAN

    coords_rad = np.radians(df[['latitude', 'longitude']].values)
    eps_rad = max_distance_km / R_EARTH

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine', algorithm='ball_tree')
    df['cluster'] = db.fit_predict(coords_rad)

    # Filter out noise (cluster == -1) and small clusters
    clustered = df[df['cluster'] >= 0]
    logger.info(f"DBSCAN found {clustered['cluster'].nunique()} clusters from {len(df)} hotspots")

    # --- build output per cluster ---
    now = datetime.now(timezone.utc)
    results = []

    # Regions that use the relaxed (North American) threshold
    NA_REGIONS = {'Alaska', 'Canada'}

    for cluster_id, group in clustered.groupby('cluster'):
        count = len(group)
        lat = group['latitude'].mean()
        lon = group['longitude'].mean()
        region = _get_region(lat, lon)

        # Stricter filtering outside North America to reduce agricultural burns / gas flares
        # Canada/Alaska: 4 hotspots (same as existing system)
        # Everywhere else: 15 hotspots minimum to appear
        region_min = min_samples if region in NA_REGIONS else 50
        if count < region_min:
            continue

        total_frp = group['frp'].sum()
        avg_confidence = group['confidence'].mean()

        # Activity level
        if total_frp >= 100:
            activity_level = 'Very High'
        elif total_frp >= 50:
            activity_level = 'High'
        elif total_frp >= 20:
            activity_level = 'Medium'
        else:
            activity_level = 'Low'

        # Fire naming
        if count >= 8:
            fire_type = 'Clustered Fire'
            fire_name = f"{region} Clustered Fire"
        else:
            fire_type = 'Fire Cluster'
            fire_name = f"{region} Fire Cluster"

        fire_id = f"global_fire_{region.lower().replace(' ', '_')}_{lat:.4f}_{lon:.4f}_{count}hs_{now.strftime('%Y%m%d_%H%M')}"

        # Latest detection
        acq_key = group['acq_date'].astype(str) + group['acq_time'].astype(str)
        latest_idx = acq_key.idxmax()
        latest_row = group.loc[latest_idx]

        latest_hotspot_str = None
        detection_datetime = None
        try:
            detection_datetime = pd.to_datetime(
                str(latest_row['acq_date']) + ' ' + str(latest_row['acq_time']).zfill(4),
                format='%Y-%m-%d %H%M'
            )
            latest_hotspot_str = detection_datetime.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass

        # Hotspot coords (limit to 200 per cluster to keep Firestore doc small)
        coords_sample = group[['latitude', 'longitude', 'frp']].head(200)
        hotspot_coords = list(coords_sample.itertuples(index=False, name=None))

        incident = {
            'id': fire_id,
            'name': fire_name,
            'latitude': float(lat),
            'longitude': float(lon),
            'location': f"{region} ({lat:.4f}, {lon:.4f})",
            'state': region,
            'country': region,
            'region': region,
            'data_source': 'clustered_firms_hotspots',
            'firms_hotspots': count,
            'is_active': True,
            'confidence': round(float(avg_confidence), 1),
            'frp': round(float(total_frp), 2),
            'total_frp': round(float(total_frp), 2),
            'avg_confidence': round(float(avg_confidence), 1),
            'activity_level': activity_level,
            'cluster_size': count,
            'created_at': now.isoformat(),
            'last_updated': now.isoformat(),
            'type': fire_type,
            'detection_date': str(latest_row.get('acq_date', '')),
            'detection_time': str(latest_row.get('acq_time', '')),
            'latest_hotspot': latest_hotspot_str,
            'datetime': detection_datetime.isoformat() if detection_datetime else None,
            'hotspot_coords': str(hotspot_coords),
        }

        results.append(incident)

    logger.info(f"Global clustering complete: {len(results)} fire clusters")
    return results


def fetch_and_cluster_global_fires():
    """
    Download FIRMS CSV and run global clustering. Returns list of fire incidents.
    """
    import requests

    logger.info("Fetching NASA FIRMS global 24h data...")
    response = requests.get(FIRMS_URL, timeout=120)
    response.raise_for_status()

    firms_df = pd.read_csv(StringIO(response.text))
    logger.info(f"Retrieved {len(firms_df)} global fire hotspots from FIRMS")

    return cluster_global_hotspots(firms_df)
