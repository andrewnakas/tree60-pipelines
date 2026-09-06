"""
NOAA HMS smoke plume ingest.

Extracted verbatim from functions/main.py (fetch_smoke_data +
save_smoke_data_to_firestore_unified) so it can run on a GitHub Actions runner
instead of an hourly Cloud Run container. The only changes are the imports and
swapping Firestore for r2_firestore.

Source: NOAA HMS smoke KMZ (a zipped KML of plume polygons), parsed into a
GeoJSON FeatureCollection plus a consolidated map_cache document that the edge
serves in one read.
"""

import json
import logging
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from io import BytesIO

import requests

import r2_firestore as firestore_shim

logger = logging.getLogger(__name__)

HMS_SMOKE_KMZ_URL = "https://www.ospo.noaa.gov/data/spl/kmlfiles/fire/smoke.kmz"
SMOKE_OPACITY = {
    "Light": 0.1,
    "Medium": 0.2,
    "Heavy": 0.4,
    "Unspecified": 0.15,
}


def get_db_client():
    """Stands in for main.get_db_client(); r2_firestore speaks the same API."""
    return firestore_shim.client()


def fetch_smoke_data():
    """Fetch and process smoke plume data from NOAA HMS."""
    try:
        logger.info("Fetching latest HMS smoke data from KMZ...")
        
        # Set longer timeout and headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(HMS_SMOKE_KMZ_URL, timeout=60, headers=headers)
        response.raise_for_status()
        
        logger.info(f"Received KMZ response, length: {len(response.content)} bytes")
        
        if len(response.content) < 100:
            logger.error("KMZ file appears to be empty or too small")
            return []
        
        smoke_plumes = []
        
        # Process KMZ file (compressed KML)
        try:
            with zipfile.ZipFile(BytesIO(response.content)) as kmz:
                kml_filename = None
                # Find the first KML file in the KMZ
                for filename in kmz.namelist():
                    logger.info(f"Found file in KMZ: {filename}")
                    if filename.endswith('.kml'):
                        kml_filename = filename
                        break
                
                if not kml_filename:
                    logger.error(f"No KML file found in KMZ archive. Files found: {kmz.namelist()}")
                    return []
                
                # Extract and parse the KML
                with kmz.open(kml_filename) as kml_file:
                    kml_content = kml_file.read()
                    logger.info(f"Extracted KML file: {kml_filename}, size: {len(kml_content)} bytes")
        except zipfile.BadZipFile as e:
            logger.error(f"Bad ZIP file: {e}")
            logger.info("Attempting to parse as raw KML...")
            kml_content = response.content
        
        try:
            root = ET.fromstring(kml_content)
            logger.info("Successfully parsed XML")
        except ET.ParseError as e:
            logger.error(f"XML Parse Error: {e}")
            logger.info(f"First 500 chars of content: {kml_content[:500]}")
            return []
            
        # Try different namespace patterns
        namespaces = [
            {'kml': 'http://www.opengis.net/kml/2.2'},
            {'kml': 'http://earth.google.com/kml/2.2'},
            {'kml': 'http://earth.google.com/kml/2.1'},
            {}  # No namespace
        ]
        
        placemarks = []
        for ns in namespaces:
            if ns:
                placemarks = root.findall('.//kml:Placemark', ns)
                if placemarks:
                    logger.info(f"Found {len(placemarks)} placemarks using namespace {ns}")
                    break
            else:
                placemarks = root.findall('.//Placemark')
                if placemarks:
                    logger.info(f"Found {len(placemarks)} placemarks without namespace")
                    break
        
        if not placemarks:
            logger.error("No placemarks found in KML with any namespace")
            # Let's see what elements we do have
            all_elements = [elem.tag for elem in root.iter()]
            logger.info(f"All elements in XML: {set(all_elements)}")
            return []
        
        logger.info(f"Processing {len(placemarks)} placemarks")
        
        # Use the working namespace
        working_ns = namespaces[0] if placemarks else {}
        
        for i, placemark in enumerate(placemarks[:50]):  # Limit to first 50 for timeout prevention
            try:
                # Default values
                style_url = "None"
                numeric_density = None
                density_category = "Unspecified"
                
                # Get style URL to determine density
                if working_ns:
                    style_url_elem = placemark.find('./kml:styleUrl', working_ns)
                else:
                    style_url_elem = placemark.find('.//styleUrl')
                    
                if style_url_elem is not None and style_url_elem.text:
                    style_url = style_url_elem.text
                    
                    if "Light" in style_url or "light" in style_url:
                        density_category = "Light"
                    elif "Medium" in style_url or "medium" in style_url:
                        density_category = "Medium"
                    elif "Heavy" in style_url or "heavy" in style_url:
                        density_category = "Heavy"
                
                # Get description for numeric density
                if working_ns:
                    desc_elem = placemark.find('./kml:description', working_ns)
                else:
                    desc_elem = placemark.find('.//description')
                    
                description = desc_elem.text if desc_elem is not None and desc_elem.text else ""
                
                # Try to extract numeric density value from description
                if description and "Density:" in description:
                    try:
                        lines = description.split('\n')
                        for line in lines:
                            if "Density:" in line:
                                numeric_density = int(line.split(':')[1].strip())
                                break
                    except (ValueError, IndexError):
                        pass
                
                # Get coordinates
                if working_ns:
                    coords_elem = placemark.find('.//kml:coordinates', working_ns)
                else:
                    coords_elem = placemark.find('.//coordinates')
                    
                if coords_elem is None or not coords_elem.text:
                    continue
                    
                coords_text = coords_elem.text.strip()
                coordinates = []
                
                # Parse coordinates more robustly
                coord_parts = coords_text.replace('\n', ' ').replace('\t', ' ').split()
                for coord_str in coord_parts:
                    if ',' in coord_str:
                        parts = coord_str.split(',')
                        if len(parts) >= 2:
                            try:
                                lon, lat = float(parts[0]), float(parts[1])
                                # Basic validation for reasonable coordinates
                                if -180 <= lon <= 180 and -90 <= lat <= 90:
                                    coordinates.append([lat, lon])  # Store as [lat, lon]
                            except ValueError:
                                continue
                
                if len(coordinates) < 3:  # Need at least 3 points for a polygon
                    continue
                
                # Add to plumes list
                smoke_plume = {
                    'id': f'plume_{i+1:04d}',
                    'coordinates_flat': [coord for pair in coordinates for coord in pair],  # Flatten to [lat1, lon1, lat2, lon2, ...]
                    'coordinate_pairs_count': len(coordinates),
                    'density_category': density_category,
                    'numeric_density': numeric_density,
                    'description': description[:500] if description else "",  # Truncate long descriptions
                    'style_url': style_url,
                    'opacity': SMOKE_OPACITY.get(density_category, SMOKE_OPACITY['Unspecified']),
                    'scraped_at': datetime.now(timezone.utc).isoformat(),
                    'coordinate_count': len(coordinates)
                }
                smoke_plumes.append(smoke_plume)
                
                # Log progress every 10 plumes
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1} placemarks, found {len(smoke_plumes)} valid plumes")
                
            except Exception as e:
                logger.error(f"Error processing placemark #{i+1}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(smoke_plumes)} smoke plumes")
        
        # Count by density for logging
        density_counts = {"Light": 0, "Medium": 0, "Heavy": 0, "Unspecified": 0}
        for plume in smoke_plumes:
            density_counts[plume['density_category']] += 1
        logger.info(f"Density breakdown: {density_counts}")
        
        return smoke_plumes
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching smoke data: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching smoke data: {e}")
        return []


def save_smoke_data_to_firestore_unified(smoke_plumes):
    """Save smoke plume data to unified historical collection structure."""
    try:
        db = get_db_client()
        batch = db.batch()
        timestamp = datetime.now(timezone.utc)
        
        # Create timestamp strings for different intervals
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M")
        iso_timestamp = timestamp.isoformat()
        
        # Update current collection
        current_collection = db.collection('smoke_current')
        
        # Clear current collection
        current_docs = current_collection.stream()
        for doc in current_docs:
            batch.delete(doc.reference)
        
        # Count plumes by density
        density_counts = {"Light": 0, "Medium": 0, "Heavy": 0, "Unspecified": 0}
        
        # Calculate geographic bounds for the data
        if smoke_plumes:
            lats = []
            lngs = []
            for plume in smoke_plumes:
                if 'coordinates' in plume and plume['coordinates']:
                    coords = plume['coordinates'][0]  # First polygon
                    for coord_pair in coords:
                        lngs.append(coord_pair[0])
                        lats.append(coord_pair[1])
            
            region_bounds = {
                'min_lat': min(lats) if lats else 0,
                'max_lat': max(lats) if lats else 0,
                'min_lng': min(lngs) if lngs else 0,
                'max_lng': max(lngs) if lngs else 0
            } if lats and lngs else {}
        else:
            region_bounds = {}
        
        # Add smoke plumes to current collection with enhanced metadata
        for plume in smoke_plumes:
            # Count by density
            density_counts[plume['density_category']] += 1
            
            # Add metadata
            plume['last_updated'] = timestamp
            plume['data_source'] = 'noaa_hms'
            plume['collection_timestamp'] = timestamp_str
            
            # Save to current collection
            current_doc_ref = current_collection.document(plume['id'])
            batch.set(current_doc_ref, plume)
        
        # Save to dedicated smoke historical collection
        smoke_historical_collection = db.collection('smoke_historical_data')
        
        # Create document ID with region hash for better organization
        region_hash = hash(str(sorted(region_bounds.items()))) % 10000 if region_bounds else 0000
        historical_doc_id = f"smoke_{timestamp_str}_{region_hash:04d}"
        
        # Prepare historical document data
        historical_data = {
            'data_type': 'smoke',
            'timestamp': timestamp,
            'iso_timestamp': iso_timestamp,
            'collection_interval': 'hourly',
            'retention_category': 'last_24h',
            'region_bounds': region_bounds,
            'data': smoke_plumes,
            'metadata': {
                'plume_count': len(smoke_plumes),
                'density_counts': density_counts,
                'data_source': 'noaa_hms',
                'collection_timestamp': timestamp_str,
                'status': 'success'
            }
        }
        
        # Save to dedicated smoke historical collection
        historical_doc_ref = smoke_historical_collection.document(historical_doc_id)
        batch.set(historical_doc_ref, historical_data)
        
        # Commit the batch
        batch.commit()
        
        # Update current metadata
        metadata_ref = db.collection('metadata').document('smoke_last_update')
        metadata_ref.set({
            'timestamp': timestamp,
            'timestamp_str': timestamp_str,
            'plume_count': len(smoke_plumes),
            'density_counts': density_counts,
            'status': 'success',
            'historical_doc_id': historical_doc_id,
            'region_bounds': region_bounds
        })
        
        logger.info(f"Successfully saved {len(smoke_plumes)} smoke plumes to unified collection")
        logger.info(f"Historical doc ID: {historical_doc_id}")
        logger.info(f"Density breakdown: {density_counts}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving smoke data to unified Firestore: {e}")
        return False
