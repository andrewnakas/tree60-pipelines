"""
Scheduled Temperature Tile Regeneration
Runs every 6 hours to regenerate temperature tiles for the full timeline range (-18h to +18h)
"""
import logging
from temp_overlay_simple import get_gfs_temperature_png

logger = logging.getLogger(__name__)


def regenerate_all_temperature_tiles():
    """
    Regenerate temperature tiles for hours -18 to +18

    This covers the full timeline slider range:
    - Historical: -18 to 0 hours (past temperatures)
    - Forecast: 0 to +18 hours (future temperatures)

    This should run every 6 hours after new GFS data is available.
    """
    results = {
        'success': [],
        'failed': []
    }

    # Full range: -18 to +18 inclusive = 37 tiles
    hours_to_generate = list(range(-18, 19))  # -18 to +18

    logger.info(f"Starting temperature tile regeneration for hours -18 to +18 ({len(hours_to_generate)} tiles)")

    for hour in hours_to_generate:
        try:
            logger.info(f"Generating hour {hour:+d}...")
            result = get_gfs_temperature_png(forecast_hour=hour)
            results['success'].append({
                'hour': hour,
                'url': result.get('url'),
                'cached': result.get('cached', False),
                'valid_time': result.get('valid_time')
            })
            logger.info(f"Hour {hour:+d} completed: {result.get('url')}")
        except Exception as e:
            logger.error(f"Hour {hour:+d} failed: {e}", exc_info=True)
            results['failed'].append({
                'hour': hour,
                'error': str(e)
            })

    logger.info(f"Regeneration complete: {len(results['success'])} success, {len(results['failed'])} failed")

    return {
        'total': len(hours_to_generate),
        'success_count': len(results['success']),
        'failed_count': len(results['failed']),
        'results': results
    }


def regenerate_key_temperature_tiles():
    """
    Regenerate only key temperature tiles for faster updates.

    Generates tiles at 3-hour intervals to cover the timeline quickly.
    On-demand requests will fill in gaps as needed.
    """
    results = {
        'success': [],
        'failed': []
    }

    # Key hours at 3-hour intervals: -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18
    key_hours = list(range(-18, 19, 3))

    logger.info(f"Starting key temperature tile regeneration ({len(key_hours)} tiles)")

    for hour in key_hours:
        try:
            logger.info(f"Generating key hour {hour:+d}...")
            result = get_gfs_temperature_png(forecast_hour=hour)
            results['success'].append({
                'hour': hour,
                'url': result.get('url'),
                'cached': result.get('cached', False),
                'valid_time': result.get('valid_time')
            })
            logger.info(f"Key hour {hour:+d} completed: {result.get('url')}")
        except Exception as e:
            logger.error(f"Key hour {hour:+d} failed: {e}", exc_info=True)
            results['failed'].append({
                'hour': hour,
                'error': str(e)
            })

    logger.info(f"Key tile regeneration complete: {len(results['success'])} success, {len(results['failed'])} failed")

    return {
        'total': len(key_hours),
        'success_count': len(results['success']),
        'failed_count': len(results['failed']),
        'results': results
    }
