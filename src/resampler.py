from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.kd_tree import resample_nearest
import numpy as np

def resample_radar_to_satellite(radar_array: np.ndarray, 
                                extent: tuple, 
                                target_def: SwathDefinition,
                                radius_of_influence: float = 10000) -> np.ndarray:
    """
    Resamples a single radar image to the target satellite grid using nearest neighbor.
    
    Args:
        radar_array (np.ndarray): 2D numpy array of radar reflectivity.
        extent (tuple): Bounding box of the radar image (min_lon, min_lat, max_lon, max_lat).
        target_def (SwathDefinition): The target MTG satellite grid definition.
        radius_of_influence (float): Radius in meters to search for nearest neighbor. Default 10000 (10km).
        
    Returns:
        np.ndarray: Resampled radar data matching the shape of target_def.
    """
    height, width = radar_array.shape
    
    # Define radar area projection (assumed Plate Carree EPSG:4326 for bounding box)
    area_id = 'radar_area'
    description = 'Radar Area'
    proj_id = 'longlat'
    projection = {'proj': 'longlat', 'datum': 'WGS84'}
    
    radar_def = AreaDefinition(
        area_id,
        description,
        proj_id,
        projection,
        width,
        height,
        extent
    )
    
    # Resample
    # fill_value is set to 0 to represent no radar signal/out of bounds
    resampled_data = resample_nearest(
        radar_def, 
        radar_array, 
        target_def,
        radius_of_influence=radius_of_influence,
        fill_value=0
    )
    
    return resampled_data

def resample_radar_to_satellite_aeqd(radar_array: np.ndarray,
                                     lat_0: float,
                                     lon_0: float,
                                     r_km: float,
                                     target_def: SwathDefinition,
                                     radius_of_influence: float = 10000) -> np.ndarray:
    """
    Resamples a single radar image to the target satellite grid using nearest neighbor,
    with an Azimuthal Equidistant projection (AEQD) centered on the radar's coordinates.
    
    Args:
        radar_array (np.ndarray): 2D numpy array of radar reflectivity.
        lat_0 (float): Latitude of the radar center (center of projection).
        lon_0 (float): Longitude of the radar center (center of projection).
        r_km (float): Radar sweep radius in kilometers.
        target_def (SwathDefinition): The target MTG satellite grid definition.
        radius_of_influence (float): Search radius for nearest neighbor in meters.
        
    Returns:
        np.ndarray: Resampled radar data matching the shape of target_def.
    """
    height, width = radar_array.shape
    
    # Distance of boundaries in meters from the radar center
    r_m = r_km * 1000.0
    extent = (-r_m, -r_m, r_m, r_m)
    
    area_id = 'radar_aeqd'
    description = f'Radar AEQD Area centered at {lat_0}, {lon_0}'
    proj_id = 'aeqd'
    projection = {
        'proj': 'aeqd',
        'lat_0': lat_0,
        'lon_0': lon_0,
        'x_0': 0,
        'y_0': 0,
        'datum': 'WGS84',
        'units': 'm'
    }
    
    radar_def = AreaDefinition(
        area_id,
        description,
        proj_id,
        projection,
        width,
        height,
        extent
    )
    
    # Resample using nearest neighbor
    resampled_data = resample_nearest(
        radar_def, 
        radar_array, 
        target_def,
        radius_of_influence=radius_of_influence,
        fill_value=0
    )
    
    return resampled_data
