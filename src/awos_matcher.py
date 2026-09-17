import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from satellite_grid import get_satellite_grid

def load_stations(csv_path: str) -> pd.DataFrame:
    """
    Loads station list and returns a DataFrame.
    """
    df = pd.read_csv(csv_path)
    # Ensure wmoid is treated as integer or string consistently. Let's use integer.
    df['wmoid'] = df['wmoid'].astype(int)
    # Drop rows without valid lat/lon
    df = df.dropna(subset=['latitude', 'longitude'])
    return df

def match_stations_to_grid(stations_df: pd.DataFrame, target_def) -> pd.DataFrame:
    """
    Matches station coordinates to the closest (y, x) indices on the target grid.
    
    Args:
        stations_df (pd.DataFrame): DataFrame containing 'latitude' and 'longitude'.
        target_def: pyresample SwathDefinition of the satellite grid.
        
    Returns:
        pd.DataFrame: A new DataFrame with 'y_idx' and 'x_idx' columns added.
    """
    # Flatten grid coordinates
    grid_lons = target_def.lons.flatten()
    grid_lats = target_def.lats.flatten()
    
    # We use (lat, lon) for the tree
    tree = cKDTree(np.c_[grid_lats, grid_lons])
    
    # Station coordinates
    station_lats = stations_df['latitude'].values
    station_lons = stations_df['longitude'].values
    
    # Query nearest neighbors
    distances, indices = tree.query(np.c_[station_lats, station_lons])
    
    # Convert 1D indices back to 2D (y, x)
    y_indices, x_indices = np.unravel_index(indices, target_def.shape)
    
    # Add to dataframe
    matched_df = stations_df.copy()
    matched_df['y_idx'] = y_indices
    matched_df['x_idx'] = x_indices
    matched_df['distance_deg'] = distances
    
    return matched_df

if __name__ == "__main__":
    # Test
    station_csv = "e:/belgeler/MTGCLM/data/raw_observation/station_list.csv"
    sample_mtg = "e:/belgeler/MTGCLM/data/raw_mtg/turkiye_kesit_20260525000817_0001.nc"
    
    stations = load_stations(station_csv)
    print(f"Loaded {len(stations)} stations.")
    
    target_def = get_satellite_grid(sample_mtg)
    
    matched = match_stations_to_grid(stations, target_def)
    print("Matched stations:")
    print(matched[['wmoid', 'station_name', 'latitude', 'longitude', 'y_idx', 'x_idx', 'distance_deg']].head())
