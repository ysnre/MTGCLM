import xarray as xr
import numpy as np
from pyresample.geometry import SwathDefinition

def get_satellite_grid(nc_file_path: str) -> SwathDefinition:
    """
    Loads an MTG satellite NetCDF file and returns a pyresample SwathDefinition
    representing the 1000x800 coordinate grid (longitudes and latitudes).
    
    Args:
        nc_file_path (str): Path to a sample satellite NetCDF file.
        
    Returns:
        SwathDefinition: The target grid for resampling.
    """
    # Open the dataset
    ds = xr.open_dataset(nc_file_path)
    
    # x represents longitude, y represents latitude
    # x shape: (1000,), y shape: (800,)
    lons_1d = ds['x'].values
    lats_1d = ds['y'].values
    
    # Create 2D coordinate arrays
    # meshgrid returns (800, 1000) arrays when indexing='ij'
    lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d, indexing='xy')
    
    # Alternatively, indexing='ij' would be lats_1d, lons_1d
    # But usually, it's (y, x). Let's use meshgrid properly:
    # lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d) # default indexing='xy' -> shape is (len(lats_1d), len(lons_1d))
    
    # Create the target SwathDefinition
    target_def = SwathDefinition(lons=lons_2d, lats=lats_2d)
    
    return target_def

if __name__ == "__main__":
    # Test
    sample_file = "e:/belgeler/MTGCLM/data/raw_mtg/turkiye_kesit_20260525000817_0001.nc"
    target_def = get_satellite_grid(sample_file)
    print(f"Target grid shape: {target_def.shape}")
