import pandas as pd
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import numpy as np
import os
from datetime import datetime

def load_radar_metadata(csv_path: str) -> pd.DataFrame:
    """
    Loads radar metadata from radar.csv and returns a DataFrame.
    Indexes the dataframe by 'oa' (filename prefix).
    
    Args:
        csv_path (str): Path to radar.csv
        
    Returns:
        pd.DataFrame: DataFrame containing radar metadata indexed by 'oa'.
    """
    df = pd.read_csv(csv_path, dtype={'oa': str})
    # Drop duplicates just in case there are multiple identical 'oa' entries
    df = df.drop_duplicates(subset=['oa'])
    df = df.set_index('oa')
    return df

def get_radar_extent(metadata_df: pd.DataFrame, oa: str) -> tuple:
    """
    Retrieves the area_extent for a given radar filename prefix.
    
    Args:
        metadata_df (pd.DataFrame): Metadata dataframe from load_radar_metadata
        oa (str): The filename prefix, e.g., 'SMN260525000002PPI82NF'
        
    Returns:
        tuple: (min_lon, min_lat, max_lon, max_lat)
    """
    if oa not in metadata_df.index:
        raise ValueError(f"Metadata for {oa} not found in CSV.")
    
    row = metadata_df.loc[oa]
    ll_str = row['ll']
    # ll is string formatted as "min_lon min_lat max_lon max_lat"
    parts = ll_str.strip().split()
    if len(parts) != 4:
        raise ValueError(f"Invalid ll format for {oa}: {ll_str}")
        
    min_lon = float(parts[0])
    min_lat = float(parts[1])
    max_lon = float(parts[2])
    max_lat = float(parts[3])
    
    return (min_lon, min_lat, max_lon, max_lat)

def load_radar_image(png_path: str) -> np.ndarray:
    """
    Loads a radar PNG image and extracts the Blue (B) channel.
    
    Args:
        png_path (str): Path to the radar PNG file.
        
    Returns:
        np.ndarray: The 2D array of the Blue channel containing reflectivity.
    """
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"Radar image not found: {png_path}")
        
    # Read image in RGB format
    try:
        img = Image.open(png_path).convert('RGB')
        img_array = np.array(img)
    except Exception as e:
        raise ValueError(f"Failed to load image: {png_path}, Error: {e}")
        
    # Extract Blue channel (index 2 in RGB)
    blue_channel = img_array[:, :, 2]
    
    return blue_channel

def parse_radar_filename_time(filename: str) -> datetime:
    """
    Parses the timestamp from the radar filename.
    Format: XXXYYMMDDHHMMSS...png
    e.g. AFY260522041206PPI7STZ.png -> 2026-05-22 04:12:06
    """
    basename = os.path.basename(filename)
    # The time part starts after the 3-letter prefix
    time_str = basename[3:15]  # YYMMDDHHMMSS
    dt = datetime.strptime(time_str, "%y%m%d%H%M%S")
    return dt

if __name__ == "__main__":
    import sys
    # Test
    csv_path = "e:/belgeler/MTGCLM/data/raw_radar/radar.csv"
    if os.path.exists(csv_path):
        df = load_radar_metadata(csv_path)
        sample_oa = df.index[0]
        print(f"Sample oa: {sample_oa}")
        extent = get_radar_extent(df, sample_oa)
        print(f"Extent for {sample_oa}: {extent}")
        
        # Check a png if it exists
        png_path = os.path.join("e:/belgeler/MTGCLM/data/raw_radar", f"{sample_oa}.png")
        if os.path.exists(png_path):
            img_data = load_radar_image(png_path)
            print(f"Loaded image {sample_oa}.png, shape: {img_data.shape}, max value: {img_data.max()}")
        else:
            print(f"Sample image {png_path} not found.")
