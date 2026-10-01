import pandas as pd

def load_observations(synop_csv: str, metar_csv: str) -> pd.DataFrame:
    """
    Loads synop and metar observations, combines them, converts timestamps,
    and filters out invalid cloud_coverage (-1).
    
    Args:
        synop_csv (str): Path to synop.csv
        metar_csv (str): Path to metar.csv
        
    Returns:
        pd.DataFrame: Combined observation dataframe.
    """
    # Load SYNOP
    try:
        synop_df = pd.read_csv(synop_csv)
    except Exception as e:
        print(f"Error loading SYNOP: {e}")
        synop_df = pd.DataFrame()
        
    # Load METAR
    try:
        metar_df = pd.read_csv(metar_csv)
    except Exception as e:
        print(f"Error loading METAR: {e}")
        metar_df = pd.DataFrame()
        
    # Combine
    obs_df = pd.concat([synop_df, metar_df], ignore_index=True)
    
    if obs_df.empty:
        return obs_df
        
    # Ensure columns exist
    if 'validity' not in obs_df.columns or 'cloud_coverage' not in obs_df.columns:
        raise ValueError("Missing 'validity' or 'cloud_coverage' columns in observation data.")
        
    # Convert timestamp
    # Format appears to be '2026-05-22 22:00:00'
    obs_df['validity'] = pd.to_datetime(obs_df['validity'])
    
    # Filter out missing cloud coverage (-1)
    obs_df = obs_df[obs_df['cloud_coverage'] != -1]
    
    # Keep necessary columns: validity, wmoid, cloud_coverage
    # Synop might have extra columns like 'sunshine_duration', we can keep it or drop it
    cols_to_keep = ['validity', 'wmoid', 'cloud_coverage']
    
    # Ensure wmoid is int for merging later
    obs_df['wmoid'] = obs_df['wmoid'].astype(int)
    
    return obs_df[cols_to_keep]

if __name__ == "__main__":
    # Test
    synop_path = "e:/belgeler/MTGCLM/data/raw_observation/synop.csv"
    metar_path = "e:/belgeler/MTGCLM/data/raw_observation/metar.csv"
    
    df = load_observations(synop_path, metar_path)
    print(f"Loaded {len(df)} valid observations.")
    print(df.head())
    print("\nDate range:", df['validity'].min(), "to", df['validity'].max())
