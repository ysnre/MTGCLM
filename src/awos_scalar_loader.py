import os
import csv
import datetime
import numpy as np

def round_to_nearest_10min(dt: datetime.datetime) -> datetime.datetime:
    """Rounds a datetime object to the nearest 10-minute interval."""
    discard = datetime.timedelta(minutes=dt.minute % 10,
                                 seconds=dt.second,
                                 microseconds=dt.microsecond)
    dt -= discard
    if discard >= datetime.timedelta(minutes=5):
        dt += datetime.timedelta(minutes=10)
    return dt

def parse_station_elevations(stationlist_path: str) -> dict:
    """Parses the stationlist CSV and returns a dict mapping wmoid -> elevation."""
    elevations = {}
    with open(stationlist_path, 'r', encoding='latin1', errors='replace') as f:
        header = next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            # The file has a weird format where the whole line might be wrapped in quotes
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
                # Also double quotes might be used for escaping
                line = line.replace('""', '"')
                
            reader = csv.reader([line])
            try:
                row = next(reader)
                if len(row) >= 5:
                    wmoid = int(row[0])
                    elev_str = row[4]
                    if elev_str.strip() and elev_str.lower() != 'nan':
                        elevations[wmoid] = float(elev_str)
            except Exception as e:
                print(f"Failed to parse line: {line} -> {e}")
    return elevations

def build_awos_lookup(required_keys: set, awos_dir: str) -> dict:
    """
    Reads the 4.4GB AWOS observation CSV line by line.
    Only stores the data for (wmoid, year, month, day, hour, minute) present in required_keys.
    
    Args:
        required_keys: set of tuples (wmoid, year, month, day, hour, minute)
        awos_dir: directory containing observation_awos.csv and stationlist_awos.csv
        
    Returns:
        dict: {(wmoid, ts_str): (temp, rh, elevation)}
    """
    stationlist_path = os.path.join(awos_dir, "stationlist_awos.csv")
    observation_path = os.path.join(awos_dir, "observation_awos.csv")
    
    print("Loading station elevations...")
    elevations = parse_station_elevations(stationlist_path)
    print(f"Loaded {len(elevations)} station elevations.")
    
    lookup = {}
    
    print(f"Scanning {observation_path} (this may take a few minutes)...")
    found_count = 0
    
    # Read the 4.4GB CSV line by line to minimize RAM usage
    with open(observation_path, 'r', encoding='latin1', errors='replace') as f:
        header = next(f) # ISTNO;istad;YIL;AY;GUN;SAAT;DAKIKA_BASLANGICI;10 Dakikalik Ort Sic;...
        for line in f:
            parts = line.strip().split(';')
            if len(parts) < 13:
                continue
                
            try:
                wmoid = int(parts[0])
                year = int(parts[2])
                month = int(parts[3])
                day = int(parts[4])
                hour = int(parts[5])
                minute = int(parts[6])
                
                key = (wmoid, year, month, day, hour, minute)
                
                if key in required_keys:
                    # Parse Temp and RH
                    temp_str = parts[7].replace(',', '.')
                    rh_str = parts[10].replace(',', '.')
                    
                    temp = float(temp_str) if temp_str.strip() and temp_str.lower() not in ['nan', '-9999'] else np.nan
                    rh = float(rh_str) if rh_str.strip() and rh_str.lower() not in ['nan', '-9999'] else np.nan
                    
                    elev = elevations.get(wmoid, np.nan)
                    
                    lookup[key] = (temp, rh, elev)
                    found_count += 1
                    
            except ValueError:
                continue
                
    print(f"Scan complete. Found {found_count} out of {len(required_keys)} requested records.")
    
    # Missing value imputation (Median)
    temps = [v[0] for v in lookup.values() if not np.isnan(v[0])]
    rhs = [v[1] for v in lookup.values() if not np.isnan(v[1])]
    elevs = [v[2] for v in lookup.values() if not np.isnan(v[2])]
    
    med_temp = np.median(temps) if temps else 15.0
    med_rh = np.median(rhs) if rhs else 60.0
    med_elev = np.median(elevs) if elevs else 500.0
    
    # Normalization (Z-score)
    mean_temp, std_temp = np.mean(temps) if temps else 0, np.std(temps) if temps else 1
    mean_rh, std_rh = np.mean(rhs) if rhs else 0, np.std(rhs) if rhs else 1
    mean_elev, std_elev = np.mean(elevs) if elevs else 0, np.std(elevs) if elevs else 1
    
    # Avoid div by zero
    std_temp = std_temp if std_temp > 0 else 1
    std_rh = std_rh if std_rh > 0 else 1
    std_elev = std_elev if std_elev > 0 else 1
    
    print(f"Normalization Stats:")
    print(f"  Temp: Mean={mean_temp:.2f}, Std={std_temp:.2f}, Median={med_temp:.2f}")
    print(f"  RH:   Mean={mean_rh:.2f}, Std={std_rh:.2f}, Median={med_rh:.2f}")
    print(f"  Elev: Mean={mean_elev:.2f}, Std={std_elev:.2f}, Median={med_elev:.2f}")
    
    # Final normalized lookup dictionary
    normalized_lookup = {}
    for key in required_keys:
        if key in lookup:
            t, r, e = lookup[key]
            t = t if not np.isnan(t) else med_temp
            r = r if not np.isnan(r) else med_rh
            e = e if not np.isnan(e) else med_elev
        else:
            t, r, e = med_temp, med_rh, med_elev
            
        norm_t = (t - mean_temp) / std_temp
        norm_r = (r - mean_rh) / std_rh
        norm_e = (e - mean_elev) / std_elev
        
        normalized_lookup[key] = [norm_t, norm_r, norm_e]
        
    return normalized_lookup
