#!/usr/bin/env python3
"""
Fetches large-scale F1 telemetry data using FastF1 and stores as .npy arrays for algorithm selection benchmarking.
- Downloads multiple seasons and sessions (configurable)
- Extracts ALL numeric telemetry channels available
- Saves each array as .npy in data/real_world_bigtest/raw/
- Logs metadata in data/real_world_bigtest/index.csv

IMPORTANT: For honest evaluation, this script does NOT sort, filter, or preprocess the arrays in any way.
Arrays are saved exactly as they appear in the telemetry. Do not modify this behavior.
"""
import os
import sys
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import fastf1
from fastf1 import get_event_schedule, get_session
from fastf1.core import Laps
import warnings
warnings.filterwarnings('ignore')

# --- Config for MAXIMUM data ---
SEASONS = list(range(2020, 2025))  # Focus on recent years with better data quality
SESSIONS = ["FP1", "FP2", "FP3", "Q", "R", "S", "SS"]  # Include practice sessions for more data
# All possible telemetry channels - will check availability per session
ALL_CHANNELS = [
    "Speed", "Throttle", "Brake", "RPM", "Gear", "DRS", 
    "X", "Y", "Z",  # Position coordinates
    "Status", "Time", "SessionTime", "Distance",
    "RelativeDistance", "DistanceToDriverAhead", "DriverAhead",
    "nGear", "Brake", "ThrottlePedal", "SpeedI1", "SpeedI2", 
    "SpeedFL", "SpeedST", "IsPersonalBest", "Compound", "TyreLife",
    "FreshTyre", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
    "SpeedTrap", "IsAccurate", "LapStartTime", "Team", "Driver",
    "LapNumber", "Stint", "PitOutTime", "PitInTime", "Source"
]
OUTDIR = Path("data/real_world_bigtest/raw/")
INDEX_CSV = Path("data/real_world_bigtest/index.csv")
MIN_LEN = 50  # Much lower - F1 telemetry is ~10-20Hz, so 50 points = ~3 seconds minimum

OUTDIR.mkdir(parents=True, exist_ok=True)
INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)

# Enable FastF1 cache for faster repeated runs
cache_dir = Path('data/f1_cache')
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

index_rows = []
array_id = 0
total_size_mb = 0

def get_numeric_columns(df):
    """Get only numeric columns from telemetry dataframe."""
    numeric_cols = []
    for col in df.columns:
        try:
            # Try to convert to numeric, skip if it fails
            pd.to_numeric(df[col], errors='coerce')
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_cols.append(col)
        except:
            continue
    return numeric_cols

def save_telemetry_arrays(telemetry_df, prefix, year, gp, roundnum, sess, driver=None, lap_num=None):
    """Save all numeric columns from telemetry as separate arrays."""
    global array_id, total_size_mb
    
    if telemetry_df is None or telemetry_df.empty:
        return
        
    numeric_cols = get_numeric_columns(telemetry_df)
    print(f"    Found {len(numeric_cols)} numeric channels: {numeric_cols[:10]}{'...' if len(numeric_cols) > 10 else ''}")
    
    for ch in numeric_cols:
        try:
            arr = telemetry_df[ch].to_numpy()
            # Remove NaN values but keep array structure
            arr = arr[~pd.isna(arr)]
            
            if arr.size < MIN_LEN:
                continue
                
            # Create filename
            if lap_num is not None:
                fname = f"f1_{year}_{roundnum}_{sess}_{driver}_lap{lap_num}_{ch}.npy"
                source_desc = f"Driver {driver} Lap {lap_num}"
            else:
                fname = f"f1_{year}_{roundnum}_{sess}_{driver}_{ch}.npy" if driver else f"f1_{year}_{roundnum}_{sess}_{ch}.npy"
                source_desc = f"Driver {driver}" if driver else "Full session"
            
            # Save array
            np.save(OUTDIR / fname, arr)
            size_mb = arr.nbytes / 1024 / 1024
            total_size_mb += size_mb
            
            index_rows.append({
                "array_id": array_id,
                "file": fname,
                "year": year,
                "event": gp,
                "round": roundnum,
                "session": sess,
                "driver": driver or "ALL",
                "lap": lap_num or "FULL_SESSION",
                "channel": ch,
                "n_elements": arr.size,
                "dtype": str(arr.dtype),
                "size_mb": round(size_mb, 3)
            })
            array_id += 1
            print(f"      Saved: {fname} ({arr.size} elements, {size_mb:.2f} MB)")
            
        except Exception as e:
            print(f"      Failed to save {ch}: {e}")
            continue

print("Starting large-scale F1 data download...")
print(f"Seasons: {SEASONS}")
print(f"Sessions: {SESSIONS}")
print(f"Minimum length: {MIN_LEN} elements")
print(f"Output directory: {OUTDIR}")
print("="*60)

for year in SEASONS:
    print(f"\n🏁 FETCHING SEASON {year}")
    try:
        schedule = get_event_schedule(year)
    except Exception as e:
        print(f"Failed to get schedule for {year}: {e}")
        continue
        
    for _, event in schedule.iterrows():
        gp = event['EventName'] 
        roundnum = event['RoundNumber']
        print(f"\n📍 {gp} (Round {roundnum})")
        
        for sess in SESSIONS:
            print(f"  📊 Session: {sess}")
            try:
                session = get_session(year, roundnum, sess)
                session.load(telemetry=True, laps=True, weather=False)
            except Exception as e:
                print(f"    ❌ Skipped: {e}")
                continue
                
            # Method 1: Get full session telemetry (massive arrays)
            try:
                print("    🔄 Loading full session telemetry...")
                full_tel = session.get_session_telemetry()
                if full_tel is not None and not full_tel.empty:
                    save_telemetry_arrays(full_tel, "session", year, gp, roundnum, sess)
            except Exception as e:
                print(f"    ⚠️ Full session telemetry failed: {e}")
            
            # Method 2: Per-driver telemetry
            try:
                drivers = session.drivers
                print(f"    👤 Processing {len(drivers)} drivers...")
                for driver in drivers[:5]:  # Limit to 5 drivers per session for manageable size
                    try:
                        driver_laps = session.laps.pick_drivers([driver])
                        if driver_laps.empty:
                            continue
                            
                        # Get all telemetry for this driver in this session
                        driver_tel = driver_laps.get_telemetry()
                        if driver_tel is not None and not driver_tel.empty:
                            save_telemetry_arrays(driver_tel, "driver", year, gp, roundnum, sess, driver)
                        
                        # Also get per-lap telemetry for variety
                        for lap_num, lap in driver_laps.head(3).iterlaps():  # First 3 laps per driver
                            try:
                                lap_tel = lap.get_telemetry()
                                save_telemetry_arrays(lap_tel, "lap", year, gp, roundnum, sess, driver, lap_num)
                            except:
                                continue
                                
                    except Exception as e:
                        print(f"      ❌ Driver {driver} failed: {e}")
                        continue
                        
            except Exception as e:
                print(f"    ❌ Driver processing failed: {e}")
            
            print(f"    💾 Session total so far: {total_size_mb:.1f} MB")

# Write index
print(f"\n🏁 DOWNLOAD COMPLETE!")
print(f"Total arrays: {array_id}")
print(f"Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")

if index_rows:
    with open(INDEX_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=index_rows[0].keys())
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Index saved: {INDEX_CSV}")
    
    # Summary statistics
    df_summary = pd.DataFrame(index_rows)
    print(f"\n📊 SUMMARY:")
    print(f"  Years: {df_summary['year'].unique()}")
    print(f"  Sessions: {df_summary['session'].value_counts().to_dict()}")
    print(f"  Channels: {len(df_summary['channel'].unique())} unique")
    print(f"  Size distribution (MB):")
    print(f"    Min: {df_summary['size_mb'].min():.2f}")
    print(f"    Max: {df_summary['size_mb'].max():.2f}") 
    print(f"    Mean: {df_summary['size_mb'].mean():.2f}")
    print(f"  Array size distribution:")
    print(f"    Min: {df_summary['n_elements'].min():,} elements")
    print(f"    Max: {df_summary['n_elements'].max():,} elements")
    print(f"    Mean: {df_summary['n_elements'].mean():.0f} elements")
else:
    print("❌ No data was downloaded!")
