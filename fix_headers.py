import pandas as pd
import glob
import os

# 1. Point the script to your three organized folders
folders = [
    'data/3_Ground_Truth/Deep_Sea_GT/',
    'data/3_Ground_Truth/Fresh_Water_GT/',
    'data/3_Ground_Truth/Surface_Ocean_GT/'
]

print("--- Starting Header Sweep ---")

for folder in folders:
    # Get all CSVs in this folder
    files = glob.glob(os.path.join(folder, "*.csv"))
    
    for file in files:
        df = pd.read_csv(file)
        
        # 2. Check if this file has the Apple Numbers "Unnamed" bug
        if 'Unnamed:_1' in df.columns or 'Unnamed: 1' in df.columns:
            
            # Grab the first row of data (which is actually your real headers)
            real_headers = df.iloc[0]
            
            # Slice the dataframe to drop that top row
            df = df[1:]
            
            # Apply the real headers
            df.columns = real_headers
            
            # Standardize them just to be safe (replace spaces with underscores)
            df.columns = [str(c).replace(' ', '_').strip() for c in df.columns]
            
            # Overwrite the broken CSV with the perfectly clean one
            df.to_csv(file, index=False)
            print(f"Fixed broken headers in: {os.path.basename(file)}")

print("\n--- Success! All CSVs are now mathematically ready. ---")