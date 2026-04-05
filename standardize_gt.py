import pandas as pd
import glob
import os

# Paths based on your screenshot
input_path = 'data/3_Ground_Truth/Raw_Partner_Files/'
output_path = 'data/3_Ground_Truth/Processed_GT/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Find all xlsx files
files = glob.glob(os.path.join(input_path, "*.xlsx"))

for file in files:
    print(f"Processing: {os.path.basename(file)}")
    
    # Load all sheets at once
    all_sheets = pd.read_excel(file, sheet_name=None)
    
    for sheet_name, df in all_sheets.items():
        # RENAME LOGIC: Replace space with underscore in headers
        df.columns = [c.replace(' ', '_') for c in df.columns]
        
        # THE FIX: pandas updated 'applymap' to just 'map' in recent versions
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Save as a clean CSV for your algorithm
        clean_name = f"{os.path.basename(file).split('.')[0]}_{sheet_name}.csv"
        df.to_csv(os.path.join(output_path, clean_name), index=False)

print("\nDone! All tables processed and standardized.")