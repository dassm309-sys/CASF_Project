import os
import glob
import re

# Point to your folder with the 105 messy CSVs
folder_path = 'data/3_Ground_Truth/Processed_GT/'

# Get all CSV files
files = glob.glob(os.path.join(folder_path, "*.csv"))

print("--- Starting Smart File Rename ---")

for file in files:
    old_name = os.path.basename(file)
    new_name = ""
    
    # 1. Target YOUR Deep Sea files
    # Example: GT_DeepSea_V01_V36_V12_DeepSea.csv -> V12_DeepSea.csv
    if "GT_DeepSea" in old_name:
        # This grabs the "V" and the numbers right before "_DeepSea"
        match = re.search(r'(V\d+_DeepSea)', old_name)
        if match:
            new_name = f"{match.group(1)}.csv"
            
    # 2. Target Teammate File 1 (Fresh Water)
    # Example: underwater video summarisation_Sheet12.csv -> V12_FreshWater.csv
    elif "underwater video summarisation" in old_name.lower() and "33 to 50" not in old_name.lower():
        # This looks for the word "Sheet" and grabs the number after it
        match = re.search(r'Sheet(\d+)', old_name)
        if match:
            num = int(match.group(1))
            new_name = f"V{num:02d}_FreshWater.csv"
            
    # 3. Target Teammate File 2 (Surface Ocean)
    # Example: clips 33 to 50_Sheet4.csv -> V04_SurfaceOcean.csv
    elif "33 to 50" in old_name.lower():
        match = re.search(r'Sheet(\d+)', old_name)
        if match:
            num = int(match.group(1))
            new_name = f"V{num:02d}_SurfaceOcean.csv"
            
    # Apply the rename
    if new_name:
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: '{old_name}'  -->  '{new_name}'")

print("\n--- Success! All 105 files are perfectly mapped and renamed. ---")