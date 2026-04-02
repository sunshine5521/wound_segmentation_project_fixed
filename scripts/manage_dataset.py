import os
import re
import shutil
import pandas as pd
import numpy as np
import cv2
import yaml
from glob import glob

CONFIG_PATH = 'config.yaml'
IMAGES_DIR = 'data/processed/images'
MASKS_DIR = 'data/processed/masks'
MAPPING_PATH = 'data/processed/mapping.csv'
TRAIN_CSV_PATH = 'data/processed/train.csv'
ANNOTATIONS_PATH = 'data/annotations.csv'
EXCEL_PATH = '1119-20260304-Results-D0-D3.xlsx'

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_numeric_id(filename):
    match = re.search(r'img_(\d+)', filename)
    return int(match.group(1)) if match else -1

def main():
    config = load_config()
    pixels_per_mm = config['data'].get('pixels_per_mm', 42.0)
    
    print("Step 1: Analyzing current images...")
    image_files = sorted(glob(os.path.join(IMAGES_DIR, 'img_*.jpg')), key=lambda x: get_numeric_id(os.path.basename(x)))
    if not image_files:
        print("No images found!")
        return

    print(f"Found {len(image_files)} images.")
    
    if os.path.exists(MAPPING_PATH):
        mapping_df = pd.read_csv(MAPPING_PATH)
        id_to_original = {}
        for _, row in mapping_df.iterrows():
            new_name = row['NewName'] 
            numeric_id = get_numeric_id(new_name)
            id_to_original[numeric_id] = row['OriginalPath']
    else:
        print("Mapping file not found! Cannot proceed safely.")
        return

    renaming_map = [] 
    
    valid_original_paths = set()
    valid_original_basenames = set()

    for idx, file_path in enumerate(image_files):
        old_name = os.path.basename(file_path)
        old_id = get_numeric_id(old_name)
        
        new_id = idx
        new_name_base = f"img_{new_id}"
        
        if old_id in id_to_original:
            orig_path = id_to_original[old_id]
            valid_original_paths.add(orig_path)
            base = os.path.basename(orig_path)
            valid_original_basenames.add(base)
            
            if base.lower().endswith('.jpeg'):
                valid_original_basenames.add(base[:-5] + '.jpg')
            elif base.lower().endswith('.jpg'):
                valid_original_basenames.add(base[:-4] + '.jpeg')
            
            renaming_map.append({
                'old_image': file_path,
                'new_image_name': f"{new_name_base}.jpg",
                'old_mask': os.path.join(MASKS_DIR, old_name.replace('.jpg', '.png')),
                'new_mask_name': f"{new_name_base}.png",
                'original_path': orig_path,
                'new_mapping_name': f"{new_name_base}.jpeg" 
            })
        else:
            print(f"Warning: No mapping found for {old_name}. It will be renumbered but lost from metadata.")
            renaming_map.append({
                'old_image': file_path,
                'new_image_name': f"{new_name_base}.jpg",
                'old_mask': os.path.join(MASKS_DIR, old_name.replace('.jpg', '.png')),
                'new_mask_name': f"{new_name_base}.png",
                'original_path': None,
                'new_mapping_name': None
            })

    print("Step 2: Renaming files and updating local metadata...")
    
    temp_img_dir = os.path.join(IMAGES_DIR, "temp_renaming")
    temp_mask_dir = os.path.join(MASKS_DIR, "temp_renaming")
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(temp_mask_dir, exist_ok=True)
    
    
    file_move_map = {}
    
    for img_path in image_files:
        basename = os.path.basename(img_path)
        temp_path = os.path.join(temp_img_dir, basename)
        shutil.move(img_path, temp_path)
        file_move_map[img_path] = temp_path
        
        mask_name = basename.replace('.jpg', '.png')
        mask_path = os.path.join(MASKS_DIR, mask_name)
        if os.path.exists(mask_path):
            temp_mask_path = os.path.join(temp_mask_dir, mask_name)
            shutil.move(mask_path, temp_mask_path)
            file_move_map[mask_path] = temp_mask_path

    
    for item in renaming_map:
        old_img_path = item['old_image']
        if old_img_path in file_move_map:
            src = file_move_map[old_img_path]
            dst = os.path.join(IMAGES_DIR, item['new_image_name'])
            shutil.move(src, dst)
            
        old_mask_path = item['old_mask']
        if old_mask_path in file_move_map:
            src = file_move_map[old_mask_path]
            dst = os.path.join(MASKS_DIR, item['new_mask_name'])
            shutil.move(src, dst)

    try:
        os.rmdir(temp_img_dir)
        os.rmdir(temp_mask_dir)
    except:
        print("Warning: Could not remove temp dirs (maybe not empty?)")

    new_mapping_data = []
    for item in renaming_map:
        if item['original_path']:
            new_mapping_data.append({
                'OriginalPath': item['original_path'],
                'NewName': item['new_mapping_name']
            })
    pd.DataFrame(new_mapping_data).to_csv(MAPPING_PATH, index=False)
    print(f"Updated {MAPPING_PATH}")

    new_train_data = []
    for item in renaming_map:
        new_train_data.append({
            'image_name': item['new_image_name']
        })
    pd.DataFrame(new_train_data).to_csv(TRAIN_CSV_PATH, index=False)
    print(f"Updated {TRAIN_CSV_PATH}")

    print("Step 3: Synchronizing external files (annotations & Excel)...")
    
    if os.path.exists(ANNOTATIONS_PATH):
        ann_df = pd.read_csv(ANNOTATIONS_PATH)
        initial_count = len(ann_df)
        ann_df = ann_df[ann_df['image_name'].isin(valid_original_basenames)]
        ann_df.to_csv(ANNOTATIONS_PATH, index=False)
        print(f"Updated {ANNOTATIONS_PATH}: {initial_count} -> {len(ann_df)} rows")
    
    if os.path.exists(EXCEL_PATH):
        excel_df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
        
        label_cols = [c for c in excel_df.columns if 'Label' in str(c)]
        
        changes_made = False
        for col in label_cols:
            for idx, value in excel_df[col].items():
                if pd.notna(value):
                    if value not in valid_original_basenames:
                        excel_df.at[idx, col] = np.nan
                        col_loc = excel_df.columns.get_loc(col)
                        if col_loc + 1 < len(excel_df.columns):
                            excel_df.iloc[idx, col_loc + 1] = np.nan 
                        if col_loc + 2 < len(excel_df.columns):
                             excel_df.iloc[idx, col_loc + 2] = np.nan 
                        changes_made = True
        
        excel_df.to_excel(EXCEL_PATH, index=False)
        print(f"Updated {EXCEL_PATH} (Cleared deleted entries)")

    print("Step 4: Verifying Data Consistency...")
    
    ann_df = pd.read_csv(ANNOTATIONS_PATH)
    excel_df = pd.read_excel(EXCEL_PATH)
    
    excel_areas = {}
    label_cols = [c for c in excel_df.columns if 'Label' in str(c)]
    for col in label_cols:
        col_loc = excel_df.columns.get_loc(col)
        if col_loc + 1 >= len(excel_df.columns): continue
        area_col = excel_df.columns[col_loc + 1]
        
        for idx, label in excel_df[col].items():
            if pd.notna(label):
                area = excel_df.iloc[idx, col_loc + 1]
                excel_areas[label] = area

    print("\n--- Consistency Report (Annotations vs Excel) ---")
    mismatches = 0
    for _, row in ann_df.iterrows():
        name = row['image_name']
        ann_area = row['actual_area_mm2']
        
        if name in excel_areas:
            excel_area = excel_areas[name]
            try:
                if abs(float(ann_area) - float(excel_area)) > 0.01:
                    print(f"Mismatch for {name}: CSV={ann_area}, Excel={excel_area}")
                    mismatches += 1
            except:
                print(f"Error comparing {name}: CSV={ann_area}, Excel={excel_area}")
        else:
            print(f"Warning: {name} found in Annotations but not in Excel!")
            mismatches += 1
            
    if mismatches == 0:
        print("All matched!")

    print("\nStep 5: Mask Quality Check...")
    print(f"Using Calibration: 1 mm = {pixels_per_mm} pixels")
    mm2_per_pixel = 1 / (pixels_per_mm ** 2)
    
    print("\n--- Mask Accuracy Report ---")
    print(f"{'Image':<15} | {'Exp Area':<10} | {'Mask Area':<10} | {'Diff':<10} | {'Status'}")
    print("-" * 65)
    
    
    for item in renaming_map:
        img_name = item['new_image_name'] 
        mask_path = os.path.join(MASKS_DIR, item['new_mask_name'])
        orig_base = os.path.basename(item['original_path'])
        
        expected_row = ann_df[ann_df['image_name'] == orig_base]
        if expected_row.empty:
            if orig_base.endswith('.jpeg'):
                 expected_row = ann_df[ann_df['image_name'] == orig_base.replace('.jpeg', '.jpg')]
            
        if expected_row.empty:
            continue
            
        expected_area = float(expected_row.iloc[0]['actual_area_mm2'])
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            mask = (mask > 127).astype(np.uint8)
            pixel_area = np.sum(mask)
            calc_area = pixel_area * mm2_per_pixel
            
            diff = calc_area - expected_area
            status = "OK"
            if abs(diff) > 0.5: 
                status = "WARNING"
            if abs(diff) > 1.0:
                status = "BAD"
                
            if status != "OK":
                print(f"{img_name:<15} | {expected_area:<10.3f} | {calc_area:<10.3f} | {diff:<10.3f} | {status}")

if __name__ == "__main__":
    main()
