import pandas as pd
import os
import shutil

ANNOTATIONS_PATH = 'data/annotations.csv'
EXCEL_PATH = '1119-20260304-Results-D0-D3.xlsx'
MAPPING_PATH = 'data/processed/mapping.csv'

def main():
    print("Step 1: Loading Data...")
    if not os.path.exists(ANNOTATIONS_PATH) or not os.path.exists(EXCEL_PATH):
        print("Error: Files not found.")
        return

    shutil.copy(ANNOTATIONS_PATH, ANNOTATIONS_PATH + '.bak')
    print(f"Backed up annotations to {ANNOTATIONS_PATH}.bak")

    ann_df = pd.read_csv(ANNOTATIONS_PATH)
    excel_df = pd.read_excel(EXCEL_PATH)
    
    
    print("Step 2: Building Excel Lookup Dictionary...")
    excel_areas = {}
    label_cols = [c for c in excel_df.columns if 'Label' in str(c)]
    
    for col in label_cols:
        col_loc = excel_df.columns.get_loc(col)
        if col_loc + 1 >= len(excel_df.columns): continue
        area_col = excel_df.columns[col_loc + 1]
        
        for idx, label in excel_df[col].items():
            if pd.notna(label):
                area = excel_df.iloc[idx, col_loc + 1]
                if pd.notna(area):
                    excel_areas[str(label).strip()] = float(area)

    print(f"Loaded {len(excel_areas)} valid area records from Excel.")

    print("Step 3: Updating Annotations CSV...")
    updated_count = 0
    
    for idx, row in ann_df.iterrows():
        img_name = str(row['image_name']).strip()
        
        if img_name in excel_areas:
            new_area = excel_areas[img_name]
            if abs(float(row['actual_area_mm2']) - new_area) > 0.001:
                ann_df.at[idx, 'actual_area_mm2'] = new_area
                updated_count += 1
        else:
            name_no_ext = os.path.splitext(img_name)[0]
            found = False
            for ext in ['.jpg', '.jpeg', '.png']:
                alt_name = name_no_ext + ext
                if alt_name in excel_areas:
                    new_area = excel_areas[alt_name]
                    if abs(float(row['actual_area_mm2']) - new_area) > 0.001:
                        ann_df.at[idx, 'actual_area_mm2'] = new_area
                        updated_count += 1
                    found = True
                    break
            
            if not found:
                print(f"Warning: Could not find match for {img_name} in Excel. Keeping original value.")

    print(f"Updated {updated_count} records.")
    
    ann_df.to_csv(ANNOTATIONS_PATH, index=False)
    print(f"Successfully saved updated annotations to {ANNOTATIONS_PATH}")

if __name__ == "__main__":
    main()
