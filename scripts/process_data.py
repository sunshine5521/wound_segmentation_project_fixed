import pandas as pd
import os

def process_excel(excel_path, output_csv):
    if not os.path.exists(excel_path):
        print(f"Error: Excel file {excel_path} not found.")
        return

    try:
        df = pd.read_excel(excel_path)
        print(f"Loaded Excel with {len(df)} rows.")
    except Exception as e:
        print(f"Failed to read Excel: {e}")
        return

    image_names = []
    areas = []

    cols = df.columns
    for i in range(len(cols)):
        col = cols[i]
        if 'Label' in str(col):
            if i + 1 < len(cols):
                area_col = cols[i+1]
                if 'Area' in str(area_col):
                    subset = df[[col, area_col]].dropna(subset=[col])
                    for _, row in subset.iterrows():
                        img_name = str(row[col]).strip()
                        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            img_name += '.jpg' 
                        
                        
                        image_names.append(img_name)
                        areas.append(row[area_col])

    out_df = pd.DataFrame({
        'image_name': image_names,
        'actual_area_mm2': areas
    })
    
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {len(out_df)} records to {output_csv}")

if __name__ == '__main__':
    excel_file = '1119-20260304-Results-D0-D3.xlsx'
    csv_file = 'data/annotations.csv'
    process_excel(excel_file, csv_file)
