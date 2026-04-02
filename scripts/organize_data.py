import os
import shutil
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import uuid

def sanitize_name(name):
    return "".join([c if c.isalnum() or c in ".-_" else "_" for c in name])

def generate_mask_from_img(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    mask = cv2.bitwise_or(mask1, mask2)
    
    if np.sum(mask) < 100:
        s_channel = hsv[:, :, 1]
        _, mask_otsu = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask_otsu

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask_clean = np.zeros_like(mask)
        cv2.drawContours(mask_clean, [largest_contour], -1, 255, thickness=cv2.FILLED)
        mask = mask_clean
        
    return mask

def main():
    source_root = r"data/images/1119-20260304-D0-D3"
    target_img_dir = r"data/processed/images"
    target_mask_dir = r"data/processed/masks"
    target_csv = r"data/processed/train.csv"
    
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_mask_dir, exist_ok=True)
    
    records = []
    
    print("Scanning files...")
    for root, dirs, files in os.walk(source_root):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                original_path = os.path.join(root, f)
                
                parent = os.path.basename(root)
                prefix = "Unknown"
                if "D0" in parent: prefix = "D0"
                elif "D1" in parent: prefix = "D1"
                elif "D3" in parent: prefix = "D3"
                
                unique_id = str(uuid.uuid4())[:8]
                new_filename = f"{prefix}_{sanitize_name(f)}_{unique_id}.png" 
                
                temp_path = f"temp_{unique_id}.png"
                try:
                    shutil.copy2(original_path, temp_path)
                    
                    img = cv2.imread(temp_path)
                    if img is None:
                        print(f"Failed to read {f}")
                        if os.path.exists(temp_path): os.remove(temp_path)
                        continue
                        
                    mask = generate_mask_from_img(img)
                    
                    target_img_path = os.path.join(target_img_dir, new_filename)
                    target_mask_path = os.path.join(target_mask_dir, new_filename)
                    
                    cv2.imwrite(target_img_path, img)
                    cv2.imwrite(target_mask_path, mask)
                    
                    records.append({'image_name': new_filename})
                    
                except Exception as e:
                    print(f"Error processing {f}: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
    df = pd.DataFrame(records)
    df.to_csv(target_csv, index=False)
    print(f"Processed {len(df)} images.")
    print(f"Data saved to {target_img_dir}")
    print(f"CSV saved to {target_csv}")

if __name__ == "__main__":
    main()
