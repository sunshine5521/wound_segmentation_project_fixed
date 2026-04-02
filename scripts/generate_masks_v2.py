import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def generate_masks(image_dir, mask_dir, csv_path):
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
        
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    print(f"Found {len(files)} images in {image_dir}")
    print(f"Generating masks to {mask_dir}...")
    
    records = []
    
    for f in tqdm(files):
        img_path = os.path.join(image_dir, f)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {f}")
            continue
            
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
            
        mask_name = os.path.splitext(f)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        cv2.imwrite(mask_path, mask)
        
        records.append({'image_name': f})
        
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(records)} masks.")
    print(f"Saved list to {csv_path}")

if __name__ == "__main__":
    IMAGE_DIR = r"data/processed/images_raw"
    MASK_DIR = r"data/processed/masks"
    CSV_PATH = r"data/processed/train.csv"
    
    generate_masks(IMAGE_DIR, MASK_DIR, CSV_PATH)
