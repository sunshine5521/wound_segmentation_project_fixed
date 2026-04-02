import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

def read_image(path):
    try:
        pil_img = Image.open(path)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        try:
            img = cv2.imread(path)
            if img is not None:
                return img
        except:
            pass
        print(f"Failed to read {path}: {e}")
        return None

def generate_masks(source_dir, target_img_dir, target_mask_dir, csv_path):
    if not os.path.exists(target_img_dir):
        os.makedirs(target_img_dir)
    if not os.path.exists(target_mask_dir):
        os.makedirs(target_mask_dir)
        
    files = sorted([f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))])
    
    print(f"Found {len(files)} files in {source_dir}")
    print(f"Processing images and generating masks...")
    
    records = []
    
    for f in tqdm(files):
        src_path = os.path.join(source_dir, f)
        
        img = read_image(src_path)
        if img is None:
            continue
            
        name_root = os.path.splitext(f)[0]
        new_filename = name_root + ".jpg"
        
        dst_img_path = os.path.join(target_img_dir, new_filename)
        cv2.imwrite(dst_img_path, img)
        
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
            
        mask_filename = name_root + ".png"
        mask_path = os.path.join(target_mask_dir, mask_filename)
        cv2.imwrite(mask_path, mask)
        
        records.append({'image_name': new_filename})
        
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Processed {len(records)} images.")
    print(f"Saved list to {csv_path}")

if __name__ == "__main__":
    SOURCE_DIR = r"data/processed/images_raw"
    TARGET_IMG_DIR = r"data/processed/images"
    TARGET_MASK_DIR = r"data/processed/masks"
    CSV_PATH = r"data/processed/train.csv"
    
    generate_masks(SOURCE_DIR, TARGET_IMG_DIR, TARGET_MASK_DIR, CSV_PATH)
