import cv2
import numpy as np
import os
from tqdm import tqdm

def generate_masks(image_dir, mask_dir, threshold_type='hsv_red'):
    """
    Auto-generate rough masks based on simple color/intensity rules.
    This is a starting point for "obvious wounds".
    """
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
        
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    files = []
    for root, dirs, filenames in os.walk(image_dir):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in valid_extensions:
                files.append(os.path.join(root, f))
    
    print(f"Found {len(files)} images in {image_dir} (recursive)")
    print(f"Generating masks to {mask_dir}...")
    
    for img_path in tqdm(files):
        f = os.path.basename(img_path)
        
        try:
            from PIL import Image
            pil_img = Image.open(img_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"PIL failed for {img_path}: {e}")
            try:
                img_data = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            except Exception as e2:
                print(f"imdecode failed for {img_path}: {e2}")
                continue
            
        if img is None:
            print(f"Failed to decode {img_path}")
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
        
    print("Done! Masks generated.")
    print("IMPORTANT: Please check the masks manually! Bad masks will confuse the model.")

if __name__ == "__main__":
    IMAGE_DIR = r"data/images/1119-20260304-D0-D3"
    MASK_DIR = r"data/masks"
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory not found: {IMAGE_DIR}")
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            IMAGE_DIR = config['data']['image_dir']
            print(f"Trying config path: {IMAGE_DIR}")

    generate_masks(IMAGE_DIR, MASK_DIR)
