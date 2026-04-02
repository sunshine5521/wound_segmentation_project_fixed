import torch
import cv2
import numpy as np
import pandas as pd
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.model import create_model

class WoundAnalyzer:
    def __init__(self, checkpoint_path, config, device=None):
        self.device = device
        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"推理使用 GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print("推理使用 CPU")
        
        self.config = config
        self.model = create_model(config)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"从 {checkpoint_path} 加载模型")
        else:
            print(f"警告: 未找到检查点 {checkpoint_path}。使用随机权重。")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = A.Compose([
            A.Resize(height=config['data']['img_size'][0], width=config['data']['img_size'][1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def preprocess(self, image):
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0) 
        return image_tensor

    def analyze_image(self, image_path, threshold=0.5):
        original_image = cv2.imread(image_path)
        if original_image is None:
             raise FileNotFoundError(f"Image not found: {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.preprocess(original_image).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits)
            mask = (probs > threshold).float()
            
        mask_np = mask.squeeze().cpu().numpy() 
        
        pixel_area = np.sum(mask_np)
        
        mask_np_resized = cv2.resize(mask_np, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        actual_area = None
        if 'pixels_per_mm' in self.config['data']:
            pixels_per_mm = self.config['data']['pixels_per_mm']
            mm2_per_pixel = 1 / (pixels_per_mm ** 2)
            actual_area = pixel_area * mm2_per_pixel
            actual_area = round(actual_area, 2)
        
        return {
            'original_image': original_image,
            'mask': mask_np_resized,
            'pixel_area': pixel_area,
            'actual_area_mm2': actual_area,
            'image_path': image_path,
            'logits': cv2.resize(logits.squeeze().cpu().numpy(), (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR) 
        }

    def visualize_results(self, result):
        import matplotlib.pyplot as plt
        
        img = result['original_image']
        mask = result['mask']
        
        
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("原始图像")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.imshow(mask_resized, alpha=0.5, cmap='jet') 
        plt.title(f"预测 (面积: {result['pixel_area']} px)")
        plt.axis('off')
        
        plt.show()

    def batch_analyze(self, image_dir, output_csv):
        results = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
        
        print(f"在 {image_dir} 中找到 {len(files)} 张图像")
        
        for f in tqdm(files, desc="处理批次"):
            path = os.path.join(image_dir, f)
            try:
                res = self.analyze_image(path)
                results.append({
                    'image_name': f,
                    'pixel_area': res['pixel_area'],
                    'actual_area_mm2': res['actual_area_mm2']
                })
            except Exception as e:
                print(f"处理 {f} 时出错: {e}")
                
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        return df
