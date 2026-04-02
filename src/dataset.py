import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

class WoundDataset(Dataset):
    def __init__(self, config, transform=None, mode='train'):
        self.config = config
        self.transform = transform
        self.mode = mode
        
        self.df = pd.read_csv(config['data']['csv_path'])
        self.image_dir = config['data']['image_dir']
        self.mask_dir = config['data']['mask_dir']
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image_name']
        
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_name = image_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
             raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        if mask.ndim == 2:
            mask = torch.unsqueeze(mask, 0)
            
        return {
            'image': image,
            'mask': mask,
            'image_name': image_name
        }

def get_transforms(config, mode='train'):
    img_size = config['data']['img_size'] 
    
    if mode == 'train' and config['augmentation']['use_aug']:
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=config['augmentation']['rotation_limit'], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=config['augmentation']['brightness_limit'],
                contrast_limit=config['augmentation']['contrast_limit'], 
                p=0.5
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

def create_dataloaders(config):
    df = pd.read_csv(config['data']['csv_path'])
    
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    
    pass

class WoundDataset(Dataset):
    def __init__(self, dataframe, config, transform=None):
        self.df = dataframe
        self.config = config
        self.transform = transform
        self.image_dir = config['data']['image_dir']
        self.mask_dir = config['data']['mask_dir']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image_name']
        
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_name = os.path.splitext(image_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
             raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        if mask.ndim == 2:
            mask = torch.unsqueeze(mask, 0)
            
        return {
            'image': image,
            'mask': mask,
            'image_name': image_name
        }

def create_dataloaders(config):
    df = pd.read_csv(config['data']['csv_path'])
    
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_transform = get_transforms(config, mode='train')
    val_transform = get_transforms(config, mode='val')
    
    train_dataset = WoundDataset(train_df, config, transform=train_transform)
    val_dataset = WoundDataset(val_df, config, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader
