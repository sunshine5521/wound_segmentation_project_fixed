import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.dataset import create_dataloaders
from src.model import create_model
from src.utils import MetricMonitor, save_checkpoint, compute_dice_score, compute_iou

class Trainer:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("警告: CUDA 不可用，使用 CPU 进行训练。")
        
        self.model = create_model(self.config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['training']['lr'])
        self.train_loader, self.val_loader = create_dataloaders(self.config)
        
        self.checkpoint_dir = self.config['paths']['checkpoint_dir']
        self.log_dir = self.config['paths'].get('log_dir', 'logs') 
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, 'training_log.csv')
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write('epoch,train_loss,train_dice,val_loss,val_dice\n')

    def log_metrics(self, epoch, train_metrics, val_metrics):
        train_loss = train_metrics['Loss']['total'] / train_metrics['Loss']['count']
        train_dice = train_metrics['Dice']['total'] / train_metrics['Dice']['count']
        val_loss = val_metrics['Loss']['total'] / val_metrics['Loss']['count']
        val_dice = val_metrics['Dice']['total'] / val_metrics['Dice']['count']
        
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{train_dice:.4f},{val_loss:.4f},{val_dice:.4f}\n")

    def compute_loss(self, pred, target):
        criterion = torch.nn.BCEWithLogitsLoss()
        return criterion(pred, target)

    def train_epoch(self, epoch):
        self.model.train()
        metric_monitor = MetricMonitor()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [训练]")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.compute_loss(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            probs = torch.sigmoid(outputs)
            dice = compute_dice_score(probs, masks)
            iou = compute_iou(probs, masks)
            
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Dice", dice)
            metric_monitor.update("IoU", iou)
            
            progress_bar.set_postfix(
                Loss=metric_monitor.get_avg("Loss"),
                Dice=metric_monitor.get_avg("Dice")
            )
        return metric_monitor.metrics

    def validate(self, epoch):
        self.model.eval()
        metric_monitor = MetricMonitor()
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} [验证]")
        
        with torch.no_grad():
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.compute_loss(outputs, masks)
                
                probs = torch.sigmoid(outputs)
                dice = compute_dice_score(probs, masks)
                iou = compute_iou(probs, masks)
                
                metric_monitor.update("Loss", loss.item())
                metric_monitor.update("Dice", dice)
                metric_monitor.update("IoU", iou)
                
                progress_bar.set_postfix(
                    Loss=metric_monitor.get_avg("Loss"),
                    Dice=metric_monitor.get_avg("Dice")
                )
        return metric_monitor.metrics

    def train(self):
        best_dice = 0.0
        patience = self.config['training']['patience']
        patience_counter = 0
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            self.log_metrics(epoch, train_metrics, val_metrics)
            
            val_dice = val_metrics["Dice"]["total"] / val_metrics["Dice"]["count"]
            
            print(f"Epoch {epoch} 总结:")
            print(f"训练 - Loss: {train_metrics['Loss']['total']/train_metrics['Loss']['count']:.4f}, Dice: {train_metrics['Dice']['total']/train_metrics['Dice']['count']:.4f}")
            print(f"验证 - Loss: {val_metrics['Loss']['total']/val_metrics['Loss']['count']:.4f}, Dice: {val_dice:.4f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0
                save_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics, save_path)
                print(f"已保存最佳模型 (Dice: {best_dice:.4f})")
            else:
                patience_counter += 1
                print(f"未提升。耐心值: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                print("触发早停。")
                break
