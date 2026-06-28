import os
import logging
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from .metrics import calculate_metrics

logger = logging.getLogger(__name__)

class Trainer:
    """
    Handles the training and validation loops for PyTorch models.
    """
    def __init__(self, model, criterion, optimizer, device, checkpoint_dir="models/checkpoints"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.model.to(self.device)

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc="Training")
        for images, masks, _ in pbar:
            images = images.to(self.device)
            # Ensure mask matches model output shape and type
            masks = masks.to(self.device).float()
            
            self.optimizer.zero_grad()
            logits = self.model(images)
            
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return epoch_loss / len(dataloader)

    def validate_epoch(self, dataloader):
        self.model.eval()
        epoch_loss = 0.0
        all_metrics = {"dice": [], "iou": [], "precision": [], "recall": []}
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating")
            for images, masks, _ in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device).float()
                
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                epoch_loss += loss.item()
                
                metrics = calculate_metrics(logits, masks)
                for k, v in metrics.items():
                    all_metrics[k].append(v)
                    
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        avg_loss = epoch_loss / len(dataloader)
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        return avg_loss, avg_metrics

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def fit(self, train_loader, val_loader, epochs=10):
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Metrics: Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, "
                        f"Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self.checkpoint_dir / "best_model.pt"
                self.save_checkpoint(best_path)
                logger.info(f"New best model saved with val loss {val_loss:.4f}")
                
        logger.info("Training complete.")
