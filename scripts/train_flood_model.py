import os
import sys
import json
import logging
from pathlib import Path

# Ensure the app module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from app.vision.split import get_splits
from app.vision.dataset import Sen1Floods11Dataset
from app.vision.transforms import Compose, ToTensor, NormalizeS1, RandomHorizontalFlip, RandomVerticalFlip
from app.vision.model import LightweightUNet
from app.vision.losses import DiceBCELoss
from app.vision.training import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    root_dir = "data/raw/sen1floods11/subset"
    checkpoint_dir = "models/checkpoints"
    
    # Small batch size to fit in 8GB RAM
    batch_size = 2
    epochs = 10
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 1. Get Splits
    logger.info("Splitting dataset...")
    train_samples, val_samples, test_samples = get_splits(root_dir)
    
    if not train_samples:
        logger.error("No training samples found. Please check the dataset path.")
        return

    # 2. Transforms
    # Light augmentations for training
    train_transforms = Compose([
        ToTensor(),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        NormalizeS1() # Add normalizer if needed
    ])
    
    val_transforms = Compose([
        ToTensor(),
        NormalizeS1()
    ])

    # 3. Datasets and DataLoaders
    logger.info("Initializing datasets...")
    train_dataset = Sen1Floods11Dataset(samples=train_samples, transform=train_transforms)
    val_dataset = Sen1Floods11Dataset(samples=val_samples, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Model, Loss, Optimizer
    logger.info("Initializing model...")
    model = LightweightUNet(n_channels=2, n_classes=1)
    
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Training
    logger.info("Starting training...")
    
    # Save config
    config = {
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "device": str(device)
    }
    with open(Path(checkpoint_dir) / "training_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # Save splits
    splits_record = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }
    with open(Path(checkpoint_dir) / "split.json", "w") as f:
        json.dump(splits_record, f, indent=4)
        
    trainer = Trainer(model, criterion, optimizer, device, checkpoint_dir=checkpoint_dir)
    trainer.fit(train_loader, val_loader, epochs=epochs)
    
    logger.info(f"Training finished. Best model saved in {checkpoint_dir}")

if __name__ == "__main__":
    main()
