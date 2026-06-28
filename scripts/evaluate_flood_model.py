import os
import sys
import json
import logging
from pathlib import Path

# Ensure the app module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from app.vision.split import get_splits
from app.vision.dataset import Sen1Floods11Dataset
from app.vision.transforms import Compose, ToTensor, NormalizeS1
from app.vision.model import LightweightUNet
from app.vision.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    root_dir = "data/raw/sen1floods11/subset"
    model_path = "models/checkpoints/best_model.pt"
    batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        logger.error("Please run the training script first.")
        return

    # 1. Get Splits
    logger.info("Getting test split...")
    _, _, test_samples = get_splits(root_dir)
    
    if not test_samples:
        logger.error("No test samples found.")
        return

    # 2. Dataset and DataLoader
    test_transforms = Compose([
        ToTensor(),
        NormalizeS1()
    ])
    
    test_dataset = Sen1Floods11Dataset(samples=test_samples, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Load Model
    logger.info("Loading model...")
    model = LightweightUNet(n_channels=2, n_classes=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Evaluation Loop
    logger.info("Starting evaluation...")
    all_metrics = {"dice": [], "iou": [], "precision": [], "recall": []}
    
    with torch.no_grad():
        for images, masks, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device).float()
            
            logits = model(images)
            metrics = calculate_metrics(logits, masks)
            
            for k, v in metrics.items():
                all_metrics[k].append(v)
                
    # 5. Print Summary
    logger.info("Evaluation Complete!")
    logger.info("="*30)
    logger.info("Test Set Metrics Summary:")
    logger.info("-" * 30)
    
    final_metrics = {}
    for k, v in all_metrics.items():
        mean_val = np.mean(v)
        logger.info(f"{k.capitalize()}: {mean_val:.4f}")
        final_metrics[k] = float(mean_val)
    logger.info("="*30)
    
    metrics_path = Path("models/checkpoints/final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
    logger.info(f"Final metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
