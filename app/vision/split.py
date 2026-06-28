import random
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def get_splits(
    root_dir: str, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15, 
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Deterministically find all paired S1Hand and LabelHand files and split them.
    
    Args:
        root_dir: Root directory containing Sen1Floods11 subset
        train_ratio: Proportion of training data
        val_ratio: Proportion of validation data
        test_ratio: Proportion of testing data
        seed: Random seed for deterministic shuffling
        
    Returns:
        Tuple containing train_samples, val_samples, and test_samples lists.
        Each list contains dictionaries with 's1' and 'label' paths.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory {root_dir} does not exist.")
        
    samples = []
    # Find all S1Hand files
    s1_files = list(root_path.rglob("*_S1Hand.tif"))
    
    for s1_path in s1_files:
        # Construct expected label path
        label_name = s1_path.name.replace("_S1Hand.tif", "_LabelHand.tif")
        label_path = s1_path.parent / label_name
        
        if label_path.exists():
            samples.append({
                's1': str(s1_path),
                'label': str(label_path)
            })
            
    if not samples:
        logger.warning(f"No paired samples found in {root_dir}")
        return [], [], []
        
    # Sort for determinism before shuffling
    samples.sort(key=lambda x: x['s1'])
    
    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(samples)
    
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train+n_val]
    test_samples = samples[n_train+n_val:]
    
    logger.info(f"Split {n_total} total samples into: "
                f"{len(train_samples)} train, "
                f"{len(val_samples)} val, "
                f"{len(test_samples)} test")
                
    return train_samples, val_samples, test_samples
