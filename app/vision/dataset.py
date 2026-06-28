import os
import glob
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset
import rasterio

class Sen1Floods11Dataset(Dataset):
    """
    PyTorch Dataset for loading Sen1Floods11 Sentinel-1 and Label pairs.
    Focuses on S1Hand (SAR data) and LabelHand (Hand-labeled masks).
    """
    def __init__(self, 
                 samples: list, 
                 transform: Optional[Callable] = None):
        """
        Args:
            samples (list): List of dictionaries containing 's1' and 'label' paths.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: Any) -> Tuple[Any, Any, Dict[str, Any]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        s1_path = self.samples[idx]['s1']
        label_path = self.samples[idx]['label']

        with rasterio.open(s1_path) as src_s1:
            # Read all bands (expected 2: VV, VH)
            image = src_s1.read()
            s1_meta = src_s1.meta.copy()

        with rasterio.open(label_path) as src_label:
            # Read all bands (expected 1: Mask)
            mask = src_label.read()
            label_meta = src_label.meta.copy()

        metadata = {
            's1_path': s1_path,
            'label_path': label_path,
            's1_bands': s1_meta['count'],
            'label_bands': label_meta['count'],
            's1_shape': (s1_meta['height'], s1_meta['width']),
            'label_shape': (label_meta['height'], label_meta['width']),
            'crs': s1_meta['crs'] if s1_meta['crs'] else None,
            'crs_match': s1_meta['crs'] == label_meta['crs'],
            'shape_match': (s1_meta['height'], s1_meta['width']) == (label_meta['height'], label_meta['width'])
        }

        # Apply transforms if any
        if self.transform:
            image, mask = self.transform((image, mask))

        return image, mask, metadata
