import torch
import numpy as np
from typing import Dict, Any, Tuple


class ToTensor:
    """
    Convert a tuple of (image, mask) numpy arrays to PyTorch tensors.
    """
    def __call__(self, sample: Tuple[np.ndarray, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = sample

        # rasterio reads as (bands, height, width), which matches PyTorch's expected format (C, H, W)
        # However, we ensure they are tensors.
        
        # Convert image to float32 and replace NaNs with 0
        image_tensor = torch.from_numpy(image).float()
        image_tensor = torch.nan_to_num(image_tensor, nan=0.0)
        
        # Convert mask to long (for classification labels) or float (for binary)
        # Masks are typically (1, H, W). Clamp to [0, 1] to avoid -1 or 255 values
        mask_tensor = torch.from_numpy(mask).float()
        mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)
        
        return image_tensor, mask_tensor

class RandomHorizontalFlip:
    """Horizontally flip the given tuple of (image, mask) randomly with a given probability."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = sample
        if torch.rand(1) < self.p:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        return image, mask

class RandomVerticalFlip:
    """Vertically flip the given tuple of (image, mask) randomly with a given probability."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = sample
        if torch.rand(1) < self.p:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        return image, mask


class NormalizeS1:
    """
    Normalize Sentinel-1 SAR imagery.
    Currently just a placeholder for basic scaling if needed.
    """
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
        
    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = sample
        # Simple z-score normalization
        # Depending on SAR data, we might need specific normalization (e.g., decibel scaling)
        # For now, leaving it as a pass-through or simple normalization
        # image = (image - self.mean) / self.std
        return image, mask


class Compose:
    """
    Compose multiple transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample: Tuple[Any, Any]) -> Tuple[Any, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample
