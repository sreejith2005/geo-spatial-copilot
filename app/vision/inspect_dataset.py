import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Ensure app is in path so we can import from app.vision
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.vision.dataset import Sen1Floods11Dataset
from app.vision.split import get_splits

def inspect_dataset(data_root: str, report_out: str):
    print(f"Inspecting dataset at {data_root}...")
    train_samples, _, _ = get_splits(data_root, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    dataset = Sen1Floods11Dataset(samples=train_samples)
    
    total_chips = len(dataset)
    print(f"Found {total_chips} matching chip pairs.")
    
    if total_chips == 0:
        print("No chips found. Exiting.")
        return

    # Trackers
    shapes_s1 = set()
    shapes_label = set()
    crs_set = set()
    s1_bands_set = set()
    label_bands_set = set()
    
    total_pixels = 0
    water_pixels = 0
    non_water_pixels = 0
    unclassified_pixels = 0
    
    # Value counts for labels to see exact class balance
    label_value_counts = defaultdict(int)
    
    # Basic statistics for S1 (VV, VH)
    vv_sum = 0.0
    vh_sum = 0.0
    vv_sq_sum = 0.0
    vh_sq_sum = 0.0
    valid_s1_pixels = 0
    
    for idx in range(total_chips):
        image, mask, meta = dataset[idx]
        
        shapes_s1.add(meta['s1_shape'])
        shapes_label.add(meta['label_shape'])
        crs_set.add(meta['crs'])
        s1_bands_set.add(meta['s1_bands'])
        label_bands_set.add(meta['label_bands'])
        
        # S1 Stats
        # Assuming band 0 is VV, band 1 is VH based on Sen1Floods11
        if image.shape[0] >= 2:
            vv = image[0]
            vh = image[1]
            # Replace NaNs or invalid if any? We assume valid for now or filter
            # Sen1Floods11 -9999 is often no data
            valid_mask = (vv != -9999) & (vh != -9999) & ~np.isnan(vv) & ~np.isnan(vh)
            valid_vv = vv[valid_mask]
            valid_vh = vh[valid_mask]
            
            vv_sum += valid_vv.sum()
            vh_sum += valid_vh.sum()
            vv_sq_sum += (valid_vv ** 2).sum()
            vh_sq_sum += (valid_vh ** 2).sum()
            valid_s1_pixels += valid_vv.size
            
        # Label Stats
        # Sen1Floods11 labels typically:
        # 0: no water / unclassified
        # 1: water
        # -1: invalid
        # Let's count unique values
        unique, counts = np.unique(mask, return_counts=True)
        for val, count in zip(unique, counts):
            label_value_counts[val] += count
            total_pixels += count
            
            if val == 1:
                water_pixels += count
            elif val == 0:
                non_water_pixels += count
            elif val == -1:
                unclassified_pixels += count
                
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{total_chips} chips...")

    print("Processing complete. Compiling statistics...")
    
    # Calculate S1 means and std devs
    vv_mean = vv_sum / valid_s1_pixels if valid_s1_pixels > 0 else 0
    vh_mean = vh_sum / valid_s1_pixels if valid_s1_pixels > 0 else 0
    
    vv_var = (vv_sq_sum / valid_s1_pixels) - (vv_mean ** 2) if valid_s1_pixels > 0 else 0
    vh_var = (vh_sq_sum / valid_s1_pixels) - (vh_mean ** 2) if valid_s1_pixels > 0 else 0
    
    vv_std = np.sqrt(max(0, vv_var))
    vh_std = np.sqrt(max(0, vh_var))

    # Calculate percentages
    pct_water = (water_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    pct_non_water = (non_water_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    pct_unclassified = (unclassified_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    # Generate Markdown Report
    report = f"""# Vision Dataset Report: Sen1Floods11 Subset

## Overview
- **Data Source:** `{data_root}`
- **Total Chips:** {total_chips}

## Metadata Validation
- **S1 Image Dimensions:** {list(shapes_s1)}
- **Label Dimensions:** {list(shapes_label)}
- **CRS:** {list(crs_set)}
- **S1 Band Count:** {list(s1_bands_set)} (Expected: 2)
- **Label Band Count:** {list(label_bands_set)} (Expected: 1)

## Class Balance (Label Masks)
- **Total Pixels:** {total_pixels:,}
- **Water Pixels (Class 1):** {water_pixels:,} ({pct_water:.2f}%)
- **Non-Water Pixels (Class 0):** {non_water_pixels:,} ({pct_non_water:.2f}%)
- **Invalid/Unclassified Pixels (Class -1):** {unclassified_pixels:,} ({pct_unclassified:.2f}%)

**Raw Class Counts:**
"""
    for val, count in sorted(label_value_counts.items()):
        report += f"- Class `{val}`: {count:,} pixels\n"
        
    report += f"""
## Sample Statistics (S1 Data)
*Note: Excludes -9999 NoData values*
- **Valid S1 Pixels Processed:** {valid_s1_pixels:,}
- **VV Band:** Mean = {vv_mean:.4f}, Std Dev = {vv_std:.4f}
- **VH Band:** Mean = {vh_mean:.4f}, Std Dev = {vh_std:.4f}
"""

    out_path = Path(report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        f.write(report)
        
    print(f"Report saved successfully to {out_path}")

if __name__ == "__main__":
    DATA_ROOT = "data/raw/sen1floods11/subset"
    REPORT_OUT = "data/audit/vision_dataset_report.md"
    
    # Paths are relative to the project root, assuming script is run from project root
    inspect_dataset(DATA_ROOT, REPORT_OUT)
