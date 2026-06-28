# Vision Dataset Report: Sen1Floods11 Subset

## Overview
- **Data Source:** `data/raw/sen1floods11/subset`
- **Total Chips:** 60

## Metadata Validation
- **S1 Image Dimensions:** [(512, 512)]
- **Label Dimensions:** [(512, 512)]
- **CRS:** ['EPSG:4326']
- **S1 Band Count:** [2] (Expected: 2)
- **Label Band Count:** [1] (Expected: 1)

## Class Balance (Label Masks)
- **Total Pixels:** 15,728,640
- **Water Pixels (Class 1):** 1,826,876 (11.61%)
- **Non-Water Pixels (Class 0):** 12,578,485 (79.97%)
- **Invalid/Unclassified Pixels (Class -1):** 1,323,279 (8.41%)

**Raw Class Counts:**
- Class `-1`: 1,323,279 pixels
- Class `0`: 12,578,485 pixels
- Class `1`: 1,826,876 pixels

## Sample Statistics (S1 Data)
*Note: Excludes -9999 NoData values*
- **Valid S1 Pixels Processed:** 15,687,345
- **VV Band:** Mean = -10.3819, Std Dev = 5.1567
- **VH Band:** Mean = -17.4601, Std Dev = 5.6243
