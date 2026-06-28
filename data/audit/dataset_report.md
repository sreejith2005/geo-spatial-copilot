# Dataset Audit Report

## 1. Dataset Inventory
* **Sen1Floods11 Subset**: Located in `data/raw/sen1floods11/subset/`. Contains filtered data separated by country into `india/` and `spain/` directories.
* **Metadata file**: Located at `data/raw/sen1floods11/subset_metadata.csv`. Tracks file split, event, and local file paths.
* **PDF Corpus**: Located in `data/documents/`. Contains flood and disaster management plans relevant to India.
* **Boundary layers**: Located in `data/raw/boundaries/maps-master/`. Contains various administrative boundaries (e.g., states, districts, country).
* **SRTM elevation data**: Located in `data/raw/elevation/srtm/`. Contains SRTM data components.

## 2. File Counts
* **Sen1Floods11 Subset**: 60 files (30 for India, 30 for Spain)
* **Metadata file**: 1 file (CSV)
* **PDF Corpus**: 4 files
* **Boundary layers**: 725 files (across subdirectories)
* **SRTM elevation data**: 4 files (constituting 1 ESRI Shapefile: `.shp`, `.shx`, `.dbf`, `.prj`)
* **Total Audited Files**: 794

## 3. File Sizes
* **Sen1Floods11 Subset**: 0.42 MB
* **PDF Corpus**: 60.55 MB
* **Boundary layers**: 484.03 MB
* **SRTM elevation data**: 0.45 MB
* **Total Approximate Size**: ~545 MB

## 4. Data Formats
* **Raster imagery**: `.tif` (GeoTIFF) for Sen1Floods11 layers.
* **Vector features**: `.shp`, `.shx`, `.dbf`, `.prj` (Shapefiles) and `.geojson` for boundaries and elevation.
* **Documents**: `.pdf` for domain knowledge.
* **Tabular data**: `.csv` for metadata tracking.

## 5. CRS Information
* **SRTM Elevation**: Verified directly from the `.prj` file as **EPSG:4326 (WGS 84)**.
* **Boundary Layers**: Typically distributed as **EPSG:4326 (WGS 84)** (DataMeet standard convention).
* **Sen1Floods11 Subset**: GeoTIFFs expected to be **EPSG:4326 (WGS 84)** based on the standard projection of the Sen1Floods11 public dataset.

## 6. Missing Files
* **Missing Source Imagery**: The `sen1floods11/subset` directories currently **only contain `LabelHand`** (ground truth label) files. The corresponding Sentinel-1 (`S1`) imagery and Quality Control (`QC`) layers are completely missing.
* **Missing Raster DEMs**: The SRTM directory does not contain actual raster elevation models (`.tif` or `.hgt`). It currently only contains a vector shapefile (`srtm.shp`).

## 7. Potential Issues
* **Critical Missing S1 Data**: Without the core `S1` radar layers, it is impossible to train models or perform flood inference. The subset is incomplete.
* **Incorrect Elevation Format**: Elevation processing requires continuous raster data (DEMs). The current SRTM data is in vector format (likely a bounding box index or contour set, given its small 0.45 MB size) and cannot be used directly as an elevation mask.
* **Absolute Paths in Metadata**: The `subset_metadata.csv` hardcodes absolute file paths (e.g., `c:/Users/user/Downloads/geospatial copilot/...`). This breaks portability if the project is moved to another location or shared with another developer. The pipeline should be updated to use relative paths.
* **Missing Environment Dependencies**: Essential geospatial Python libraries such as `rasterio` and system tools like `gdal` are not installed in the `.venv`, preventing automated and programmatic extraction of raster metadata (like checking TIF CRS).
