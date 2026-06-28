import os
import glob
import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import pyproj
from shapely.ops import transform
import reverse_geocoder as rg
import warnings
import uuid

# Suppress Pyproj warnings
warnings.filterwarnings("ignore", category=UserWarning)

def compute_chip_metadata(file_path, districts_gdf=None):
    try:
        with rasterio.open(file_path) as src:
            crs = src.crs
            bounds = src.bounds
            
            # Create a shapely box for bounds
            geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            # We need centroid in WGS84 for reverse geocoding
            project_to_wgs84 = pyproj.Transformer.from_crs(crs, pyproj.CRS("EPSG:4326"), always_xy=True).transform
            geom_wgs84 = transform(project_to_wgs84, geom)
            
            centroid_lon, centroid_lat = geom_wgs84.centroid.x, geom_wgs84.centroid.y
            
            # We need area in sq meters. Project to a suitable equal area (Web Mercator 3857 for simplicity, though not perfect)
            project_to_3857 = pyproj.Transformer.from_crs(crs, pyproj.CRS("EPSG:3857"), always_xy=True).transform
            geom_3857 = transform(project_to_3857, geom)
            area_sqm = geom_3857.area
            
            # Reverse geocode country
            res = rg.search((centroid_lat, centroid_lon))
            country = res[0]['cc'] if res else "Unknown"
            
            district_name = "Unknown"
            state_name = "Unknown"
            
            if districts_gdf is not None:
                chip_gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs=crs)
                if chip_gdf.crs != districts_gdf.crs:
                    if districts_gdf.crs is None:
                        districts_gdf.set_crs(epsg=4326, inplace=True)
                    if chip_gdf.crs is None:
                        chip_gdf.set_crs(epsg=4326, inplace=True)
                    chip_gdf = chip_gdf.to_crs(districts_gdf.crs)
                    
                intersection = gpd.overlay(chip_gdf, districts_gdf, how='intersection')
                if not intersection.empty:
                    intersection['area'] = intersection.geometry.area
                    largest = intersection.sort_values(by='area', ascending=False).iloc[0]
                    dist_col = next((c for c in largest.index if 'dist' in c.lower()), None)
                    state_col = next((c for c in largest.index if 'st_nm' in c.lower() or 'state' in c.lower()), None)
                    if dist_col: district_name = str(largest[dist_col])
                    if state_col: state_name = str(largest[state_col])
            
            return {
                "chip_id": str(uuid.uuid4()),
                "file_path": file_path,
                "crs": str(crs),
                "bounds_wgs84": list(geom_wgs84.bounds), # minx, miny, maxx, maxy
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
                "area_sqm": area_sqm,
                "area_km2": area_sqm / 1e6,
                "district": district_name,
                "state": state_name,
                "country": country,
                "region": res[0]['admin1'] if res else "Unknown",
                "width": src.width,
                "height": src.height
            }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    base_dir = "data/raw/sen1floods11/subset"
    output_parquet = "data/chip_catalog.parquet"
    shp_path = "data/raw/boundaries/maps-master/maps-master/Districts/Census_2011/2011_Dist.shp"
    
    print(f"Loading DataMeet district boundaries from {shp_path}...")
    districts_gdf = None
    if os.path.exists(shp_path):
        districts_gdf = gpd.read_file(shp_path)
    else:
        print("Warning: District shapefile not found. Spatial intersection will be skipped.")

    print(f"Scanning for TIFF files in {base_dir}...")
    tif_files = []
    if os.path.exists(base_dir):
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".tif"):
                    tif_files.append(os.path.join(root, file))
                    
    print(f"Found {len(tif_files)} files. Computing spatial metadata...")
    records = []
    for f in tif_files:
        # replace backslashes for consistency
        f = f.replace('\\\\', '/')
        rec = compute_chip_metadata(f, districts_gdf)
        if rec:
            records.append(rec)
            
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(output_parquet, engine='fastparquet')
        print(f"Successfully wrote {len(records)} records to {output_parquet}")
    else:
        print("No valid records found.")

if __name__ == "__main__":
    main()
