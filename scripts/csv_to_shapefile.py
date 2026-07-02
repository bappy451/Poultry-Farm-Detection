import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import argparse
import os
import warnings

# Suppress the shapefile 10-character column limit warnings since it truncates automatically
warnings.filterwarnings("ignore", message="Column names longer than 10 characters will be truncated when saved to ESRI Shapefile.")

def csv_to_shapefile(csv_path, shp_path, epsg_code):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, comment='#')
    
    if len(df) == 0:
        print("CSV is empty. No shapefile will be created.")
        return

    required_cols = ['x1_easting', 'y1_northing', 'x2_easting', 'y2_northing']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in CSV. Make sure you are using the output from run_naip_detection.py")

    print(f"Creating polygon geometries for {len(df)} detections...")
    
    # Create bounding box polygons for the target CRS
    geometries = [
        box(row['x1_easting'], row['y1_northing'], row['x2_easting'], row['y2_northing'])
        for _, row in df.iterrows()
    ]
    
    # Optional: Shorten some long column names to prevent ugly shapefile truncation
    rename_map = {
        'latitude_wgs84': 'lat_wgs84',
        'longitude_wgs84': 'lon_wgs84',
        'bbox_area_sqft': 'area_sqft',
        'bbox_width_ft': 'width_ft',
        'bbox_height_ft': 'height_ft'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Drop raw pixel and src coordinates as they aren't useful in GIS
    cols_to_drop = [c for c in df.columns if c.endswith('_px') or c.endswith('_src')]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Create GeoDataFrame
    crs_str = f"EPSG:{epsg_code}"
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=crs_str)
    
    os.makedirs(os.path.dirname(shp_path), exist_ok=True)
    
    print(f"Saving ArcGIS shapefile to {shp_path} with CRS {crs_str}...")
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    print("Done! You can now drag and drop this .shp file into ArcGIS Pro or ArcMap.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO NAIP detection CSV to ArcGIS Polygon Shapefile")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV")
    parser.add_argument("--output-shp", required=True, help="Path to output Shapefile (.shp)")
    parser.add_argument("--epsg", type=int, default=6444, help="Target EPSG code for the shapefile (default: 6444 for GA East State Plane)")
    
    args = parser.parse_args()
    csv_to_shapefile(args.input_csv, args.output_shp, args.epsg)
