# GIS Post-Processing and Environmental Analytics

Extracting bounding boxes from raw pixels is only the first half of the geospatial detection pipeline. Because inference relies on strided, overlapping sliding windows, the raw tabular output contains significant artifacting and duplication. The `postprocess.py` and `gis.py` modules exist to clean, consolidate, and analyze this data to produce production-ready GIS artifacts.

## 1. Spatial Deduplication
When a large, 500-foot poultry house spans across three individual image tiles, the model will output three separate (and overlapping) detections. To rectify this, two advanced suppression techniques are employed:

- **Centroid Suppression:** The pipeline computes the geographic center (centroid) of every bounding box. If multiple centroids fall within a tight geographic radius (e.g., 5 meters), the pipeline assumes they represent the same object and suppresses the duplicates, keeping the prediction with the highest confidence score.
- **Mask Non-Maximum Suppression (NMS):** For segmentation masks, the pipeline computes the Intersection-over-Union (IoU) of the actual geospatial polygons. If two masks overlap by more than a specified threshold (e.g., 50%), they are dissolved and merged into a single, unified polygon geometry representing the complete roof.

## 2. Farm Entity Clustering
Poultry operations rarely consist of a single barn. They are typically arrayed in densely packed parallel sets of 4 to 12 houses. To understand environmental impact, researchers need to quantify "farms" rather than just individual roofs.

- **DBSCAN Clustering:** The `cluster-farms` command applies DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to the deduplicated houses.
- **Mechanism:** Using parameters like `eps-house=50.0` (search radius in meters) and `min-samples=2`, the algorithm groups proximal houses.
- **Output:** It generates a new logical entity (a "Farm ID") and computes a geographic centroid for the entire clustered operation, providing a macroscopic view of agricultural density.

## 3. GIS Vector Export (`gis.py` & `conversion.py`)
To be useful for policymakers, the pandas DataFrames must be converted into standardized geographic formats readable by software like QGIS or ArcGIS.

- **ESRI Shapefiles:** Detections can be natively exported as Bounding Box shapefiles, Point shapefiles (centroids), or complex Polygon shapefiles that perfectly trace the outline of the detected roofs.
- **KML Export:** For rapid stakeholder review, the pipeline exports detections to Keyhole Markup Language (`.kml`), allowing the boundaries to be overlaid natively in Google Earth.
- **Coordinate Reference Systems (CRS):** All data is re-projected on the fly to a localized, metrical CRS (e.g., `EPSG:26916` for the Southeastern US) to ensure area calculations are strictly accurate in square meters, rather than distorted degrees of latitude/longitude.

## 4. County-Level Environmental Analytics
The ultimate goal of the project is to quantify the potential environmental impact of these concentrated animal feeding operations on local water systems.

- **County Aggregation:** The pipeline spatially intersects farm point locations with official county boundary datasets to tally operations by region.
- **Runoff Screening:** By calculating the geometric surface area of the merged polygon roofs, the module estimates the total impervious surface area. It then multiplies this area by historical county precipitation data to calculate an estimated volume of rainfall runoff that could carry contaminants into the watershed.
- **NASS Correction & Bootstrapping:** Deep learning models are not perfect (they yield false positives and false negatives). The `analytics.py` module performs statistical bootstrapping and corrects the calculated runoff figures against official, ground-truth agricultural census data provided by the National Agricultural Statistics Service (NASS).