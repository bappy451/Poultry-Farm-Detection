# 02 - Geospatial Data Processing

Processing high-resolution aerial imagery (such as NAIP GeoTIFFs) introduces unique engineering challenges that standard Deep Learning pipelines (designed for small JPEGs) cannot natively handle. The `poultry_geo` pipeline abstracts these challenges to ensure memory safety, coordinate accuracy, and maximal hardware utilization.

## 1. GeoTIFF Slicing Strategy (`geotiff.py`)
Neural networks require fixed-size inputs (e.g., 224x224, 256x256, 512x512). Generating these tiles from multi-gigabyte rasters is a major bottleneck.

### Edge-Snapped No-Padding Slicing
A naive image slicer divides a raster into a strict grid. If the edge of the raster does not divide perfectly by the tile size, the slicer injects zero-padding (black borders). These artificial straight lines severely confuse neural networks and degrade model performance during training and inference. 
- **The Solution:** The `geotiff.py` module mathematically shifts the final tile windows backward to "snap" exactly to the valid edge of the raster. Every extracted tile contains 100% valid imagery, ensuring no artificial artifacts are introduced into the dataset.

### Tile-Level Multiprocessing
To handle extreme I/O loads, slicing operations are parallelized across multiple CPU cores (`--workers auto`). Instead of processing one massive GeoTIFF entirely before moving to the next, the pipeline parallelizes the extraction of individual tile chunks, saturating NVMe/SSD read bandwidth and significantly accelerating dataset generation. Outputs are configurable to `.tif`, `.png`, or `.jpg`.

## 2. Streaming Inference on Massive Rasters (`inference.py`)
Evaluating a 5GB county-level mosaic is impossible if the file is loaded into RAM all at once.

### Chunked I/O and Windowed Execution
The module relies heavily on `rasterio`'s `Window` API. It streams geographic chunks directly from the disk-backed GeoTIFF, passes the localized numpy array to the PyTorch/YOLO inference engine, maps the resulting bounding boxes to geographic coordinates, and then releases the memory. This guarantees the pipeline can run on systems with highly constrained RAM.

### Strided Overlap for Detection Fidelity
If a poultry house happens to lie exactly on the boundary between two adjacent tile crops, the model will see two halves of a building and fail to detect it (a false negative). 
To prevent this, the inference engine supports an overlapping sliding window (e.g., extracting `3000x3000` windows, but striding by `2500` pixels). This guarantees that every geographical coordinate passes through the center of the model's receptive field at least once, ensuring complete detection coverage.

## 3. Reconstructive Stitching
When generating intermediate visualization maps, the pipeline provides a `stitch` command. Similar to slicing, it avoids `numpy.vstack()` (which attempts to hold the final stitched image in RAM). Instead, it pre-allocates an empty GeoTIFF matching the original dimensions on disk, and streams individual tiles directly into their correct spatial blocks, keeping memory overhead flat regardless of the output image size.

## 4. PyTorch Dataloader (`naip_data.py`)
Training modern networks requires a highly optimized PyTorch `Dataset` and `DataLoader`.

### Multi-Modal Dynamic Handling
The `NAIPDataset` natively traverses recursively nested directory structures. It intelligently handles RGB `.jpg` or `.png` files, multi-band `.tif` files (e.g., Red, Green, Blue, Near-Infrared), and uncompressed `.npy` arrays, normalizing all inputs to PyTorch's expected `[C, H, W]` tensor format.

### Geographically Aware Augmentation & Splits
- **Synchronized Augmentation:** During training, samples undergo synchronized random cropping, spatial flipping (horizontal and vertical), and channel-aware color jittering across all spectral bands. Validation samples undergo deterministic center crops to stabilize metric tracking.
- **Leakage Prevention:** Adjacent geographical tiles from the same source raster are highly correlated. If an 80/20 split is applied randomly, identical visual features "leak" into the validation set, artificially inflating accuracy metrics. `NAIPDataset` prevents this by splitting data strictly at the *parent source directory level*.

### Precision Controls
To optimize GPU memory and throughput, the Dataloader supports on-the-fly casting to user-defined precisions (`fp32`, `fp16`, or `bf16`). Experimental `nf4` (4-bit NormalFloat) quantization is also supported to maximize batch sizes on constrained consumer hardware.
