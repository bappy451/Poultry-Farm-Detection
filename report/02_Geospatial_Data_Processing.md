# Geospatial Data Processing

Processing satellite and aerial imagery introduces significant engineering and computational challenges. Traditional deep learning computer vision pipelines assume that individual inputs (like JPEGs) are small enough to fit entirely into RAM. However, geospatial rasters like NAIP GeoTIFFs can easily exceed several gigabytes, encapsulate multi-band spectral data (e.g., Red, Green, Blue, Near-Infrared), and embed complex coordinate reference systems (CRS).

The `poultry_geo` package abstracts this complexity, enabling seamless data flow from raw rasters into PyTorch tensors and back to georeferenced maps.

## 1. Slicing Strategy (`geotiff.py`)
Neural networks require fixed-size inputs (e.g., 224x224, 256x256, 512x512). The pipeline handles this through high-performance slicing.
- **Edge-Snapped No-Padding Slicing:** A naive slicer will zero-pad the edges of an image to fit a strict tile grid, introducing artificial black borders. These borders severely degrade model performance. The `geotiff.py` module mathematically shifts the final grid windows to snap exactly to the valid edge of the raster, ensuring every extracted tile contains 100% valid imagery.
- **Multiprocessing (`--workers auto`):** To manage the immense I/O bottleneck of reading multi-gigabyte files, the slicer relies on tile-level multiprocessing. This allows simultaneous extractions of `.tif`, `.png`, or `.jpg` formats, drastically accelerating dataset preparation.

## 2. Streaming Inference (`inference.py`)
To perform inference across entire counties or states, loading the entire mosaic into GPU VRAM—or even system RAM—is impossible.
- **Sliding-Window Inference:** The module natively streams chunks directly from the disk-backed GeoTIFF using `rasterio`. It pulls a tile, processes it through the segmentation model, records any detected polygons in georeferenced space, and immediately releases the memory before moving to the next stride.
- **Strided Overlap:** Poultry houses situated on the boundary of two tiles might be sliced in half, leading to false negatives. The sliding window supports an overlapping stride (e.g., `window-size 3000` with `stride 2500`). This ensures that every geographic point is evaluated within the center of a receptive field at least once.

## 3. Memory-Efficient Stitching
When researchers need to generate intermediate visualizations or rebuild massive probability maps:
- **Windowed Writing:** The `stitch` CLI command utilizes `rasterio`'s windowed writing API. It accepts a directory of smaller output tiles and places them directly into their correct spatial coordinate block on a massive, disk-backed GeoTIFF. This circumvents Out-Of-Memory (OOM) errors that plague `numpy.vstack()` based scripts.

## 4. PyTorch Data Loading (`naip_data.py`)
To train models effectively, a highly optimized PyTorch `Dataset` and `DataLoader` are implemented.
- **Dynamic File Handling:** `NAIPDataset` traverses recursively nested directory structures, natively ingesting RGB JPEGs/PNGs, multi-band GeoTIFFs, or uncompressed `CHW/HWC` numpy arrays.
- **Augmentation Pipeline:** Training samples undergo synchronized random cropping, horizontal/vertical flipping across all active spectral bands, and channel-aware color jittering.
- **Deterministic Validation:** The validation pipeline utilizes strict center-crops to ensure consistent metric tracking across epochs.
- **Leakage Prevention:** Adjacent geographic tiles from the same source GeoTIFF are highly correlated. The dataset strictly splits Train and Validation datasets at the *parent source directory level*, preventing data leakage and artificially inflated accuracy metrics.
- **Precision Modes:** Tensors can be dynamically cast during the dataloading phase to `fp32`, `fp16`, or `bf16` to accommodate mixed-precision training.