# Engineering Design Choices and Future Roadmap

Migrating an exploratory data science project from a collection of Jupyter Notebooks into a reliable, terminal-runnable software package requires critical architectural decisions. The `poultry_geo` pipeline is engineered around safety, scalability, and statefulness.

## 1. Core Engineering & Design Choices

### CLI Wrapper Architecture and Logging
Instead of relying solely on pure Python `argparse`, the project orchestrates execution via OS-specific shell wrappers (`.sh`, `.cmd`, `.ps1`).
- **Safety First:** The Windows `.cmd` wrappers safely invoke PowerShell, bypassing strict execution policies without permanently altering the host machine's security posture.
- **Append-Only Auditing:** Every command executed is intercepted by a logging wrapper. It writes the exact command line, execution path, start/finish timestamps, and elapsed duration to a global `command.log`. If a script fails, the entire Python traceback—or PowerShell error stack—is appended, ensuring no failure is silent.

### Avoiding Naive Memory Loading
A common pitfall in geospatial machine learning is treating massive GeoTIFFs like standard JPEGs.
- **Chunked I/O:** The pipeline exclusively uses the `rasterio` library's `Window` definitions. At no point is a full GeoTIFF loaded into RAM. Both inference reads and stitching writes act purely on small, localized chunked buffers, completely circumventing Out-Of-Memory (OOM) operating system crashes.

### Stateful Execution Over Multiprocessing (For ML)
While multiprocessing is excellent for I/O tasks like slicing, it is disastrous for GPU training.
- **Sequential Safety:** Massive hyperparameter grid searches (`train_ssl_grid`, `train_segmentation`) avoid multiprocessing to prevent VRAM overallocation.
- **Statefulness:** Experiments run sequentially. To protect against lost compute time (e.g., from Windows Update reboots), the training loops aggressively checkpoint the optimizer, scheduler, and RNG state per epoch. Re-running the identical CLI command will instantly resume the exact epoch that was interrupted.

### Abstracting SSL Complexity
Implementing Sparse Convolutions (SparK) or Masked Image Modeling (MIM) typically requires highly intrusive modifications to underlying model libraries.
- **Encapsulation:** The `ssl_models.py` module encapsulates this complexity via wrapper classes (`RemoteSensingMIMModel`, `BandAdapter`). Users define intent via YAML configurations, and the backend dynamically injects sparse layers or LoRA bottlenecks into Ultralytics YOLO or `timm` Vision Transformers without requiring forks of those massive external libraries.

### Test-Driven Assurance
Geospatial transformations are uniquely prone to silent mathematical failures (e.g., silently swapping Latitude and Longitude).
- **Unit Testing:** A comprehensive PyTest suite of over 140 tests rigidly locks down the mathematical behavior of polygon intersections, coordinate transforms (CRS shifts), multiprocessing thread locks, and tensor shapes.

---

## 2. Current Project Status

The pipeline is actively functional and has achieved its primary goal: converting the brittle notebook workflows into a robust, terminal-runnable Python package.

- **Package Deployment:** The `src/poultry_geo/` library is fully installable and operational.
- **CLI Accessibility:** End-to-end processing (from slicing to KML export) is fully accessible via the CLI wrappers.
- **Model Training:** Automated self-supervised pipelines (SparK, MIM) and downstream YOLO/DETR segmentation fine-tuning are fully operational with stateful grid search capabilities.
- **Notebook Archival:** As detailed in the documentation, interactive visualization, exploratory one-off plots, and custom manuscript analyses remain intentionally within the `notebook/` and `notebook_new/` directories as the project's research record.

---

## 3. Planned Work and Future Roadmap

While the core pipeline is complete, several advanced features are planned for future iterations:

### Advanced Model Architectures
- **SAM2 Integration:** The current pipeline uses the original Segment Anything Model (SAM) for polygonal refinement. Future work will fully integrate SAM2 to provide dramatically faster and more accurate zero-shot masking.
- **Multi-Modal Foundation Models:** Expanding the SparK self-supervised paradigm to ingest elevation data (LiDAR/DEM) alongside the standard RGB-NIR optical bands.

### Temporal and Hydrological Analytics
- **Time-Series Analysis:** Upgrading the pipeline to ingest NAIP imagery across multiple years (e.g., 2018, 2020, 2022) to automatically detect farm construction, expansion, or abandonment over time.
- **Advanced Topography:** Enhancing the runoff calculation by incorporating localized slope, elevation, and soil-type datasets to provide highly accurate watershed contamination modeling.

### Scale and Optimization
- **Distributed Cloud Inference:** Transitioning the sliding-window inference logic from single-node execution to distributed cloud frameworks (like Apache Spark or AWS Batch) to process entire states simultaneously.
- **NF4 Quantization:** The pipeline currently supports experimental 4-bit (`nf4`) precision loading. Future work will benchmark inference throughput and accuracy degradation to establish quantization as the default for rapid deployment on memory-constrained edge hardware.