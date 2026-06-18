# 06 - Engineering Design Choices and Future Roadmap

Transitioning the Poultry Farm Detection project from an exploratory data science paradigm (Jupyter Notebooks) into a production-grade, terminal-runnable software package (`poultry_geo`) required rigorous architectural decisions. The pipeline is heavily engineered around fault tolerance, hardware maximization, and mathematical safety.

## 1. Core Engineering & Design Choices

### CLI Wrapper Architecture and System Logging
Instead of relying solely on a naked Python entrypoint, execution is orchestrated via OS-specific shell wrappers (`.sh`, `.cmd`, `.ps1`).
- **Policy Safety:** On Windows environments, executing raw scripts can trigger restrictive execution policies. The `.cmd` wrappers act as launchers that safely invoke PowerShell, bypassing strict execution policies transparently without permanently altering the host machine's global security posture.
- **Append-Only Auditing:** Every command executed through the wrappers is intercepted. The script logs the exact command line arguments, the execution path, start/finish timestamps, and elapsed duration to a global `command.log`. Crucially, if a script fails, the full Python traceback (or PowerShell error stack) is caught and appended to the log, ensuring silent failures do not occur during multi-day inference runs.

### Out-Of-Memory (OOM) Protection
A widespread failure point in geospatial machine learning is treating large rasters like standard images.
- **Exclusive Chunked I/O:** The pipeline strictly avoids loading entire GeoTIFFs into RAM. Using `rasterio`'s windowed reading and writing API, operations act on small, localized chunk buffers. Slicing, inference, and stitching are entirely shielded from OOM operating system crashes, regardless of file size.

### Stateful ML Execution
While multiprocessing is excellent for I/O tasks like slicing tiles, applying it to GPU training leads to disastrous VRAM overallocation.
- **Sequential Safety:** Massive hyperparameter grid searches (`train_ssl_grid`, `train_segmentation`) avoid multiprocessing. Permutations are evaluated sequentially.
- **Fault-Tolerant Statefulness:** Because experiments take days, the training loops rigorously checkpoint the optimizer, scheduler, and RNG state per epoch. If a run is interrupted by a power failure or OS reboot, re-running the exact identical CLI command will instantly resume the exact epoch that was interrupted.

### Modular Abstraction of SSL
Implementing Sparse Convolutions (SparK) or Masked Image Modeling (MIM) typically requires intrusive forks of massive external libraries.
- **Encapsulation:** The `ssl_models.py` module encapsulates this complexity via dynamic wrapper classes. Users define intent via YAML configurations, and the backend dynamically injects sparse layers or LoRA bottlenecks into Ultralytics YOLO or `timm` Vision Transformers—keeping the core environment dependencies pristine.

### Test-Driven Mathematical Assurance
Geospatial transformations are uniquely prone to silent logic errors (e.g., accidentally swapping Latitude and Longitude variables during projection).
- **Unit Testing:** A comprehensive PyTest suite of over 140 tests rigidly locks down the mathematical behavior of polygon intersections, coordinate transforms (CRS shifts), multiprocessing thread locks, and tensor dimension geometries.

---

## 2. Planned Work and Future Roadmap

While the pipeline currently delivers state-of-the-art segmentation and GIS vector export, several advanced capabilities are planned for upcoming iterations:

### Advanced Model Architectures
- **SAM2 Integration:** The current pipeline utilizes the original Segment Anything Model (SAM) as an optional post-processing refinement tool (`sam-refine`). Future iterations will fully integrate SAM2 to deliver drastically faster and more accurate zero-shot mask generation.
- **Multi-Modal Foundation Models:** Expanding the SparK self-supervised paradigm to natively ingest elevation data (LiDAR/DEM) directly alongside standard RGB-NIR optical bands, providing the models with a 3D structural understanding of the terrain.

### Temporal and Hydrological Analytics
- **Time-Series Analysis:** Upgrading the inference pipeline to ingest NAIP imagery spanning multiple years (e.g., 2018, 2020, 2022). This will allow the automated detection of farm construction, expansion, or abandonment over time.
- **Advanced Topography Analytics:** Enhancing the runoff calculation by incorporating localized slope, elevation, and specific soil-type datasets to provide highly accurate, granular watershed contamination modeling.

### Scale and Deployment Optimization
- **Distributed Cloud Inference:** Transitioning the sliding-window inference logic from single-node execution to distributed cloud computing frameworks (such as Apache Spark or AWS Batch), enabling the simultaneous processing of entire states.
- **NF4 Quantization Benchmark:** The pipeline currently includes experimental support for 4-bit (`nf4`) precision loading. Future work will extensively benchmark inference throughput versus accuracy degradation to establish quantization as the standard default for rapid, memory-constrained edge deployment.
