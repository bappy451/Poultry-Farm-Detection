# Poultry Farm Geospatial Detection Pipeline - Documentation Index

Welcome to the comprehensive technical documentation for the Poultry Farm Geospatial Detection Pipeline. 

This directory contains an extensive set of reports covering every aspect of the project, from high-level architecture to the deep mathematical reasoning behind our machine learning choices.

## Document Manifest

1. **[01 - Executive Summary and Project Overview](./01_Executive_Summary_and_Overview.md)**
   - High-level overview of the project domain.
   - Core capabilities and the transition from Jupyter notebooks to a production-grade CLI.
   - Outline of the pipeline flow and project status.

2. **[02 - Geospatial Data Processing](./02_Geospatial_Data_Processing.md)**
   - Detailed breakdown of GeoTIFF slicing (edge-snapped, no-padding).
   - `rasterio` chunked I/O mechanics for OOM protection.
   - PyTorch `DataLoader` design, augmentation, and data leakage prevention.

3. **[03 - Machine Learning and Self-Supervised Learning (SSL)](./03_Machine_Learning_and_SSL.md)**
   - Deep dive into Masked Image Modeling (MIM) for Vision Transformers.
   - Deep dive into Sparse Masked Modeling (SparK) for CNNs (YOLO/ResNet).
   - Explanation of initialization strategies, mask ratios (75%), and parameter-efficient tuning (LoRA, NF4).

4. **[04 - Downstream Fine-Tuning and Model Evaluation](./04_Downstream_Fine_Tuning.md)**
   - Fine-tuning protocols for binary segmentation.
   - Comprehensive performance comparison (Precision, Recall, F1, mAP50) between YOLO11 and YOLO26.
   - Tangible results proving the superiority of the SparK SSL transfer.

5. **[05 - GIS Post-Processing, Clustering, and Analytics](./05_GIS_PostProcessing_and_Analytics.md)**
   - Mechanics of spatial deduplication (Centroid Suppression and Mask NMS).
   - Farm entity generation using DBSCAN clustering.
   - Vector projection and county-level hydrological runoff analytics.

6. **[06 - Engineering Design Choices and Future Roadmap](./06_Engineering_Design_and_Future_Plans.md)**
   - Architectural constraints, stateful grid-search design, and CLI wrapper safety.
   - Information on test-driven development (140+ unit tests).
   - Future roadmap, including SAM2, multi-year temporal analysis, and cloud deployment.
