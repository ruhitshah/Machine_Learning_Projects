# 3D Point Cloud Registration and Alignment Using Open3D

This project implements and compares multiple advanced 3D point cloud registration algorithms using the Open3D library. The goal is to robustly align two point clouds and quantitatively evaluate the alignment accuracy using RMSE against a reference image.

---

## Features

- Adaptive voxel downsampling for efficient point cloud preprocessing.
- Normal estimation to improve registration quality.
- Coarse-to-fine registration pipelines:
  - Iterative Closest Point (ICP) based registration.
  - Fast Global Registration (FGR) using feature matching.
  - RANSAC-based feature matching registration.
- ICP refinement using the Point-to-Plane method for fine alignment.
- Quantitative evaluation of alignment accuracy via RMSE as a percentage of maximum error relative to a reference image.
- Visualization of aligned point clouds projected onto 2D images for qualitative assessment.

---

## Requirements

- Python 3.x
- [Open3D](http://www.open3d.org/) (`pip install open3d`)
- NumPy
- Matplotlib
- scikit-learn (for optional nearest neighbors if needed)

---

## File Descriptions

- `icp_registration.py` - Implementation of coarse and refined ICP registration with timing and RMSE evaluation.
- `fast_global_registration.py` - Pipeline combining Fast Global Registration (FGR) and ICP refinement.
- `ransac_registration.py` - RANSAC-based registration with ICP refinement and RMSE analysis.
- `/data/` - Folder containing example point cloud files (`a.pcd`, `b.pcd`) and the reference image (`a_b.png`).

---

## Usage

1. Place your source and target point cloud files (`a.pcd`, `b.pcd`) in the `/data/` directory.
2. Place the reference image (`a_b.png`) for RMSE comparison in the same directory.
3. Run the desired registration script:
   ```bash
   python icp_registration.py
   python fast_global_registration.py
   python ransac_registration.py
