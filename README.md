# 🌌 3D Surface Reconstruction from Point Clouds

This repository implements two methods to reconstruct 3D surfaces from sparse point clouds using classical geometric distance and neural network-based signed distance functions, with mesh extraction via marching cubes.

---

## Methods

### 1. Geometric Method
- Computes the signed distance of a 3D query point to the **tangent plane of its nearest surface point**.
- Uses surface normals provided in the point cloud to determine distance direction.
- Implemented in `naiveReconstruction.py`.
- Produces polygon meshes using the **marching cubes** algorithm (`scikit-image`).

### 2. Neural Network Method (Neural SDF)
- Implements a **fully-connected PyTorch neural network** to approximate the signed distance function.
- Architecture:
  - **Input:** 3D point coordinates `(x, y, z)`
  - **8 fully-connected layers**, hidden vector size 512
  - **Skip connection**: After the 4th layer, 509-dim hidden vector concatenated with original 3D point to form 512-dim
  - First 7 layers: **weight normalization**, **Leaky ReLU**, **dropout (0.1)**
  - Final layer: outputs **SDF value** with **tanh** activation
- Training:
  - Samples points along surface normals: `p' = p + ε * n`, `ε ~ N(0, 0.052)`
  - Uses **clamped L1 loss** to compare predicted and true SDF
  - Optimized with **AdamW**
- Evaluation:
  - Meshes generated via marching cubes on the trained implicit function
  - Supports point clouds like bunny (500 & 1000 points) and sphere

---

## Dependencies

```bash
pip install numpy
pip install scikit-image
pip install scikit-learn
pip install trimesh
pip install pyglet<2
pip install torch
