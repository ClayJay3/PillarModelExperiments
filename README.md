# **Sparse PillarNeXt 3D Object Detector**

This repository contains a single-file, production-grade implementation of a LiDAR 3D Object Detector inspired by the **PillarNeXt** architecture.

It achieves high performance on sparse point clouds by leveraging **Sparse Convolutions (spconv)** to maintain high resolution (**0.075m voxel size**) without running out of GPU memory.

---

## 🧠 **Architecture Overview**

The model processes raw LiDAR point clouds into 3D bounding box detections using the following pipeline:

### **1. Sparse Pillar Encoder**

* Voxelizes raw point clouds into vertical columns (“pillars”)
* Encodes them using a PointNet-like MLP
* Uses sparse tensors to avoid computing empty space

### **2. Sparse ResNet Backbone**

* A 4-stage ResNet-34–style backbone
* Uses **Submanifold Sparse Convolutions**
* Extracts multi-scale features efficiently

### **3. ASPP Neck (Atrous Spatial Pyramid Pooling)**

* Expands the receptive field using dilated convolutions
* Captures context for both large and small objects

### **4. Upsampling CenterHead**

* Dense detection head
* Upsamples features to high resolution
* Predicts:

  * Heatmaps (object centers)
  * Regression maps (box dimensions, rotation)

---

## 🛠️ **Prerequisites & Installation**

This project uses **pipenv** for dependency management.

---

### **1. Install Pipenv**

```bash
pip install --user pipenv
```

---

### **2. Create Environment & Install Dependencies**

Run the following commands to set up the environment.

> **Note on PyTorch & CUDA:**
> Ensure your PyTorch install matches your CUDA version.

```bash
# 1. Initialize the shell
pipenv shell

# 2. Install Standard Dependencies
pipenv install numpy torch nuscenes-devkit pyquaternion open3d
```

---

### **3. Install Spconv (Critical Step)**

This project requires **spconv** for sparse convolutions.

Check your CUDA version:

```bash
nvcc --version
```

Install the matching spconv version:

```bash
# For CUDA 12.x
pipenv install spconv-cu120

# For CUDA 11.8
pipenv install spconv-cu118

# For CUDA 11.3
pipenv install spconv-cu113
```

---

## 📂 **Dataset Setup**

Register and download the **nuScenes v1.0-mini** dataset from nuscenes.org.

Extract it so the directory looks like:

```
PillarModelExperiments/
├── run_detection_pipeline.py
└── v1.0-mini/
    ├── maps/
    ├── samples/
    ├── sweeps/
    └── v1.0-mini/
        ├── attribute.json
        ├── calibrated_sensor.json
        ...
```

If your dataset is located elsewhere, change the `--data_root` argument in the commands below.

---

## 🚀 **Usage Commands**

### **1. Training**

Trains from scratch and saves:

* `ckpt_last.pth` (latest)
* `best.pth` (best F1 score)

```bash
python run_detection_pipeline.py \
  --mode train \
  --dataset nuscenes \
  --data_root ./v1.0-mini \
  --nusc_version v1.0-mini \
  --epochs 40
```

---

### **2. Evaluation**

Runs inference + computes metrics (Recall/Precision/F1):

```bash
python run_detection_pipeline.py \
  --mode infer \
  --dataset nuscenes \
  --data_root ./v1.0-mini \
  --nusc_version v1.0-mini \
  --checkpoint ./checkpoints/best.pth \
  --evaluate
```

---

### **3. Visualization**

Runs inference and opens an interactive 3D visualization (Open3D).

* **Black boxes:** Ground truth
* **Red boxes:** Model predictions
* **Blue box:** Detection range (“canvas”)

```bash
python run_detection_pipeline.py \
  --mode infer \
  --dataset nuscenes \
  --data_root ./v1.0-mini \
  --nusc_version v1.0-mini \
  --checkpoint ./checkpoints/best.pth \
  --visualize
```

---

## ⚙️ **Configuration**

You can tweak advanced hyperparameters at the top of `run_detection_pipeline.py`:

```python
CONFIG = {
    'x_range': (-51.2, 51.2),  # Detection range (meters)
    'y_range': (-51.2, 51.2),
    'grid_size': 0.075,        # Voxel size (smaller = higher res, more memory)
    'batch_size': 2,           # Reduce if you hit CUDA OOM
    'lr': 0.001,               # Learning rate
}
```
