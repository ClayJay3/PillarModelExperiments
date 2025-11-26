#!/usr/bin/env python3
"""
run_detection_pipeline.py

This script implements a single-file, end-to-end 3D object detection pipeline 
named 'PillarNeXt'. It utilizes sparse convolutions (via spconv) to process 
LiDAR point clouds efficiently.

The architecture is designed to handle high-resolution grids (0.075m) without 
running out of GPU memory (OOM) by leveraging the sparsity of LiDAR data. It 
follows a CenterPoint-like detection head structure.

Prerequisites:
    - spconv-cu1XX (matching your CUDA version)
    - nuscenes-devkit
    - open3d (for visualization)

Usage:
  # Train the model
  python run_detection_pipeline.py --mode train --dataset nuscenes \
      --data_root ./v1.0-mini --nusc_version v1.0-mini --epochs 40

  # Inference and Evaluation
  python run_detection_pipeline.py --mode infer --dataset nuscenes \
      --data_root ./v1.0-mini --nusc_version v1.0-mini --checkpoint ./checkpoints/best.pth --evaluate --visualize
"""

import os
import argparse
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# --- Check Spconv ---
# We verify that spconv is installed because standard PyTorch Conv3d is too memory intensive 
# for the voxel resolutions required in modern autonomous driving stacks.
try:
    import spconv.pytorch as spconv
except ImportError:
    raise ImportError("Spconv not found! Please install: pip install spconv-cu118 (or cu120 depending on CUDA version)")

# --- Configuration ---
# Global configuration dictionary controlling data ranges, model hyperparameters, and training settings.
CONFIG = {
    # The physical range of the LiDAR point cloud in meters [min, max].
    'x_range': (-51.2, 51.2),
    'y_range': (-51.2, 51.2),
    'z_range': (-5.0, 3.0),
    
    # The size of a single voxel/pillar in meters. 0.075m is standard for SOTA methods.
    'grid_size': 0.075, 
    
    # Training batch size.
    'batch_size': 2,
    
    # Number of CPU workers for data loading.
    'num_workers': 4,
    
    # Learning rate for the AdamW optimizer.
    'lr': 0.001,
    
    # Weight decay for regularization.
    'weight_decay': 0.01,
}

# --- Utilities ---

def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mkdir(path):
    """
    Creates a directory if it does not already exist.
    
    Args:
        path (str): The file path to the directory.
    """
    os.makedirs(path, exist_ok=True)

# --- Visualization Helper ---

def get_box_lineset(box, color):
    """
    Creates an Open3D LineSet object representing a 3D bounding box.
    
    Args:
        box (list or np.array): Bounding box parameters [x, y, z, l, w, h, yaw].
        color (list): RGB color values [r, g, b] normalized to 0-1.
        
    Returns:
        o3d.geometry.LineSet: The geometric representation of the box edges.
    """
    import open3d as o3d
    # Unpack box parameters.
    x, y, z, l, w, h, yaw = box[:7]
    
    # Define the 8 corners of the box relative to the center (0,0,0).
    # Length (l) aligns with x, Width (w) with y, Height (h) with z.
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    
    # Create the rotation matrix from the yaw angle (around Z-axis).
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    # Stack corners into a 3x8 matrix and rotate them.
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d = np.dot(R, corners_3d)
    
    # Translate the rotated corners to the global position (x, y, z).
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    # Define the 12 lines connecting the 8 corners of the cube.
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    
    # Construct the Open3D objects.
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_3d.T)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
    
    return line_set

def visualize_sample(points, gt_boxes, pred_boxes):
    """
    Visualizes a point cloud along with ground truth and predicted bounding boxes using Open3D.
    
    Args:
        points (np.array): Point cloud data of shape (N, 3+).
        gt_boxes (np.array): Ground truth boxes of shape (M, 7+).
        pred_boxes (list or np.array): Predicted boxes of shape (K, 7+).
    """
    import open3d as o3d
    geometries = []
    
    # Create the Point Cloud object.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    # Color the points grey.
    pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (points.shape[0], 1)))
    geometries.append(pcd)
    
    # Add Ground Truth boxes (Black).
    for box in gt_boxes:
        geometries.append(get_box_lineset(box, [0, 0, 0]))
        
    # Add Predicted boxes (Red).
    for box in pred_boxes:
        geometries.append(get_box_lineset(box, [1, 0, 0]))
        
    # Draw the boundary of the detection range (Blue).
    xr, yr, zr = CONFIG['x_range'], CONFIG['y_range'], CONFIG['z_range']
    l, w, h = xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]
    boundary = [(xr[0]+xr[1])/2, (yr[0]+yr[1])/2, (zr[0]+zr[1])/2, l, w, h, 0]
    geometries.append(get_box_lineset(boundary, [0, 0, 1]))
    
    # Add a coordinate frame at the origin (RGB axes).
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0,0,0]))
    
    # Launch the visualization window.
    o3d.visualization.draw_geometries(geometries, window_name="PillarNeXt Sparse")

# --- Dataset ---
# We attempt to import NuScenes libraries. If not found, we set a flag to warn the user later.
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
    _NUSC_OK = True
except:
    _NUSC_OK = False

class NuScenesDataset(Dataset):
    """
    A PyTorch Dataset wrapper for the NuScenes autonomous driving dataset.
    It handles loading point clouds, transforming coordinates, and parsing annotations.
    """
    def __init__(self, dataroot, version='v1.0-mini', augment=False):
        """
        Initializes the dataset.

        Args:
            dataroot (str): Path to the dataset root directory.
            version (str): NuScenes version string (e.g., 'v1.0-mini', 'v1.0-trainval').
            augment (bool): Whether to apply data augmentation (flip, rotate, scale).
        """
        if not _NUSC_OK:
            raise RuntimeError("pip install nuscenes-devkit")
            
        # Initialize the NuScenes API.
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        # We iterate over samples (snapshots in time).
        self.sample_tokens = [s['token'] for s in self.nusc.sample]
        self.augment = augment

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.sample_tokens)

    def _map_class(self, cat_name):
        """
        Maps NuScenes category names to integer class IDs.
        Currently configured to only detect 'vehicle' class.

        Args:
            cat_name (str): The raw category string from NuScenes.

        Returns:
            int or None: Class ID (0 for vehicle) or None if ignored.
        """
        if 'vehicle' in cat_name.lower():
            return 0
        return None 

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        
        This involves:
        1. Loading the LiDAR point cloud.
        2. Transforming annotations from Global Frame -> Ego Frame -> Sensor Frame.
        3. Filtering boxes outside the region of interest.
        4. Applying data augmentation if enabled.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Contains 'points' (np.array) and 'boxes' (np.array).
        """
        token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', token)
        
        # Get the LIDAR_TOP data record.
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pc_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        
        # Load point cloud from file.
        pc = LidarPointCloud.from_file(pc_path)
        # Points are (4, N) -> Transpose to (N, 4) [x, y, z, intensity].
        pts = pc.points[:4, :].T.astype(np.float32)
        # Normalize intensity to 0-1 range (usually 0-255 in raw data).
        pts[:, 3] /= 255.0 

        # --- Coordinate Transformation Logic ---
        # NuScenes stores boxes in Global coordinates.
        # LiDAR points are in Sensor coordinates.
        # We must transform Boxes: Global -> Ego -> Sensor.
        
        # 1. Get Calibration record (Sensor <-> Ego).
        cs_rec = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        # 2. Get Ego Pose record (Ego <-> Global).
        pose_rec = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # Prepare inverse transforms (to go Global -> Local).
        q_cs_inv = Quaternion(cs_rec['rotation']).inverse
        t_cs = np.array(cs_rec['translation'])
        q_pose_inv = Quaternion(pose_rec['rotation']).inverse
        t_pose = np.array(pose_rec['translation'])

        boxes = []
        for ann_t in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_t)
            
            # Filter by class.
            cls_id = self._map_class(ann['category_name'])
            if cls_id is None:
                continue
            
            # --- Transform Box Center ---
            box_glob = np.array(ann['translation'])
            # Global -> Ego
            box_ego = q_pose_inv.rotate(box_glob - t_pose)
            # Ego -> Sensor
            box_sens = q_cs_inv.rotate(box_ego - t_cs)
            
            # --- Transform Rotation ---
            # Combine rotations: (Sensor <- Ego) * (Ego <- Global) * (Global Box Rotation)
            q_box = Quaternion(ann['rotation'])
            yaw, _, _ = (q_cs_inv * q_pose_inv * q_box).yaw_pitch_roll
            
            # Get dimensions.
            l, w, h = ann['size'][1], ann['size'][0], ann['size'][2]
            
            # --- Range Filter ---
            # Discard boxes whose centers are outside the defined grid area.
            if (box_sens[0] < CONFIG['x_range'][0] or box_sens[0] > CONFIG['x_range'][1] or 
                box_sens[1] < CONFIG['y_range'][0] or box_sens[1] > CONFIG['y_range'][1]):
                continue 
                
            boxes.append([box_sens[0], box_sens[1], box_sens[2], l, w, h, yaw, int(cls_id)])
            
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 8), dtype=np.float32)

        # --- Data Augmentation ---
        if self.augment:
            # 1. Random Flip along Y-axis (Left/Right flip).
            if np.random.rand() > 0.5: 
                pts[:, 1] = -pts[:, 1]
                boxes[:, 1] = -boxes[:, 1]
                boxes[:, 6] = -boxes[:, 6] # Flip yaw
            
            # 2. Random Flip along X-axis (Front/Back flip).
            if np.random.rand() > 0.5: 
                pts[:, 0] = -pts[:, 0]
                boxes[:, 0] = -boxes[:, 0]
                boxes[:, 6] = np.pi - boxes[:, 6] # Adjust yaw
                
            # 3. Global Rotation (around Z-axis).
            rot = np.random.uniform(-0.78, 0.78) # +/- 45 degrees
            c, s = np.cos(rot), np.sin(rot)
            mat = np.array([[c, -s], [s, c]])
            
            # Rotate points (x, y).
            pts[:, :2] = np.dot(pts[:, :2], mat.T)
            # Rotate box centers and update yaw.
            if len(boxes) > 0:
                boxes[:, :2] = np.dot(boxes[:, :2], mat.T)
                boxes[:, 6] += rot
                
            # 4. Global Scaling.
            scale = np.random.uniform(0.95, 1.05)
            pts[:, :3] *= scale
            if len(boxes) > 0:
                boxes[:, :6] *= scale # Scale x, y, z, l, w, h

        return {'points': pts, 'boxes': boxes}

def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Since point clouds have variable point counts, we cannot stack them into a single tensor directly.
    We return a list of dictionaries, which the SparsePillarEncoder will handle.
    """
    return batch

# --- Sparse Architecture ---

class SparsePillarEncoder(nn.Module):
    """
    Encodes raw point clouds into a sparse tensor using pillar-based voxelization.
    This replaces the heavy fully-connected layers of PointPillars with a simplified sparse approach.
    """
    def __init__(self, out_c=32):
        super().__init__()
        self.grid_size = CONFIG['grid_size']
        self.x_range = CONFIG['x_range']
        self.y_range = CONFIG['y_range']
        
        # Calculate grid dimensions (number of pillars in X and Y).
        self.nx = int((self.x_range[1] - self.x_range[0]) / self.grid_size)
        self.ny = int((self.y_range[1] - self.y_range[0]) / self.grid_size)
        # Spatial shape needed for spconv tensor: [Y, X] (Z is 1 for pillars).
        self.spatial_shape = [self.ny, self.nx]
        
        # MLP to learn features from raw point coordinates.
        self.mlp = nn.Sequential(
            nn.Linear(9, out_c), # Inputs: x, y, z, i, x_offset, y_offset, z_offset, x_abs, y_abs
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, points_list):
        """
        Forward pass for the encoder.
        
        Args:
            points_list (list of np.array): List of point clouds for the batch.
            
        Returns:
            spconv.SparseConvTensor: The sparse tensor representation of the batch.
        """
        device = next(self.parameters()).device
        batch_indices = []
        batch_features = []
        
        for b_idx, pts in enumerate(points_list):
            # Filter points outside the range (safety check).
            keep = (pts[:,0] >= self.x_range[0]) & (pts[:,0] < self.x_range[1]) & \
                   (pts[:,1] >= self.y_range[0]) & (pts[:,1] < self.y_range[1])
            pts = pts[keep]
            if len(pts) == 0: continue
            
            pts_t = torch.from_numpy(pts).to(device)
            
            # --- Voxelization (Coordinate Calculation) ---
            # Calculate integer indices (coor_x, coor_y) for every point.
            coor_x = ((pts_t[:, 0] - self.x_range[0]) / self.grid_size).long()
            coor_y = ((pts_t[:, 1] - self.y_range[0]) / self.grid_size).long()
            
            # Calculate the physical center of the voxel/pillar.
            c_x = coor_x.float() * self.grid_size + self.x_range[0] + self.grid_size/2
            c_y = coor_y.float() * self.grid_size + self.y_range[0] + self.grid_size/2
            
            # --- Feature Decorrelation ---
            # Construct 9-dim feature vector:
            # [x, y, z, intensity, x-center, y-center, z, x, y]
            # (Note: Some redundant features are kept for compatibility with standard PointPillars).
            features = torch.cat([
                pts_t, 
                pts_t[:, :2] - torch.stack([c_x, c_y], dim=1), 
                pts_t[:, 2:3], 
                pts_t[:, :2]
            ], dim=1)
            
            # Apply MLP to lift point features.
            features = self.mlp(features)
            
            # --- Scatter / Max Pooling ---
            # We need to pool features from all points inside the same pillar into one feature.
            # 1. Create a unique hash key for every pillar (y * width + x).
            keys = coor_y * self.nx + coor_x
            
            # 2. Sort everything by keys so points in the same pillar are adjacent.
            order = torch.argsort(keys)
            keys = keys[order]
            features = features[order]
            coor_x = coor_x[order]
            coor_y = coor_y[order]
            
            # 3. Find boundaries between unique keys.
            unique_keys, inverse = torch.unique_consecutive(keys, return_inverse=True)
            
            # 4. Max Pool over the groups.
            # We split the sorted features based on how many points are in each unique key group.
            feat_chunks = torch.split(features, torch.bincount(inverse).tolist())
            # Take the max feature vector for each chunk (pillar).
            max_feats = torch.stack([chunk.max(dim=0)[0] for chunk in feat_chunks])
            
            # Get the coordinates corresponding to the unique keys.
            unique_mask = torch.cat([torch.tensor([1], device=device, dtype=torch.bool), keys[1:] != keys[:-1]])
            unique_x = coor_x[unique_mask]
            unique_y = coor_y[unique_mask]
            
            # Create sparse indices: [batch_id, y, x].
            indices = torch.stack([
                torch.full_like(unique_x, b_idx),
                unique_y,
                unique_x
            ], dim=1).int()
            
            batch_indices.append(indices)
            batch_features.append(max_feats)
            
        # Handle empty batch case.
        if not batch_indices:
            return spconv.SparseConvTensor(
                torch.zeros(1, 32).to(device), torch.zeros(1, 3).int().to(device), self.spatial_shape, len(points_list)
            )
            
        # Concatenate all batches into a single coordinate list and feature list.
        indices = torch.cat(batch_indices, dim=0)
        features = torch.cat(batch_features, dim=0)
        
        # Construct the spconv Tensor.
        return spconv.SparseConvTensor(features, indices, self.spatial_shape, len(points_list))

class SparseBasicBlock(spconv.SparseModule):
    """
    A ResNet-style BasicBlock implemented with Sparse Convolutions.
    Maintains sparsity (SubMConv) unless downsampling is required.
    """
    def __init__(self, in_c, out_c, stride=1, indice_key=None):
        super().__init__()
        # Conv1: Uses Submanifold Convolution (SubM) if stride is 1. 
        # SubM convolutions only compute outputs at active input sites, preserving sparsity.
        self.conv1 = spconv.SubMConv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = spconv.SubMConv2d(out_c, out_c, 3, stride=1, padding=1, bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
        self.downsample = None
        
        # Residual connection handling.
        if stride != 1 or in_c != out_c:
            self.downsample = spconv.SparseSequential(
                spconv.SubMConv2d(in_c, out_c, 1, stride=stride, bias=False, indice_key=indice_key),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        # replace_feature allows modifying the feature vector while keeping indices same.
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # Add residual connection.
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))
        return out

class SparsePillarNeXtBackbone(nn.Module):
    """
    The main backbone network. It processes the sparse pillars through multiple stages,
    downsampling the spatial resolution to extract high-level features.
    
    Structure:
    - Stage 1: Stride 1 (Original Resolution)
    - Stage 2: Stride 2 (Downsample)
    - Stage 3: Stride 4 (Downsample)
    - Stage 4: Stride 8 (Downsample)
    """
    def __init__(self):
        super().__init__()
        # Stage 1 (Stride 1)
        self.stage1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, indice_key='res1'),
            SparseBasicBlock(32, 32, indice_key='res1')
        )
        # Stage 2 (Stride 2)
        # Standard SparseConv2d is used for downsampling (increases receptive field).
        self.conv2 = spconv.SparseConv2d(32, 64, 3, 2, 1, bias=False) 
        self.stage2 = spconv.SparseSequential(
            nn.BatchNorm1d(64), nn.ReLU(),
            SparseBasicBlock(64, 64, indice_key='res2'),
            SparseBasicBlock(64, 64, indice_key='res2')
        )
        # Stage 3 (Stride 4)
        self.conv3 = spconv.SparseConv2d(64, 128, 3, 2, 1, bias=False)
        self.stage3 = spconv.SparseSequential(
            nn.BatchNorm1d(128), nn.ReLU(),
            SparseBasicBlock(128, 128, indice_key='res3'),
            SparseBasicBlock(128, 128, indice_key='res3')
        )
        # Stage 4 (Stride 8)
        self.conv4 = spconv.SparseConv2d(128, 256, 3, 2, 1, bias=False)
        self.stage4 = spconv.SparseSequential(
            nn.BatchNorm1d(256), nn.ReLU(),
            SparseBasicBlock(256, 256, indice_key='res4'),
            SparseBasicBlock(256, 256, indice_key='res4')
        )
        self.out_c = 256

    def forward(self, x):
        x = self.stage1(x)
        x = self.conv2(x)
        x = self.stage2(x)
        x = self.conv3(x)
        x = self.stage3(x)
        x = self.conv4(x)
        x = self.stage4(x)
        # Convert Sparse Tensor back to Dense Tensor (N, C, H, W) for the detection head.
        return x.dense()

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    Captures multi-scale context by using convolutions with different dilation rates.
    This helps the model 'see' context at different distances without losing resolution.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # 1x1 Conv
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, in_c, 1, bias=False), nn.BatchNorm2d(in_c), nn.ReLU())
        # 3x3 Conv, Dilation 6
        self.conv2 = nn.Sequential(nn.Conv2d(in_c, in_c, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(in_c), nn.ReLU())
        # 3x3 Conv, Dilation 12
        self.conv3 = nn.Sequential(nn.Conv2d(in_c, in_c, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(in_c), nn.ReLU())
        # Projection layer to combine features.
        self.project = nn.Sequential(nn.Conv2d(in_c * 3, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU())

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return self.project(torch.cat([x1, x2, x3], dim=1))

class UpsamplingHead(nn.Module):
    """
    Upsamples the backbone features and predicts detection targets.
    """
    def __init__(self, in_c, n_classes=1):
        super().__init__()
        # Upsample: Input stride 8 -> Output stride 4.
        # This recovers some spatial resolution lost during downsampling.
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        # Classification Head (Heatmap).
        self.conv_cls = nn.Conv2d(128, n_classes, 1)
        # Regression Head (Box parameters).
        self.conv_reg = nn.Conv2d(128, 8, 1)
        
        # Initialize bias for classification to -4.6 (approx probability 0.01).
        # This prevents a high loss at the start of training due to background dominance.
        self.conv_cls.bias.data.fill_(-4.6)

    def forward(self, x):
        x = self.up(x)
        # Sigmoid for heatmap (0 to 1 probability).
        # Regression output is raw (no activation).
        return torch.sigmoid(self.conv_cls(x)), self.conv_reg(x)

class SparsePillarNeXt(nn.Module):
    """
    The full model wrapper combining Encoder, Backbone, Neck, and Head.
    """
    def __init__(self):
        super().__init__()
        self.encoder = SparsePillarEncoder()
        self.backbone = SparsePillarNeXtBackbone() # Output Stride: 8
        self.neck = ASPP(self.backbone.out_c, 256)
        self.head = UpsamplingHead(256) # Output Stride: 4
        self.stride = 4

    def forward(self, x):
        x = self.encoder(x)
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(x)

# --- Loss & Targets ---

def gaussian_2d(shape, sigma=1):
    """
    Generates a 2D Gaussian kernel.
    Used for creating ground truth heatmaps where the center of an object is 1.0, 
    decaying to 0.0 outwards.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draws a 2D Gaussian on the target heatmap at the specified center.
    If multiple Gaussians overlap, we take the element-wise maximum.
    """
    radius = int(radius)
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape
    
    # Calculate bounds to handle edges of the image.
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(gaussian[radius - top:radius + bottom, radius - left:radius + right]).to(heatmap.device)
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def build_targets(gt_boxes, feature_shape, device):
    """
    Constructs the ground truth targets for the network.
    
    Args:
        gt_boxes (list of list): Batch of ground truth boxes.
        feature_shape (tuple): Shape of the network output [B, C, H, W].
        device (torch.device): Target device.
        
    Returns:
        hm: Heatmap target [B, 1, H, W].
        reg: Regression target [B, 8, H, W].
        mask: Boolean mask indicating object locations [B, 1, H, W].
    """
    B, _, H, W = feature_shape
    hm = torch.zeros(B, 1, H, W, device=device)
    reg = torch.zeros(B, 8, H, W, device=device)
    mask = torch.zeros(B, 1, H, W, device=device)
    
    # Output stride is 4, so the output grid is 4x larger than the input grid.
    stride = 4
    grid_size = CONFIG['grid_size'] * stride
    
    for b in range(B):
        for box in gt_boxes[b]:
            x, y, z, l, w, h, yaw, cls = box
            
            # Map physical coordinates to feature map indices.
            cx, cy = (x - CONFIG['x_range'][0]) / grid_size, (y - CONFIG['y_range'][0]) / grid_size
            ix, iy = int(cx), int(cy)
            
            # Ensure index is within bounds.
            if 0 <= ix < W and 0 <= iy < H:
                # Radius is determined by object size to adapt Gaussian spread.
                radius = max(2, int(max(l, w) / grid_size / 2))
                draw_umich_gaussian(hm[b, 0], (ix, iy), radius)
                
                # Set mask to 1 at object center.
                mask[b, 0, iy, ix] = 1.0
                
                # --- Regression Targets ---
                # 0-1: Sub-pixel offset (to correct quantization error).
                reg[b, 0, iy, ix] = float(cx - ix)
                reg[b, 1, iy, ix] = float(cy - iy)
                # 2: Absolute Z height.
                reg[b, 2, iy, ix] = float(z)
                # 3-5: Log dimensions (log(l), log(w), log(h)) for numerical stability.
                reg[b, 3, iy, ix] = float(math.log(max(l, 0.01)))
                reg[b, 4, iy, ix] = float(math.log(max(w, 0.01)))
                reg[b, 5, iy, ix] = float(math.log(max(h, 0.01)))
                # 6-7: Sine and Cosine of yaw (continuous representation of angle).
                reg[b, 6, iy, ix] = float(math.sin(yaw))
                reg[b, 7, iy, ix] = float(math.cos(yaw))
                
    return hm, reg, mask

def compute_loss(pred_cls, pred_reg, gt_cls, gt_reg, gt_mask):
    """
    Computes the total loss.
    
    1. Classification: Penalty-reduced Focal Loss (from CenterNet).
       Handles extreme class imbalance (vast majority of grid is background).
    2. Regression: L1 Loss, calculated only at ground truth center locations.
    """
    # Identify positive samples.
    pos_inds = gt_cls.eq(1).float()
    # Negative samples are weighted by their closeness to the ground truth (soft labels).
    neg_weights = torch.pow(1 - gt_cls, 4)
    
    pred_cls = torch.clamp(pred_cls, 1e-6, 1 - 1e-6)
    
    # Focal Loss Calculation.
    pos_loss = torch.log(pred_cls) * torch.pow(1 - pred_cls, 2) * pos_inds
    neg_loss = torch.log(1 - pred_cls) * torch.pow(pred_cls, 2) * neg_weights * gt_cls.lt(1).float()
    
    loss_cls = - (pos_loss.sum() + neg_loss.sum()) / max(1, pos_inds.sum())
    
    # Regression Loss (only at positive locations).
    mask = gt_mask.expand_as(pred_reg).bool()
    loss_reg = F.l1_loss(pred_reg[mask], gt_reg[mask], reduction='sum') / max(1, pos_inds.sum())
    
    # Total loss is weighted combination.
    return loss_cls + 2.0 * loss_reg

# --- Main Loop ---

def train(model, loader, opt, scaler, epoch):
    """
    Executes one training epoch.
    """
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(loader):
        # Use Automatic Mixed Precision (AMP) for speed and memory efficiency.
        with autocast('cuda'):
            pred_cls, pred_reg = model([b['points'] for b in batch])
            gt_cls, gt_reg, gt_mask = build_targets([b['boxes'] for b in batch], pred_cls.shape, 'cuda')
            loss = compute_loss(pred_cls, pred_reg, gt_cls, gt_reg, gt_mask)
        
        # Check for NaN loss (instability).
        if torch.isnan(loss):
            opt.zero_grad(); continue

        # Standard PyTorch Backprop with Scaler (for AMP).
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        # Gradient Clipping to prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt)
        scaler.update()
        
        epoch_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Ep {epoch} | Loss: {loss.item():.4f} | Max Conf: {pred_cls.max().item():.4f} | Pos: {gt_mask.sum()}")
            
    return epoch_loss / len(loader)

@torch.no_grad()
def decode_predictions(pred_cls, pred_reg):
    """
    Decodes network output maps into bounding box lists.
    
    1. Finds local maxima (peaks) in the heatmap using MaxPool.
    2. Extracts regression parameters at those peak locations.
    3. Converts regression parameters back to physical units (meters, radians).
    """
    B, _, H, W = pred_cls.shape
    stride = 4
    grid_size = CONFIG['grid_size'] * stride
    
    # Non-Maximum Suppression (NMS) via Max Pooling.
    nms_kernel = 3
    hmax = F.max_pool2d(pred_cls, (nms_kernel, nms_kernel), stride=1, padding=(nms_kernel-1)//2)
    # Keep only points that are equal to the local max.
    keep = (hmax == pred_cls).float()
    pred_cls = pred_cls * keep
    
    batch_boxes = []
    for b in range(B):
        scores = pred_cls[b, 0].view(-1)
        # Take top 50 predictions.
        topk_scores, topk_inds = torch.topk(scores, 50)
        
        boxes = []
        for i in range(50):
            if topk_scores[i] < 0.1: continue # Score threshold.
            
            idx = topk_inds[i]
            iy, ix = (idx // W).long(), (idx % W).long()
            
            # Retrieve regression values.
            reg = pred_reg[b, :, iy, ix]
            
            # Restore coordinates: (Index + offset) * grid_size + range_start.
            cx = (ix + reg[0]) * grid_size + CONFIG['x_range'][0]
            cy = (iy + reg[1]) * grid_size + CONFIG['y_range'][0]
            cz = reg[2]
            
            # Restore dimensions: exp(log_dim).
            l, w, h = torch.exp(reg[3]), torch.exp(reg[4]), torch.exp(reg[5])
            
            # Restore Yaw: atan2(sin, cos).
            yaw = torch.atan2(reg[6], reg[7])
            
            boxes.append([cx.item(), cy.item(), cz.item(), l.item(), w.item(), h.item(), yaw.item()])
        batch_boxes.append(boxes)
    return batch_boxes

@torch.no_grad()
def evaluate(model, loader):
    """
    Runs evaluation on the validation set.
    Computes standard detection metrics: Recall, Precision, and F1 Score.
    Uses a Euclidean distance threshold (2.0m) for matching predictions to ground truth.
    """
    model.eval()
    matches, total_gt, total_pred = 0, 0, 0
    for batch in loader:
        pred_cls, pred_reg = model([b['points'] for b in batch])
        batch_boxes = decode_predictions(pred_cls, pred_reg)
        
        for b in range(len(batch_boxes)):
            preds_box = batch_boxes[b]
            gt_boxes = batch[b]['boxes']
            
            total_gt += len(gt_boxes)
            total_pred += len(preds_box)
            
            if len(gt_boxes) > 0 and len(preds_box) > 0:
                gt_xy = gt_boxes[:, :2]
                pred_xy = np.array([p[:2] for p in preds_box])
                
                # Calculate distance matrix between all GT and Pred pairs.
                dists = np.linalg.norm(gt_xy[:, None] - pred_xy[None, :], axis=2)
                
                # Count matches where distance < 2.0 meters.
                matches += np.any(dists < 2.0, axis=1).sum()
                
    rec = matches / max(1, total_gt)
    prec = matches / max(1, total_pred)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    return rec, prec, f1

def main():
    """
    Main entry point. Handles argument parsing, setup, and mode selection (Train vs Infer).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='Operation mode: train or infer')
    parser.add_argument('--dataset', default='nuscenes', help='Dataset type (only nuscenes supported)')
    parser.add_argument('--data_root', default='./v1.0-mini', help='Path to dataset root')
    parser.add_argument('--nusc_version', default='v1.0-mini', help='NuScenes version string')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--checkpoint', default=None, help='Path to .pth checkpoint to load')
    parser.add_argument('--evaluate', action='store_true', help='Enable numeric evaluation during inference')
    parser.add_argument('--visualize', action='store_true', help='Enable Open3D visualization during inference')
    args = parser.parse_args()
    
    seed_everything()
    device = torch.device('cuda')
    mkdir('./checkpoints')
    
    # Initialize Datasets.
    train_ds = NuScenesDataset(args.data_root, args.nusc_version, augment=True)
    val_ds = NuScenesDataset(args.data_root, args.nusc_version, augment=False)
    
    # Initialize Loaders.
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Build Model.
    model = SparsePillarNeXt().to(device)
    
    # --- Print Model Structure ---
    if args.mode == 'train':
        print("\n" + "="*50)
        print(f"Model Architecture: {model.__class__.__name__}")
        print("="*50)
        print(model)
        print("="*50 + "\n")
    # --------------------------------------------

    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scaler = GradScaler('cuda')
    
    start_epoch = 1
    
    # Load Checkpoint if provided.
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded {args.checkpoint}")
    
    # --- Training Loop ---
    if args.mode == 'train':
        best_f1 = 0
        for ep in range(start_epoch, args.epochs + 1):
            loss = train(model, train_loader, opt, scaler, ep)
            
            # Save latest checkpoint.
            torch.save({'model': model.state_dict()}, './checkpoints/ckpt_last.pth')
            
            # Evaluate every 5 epochs (or late in training).
            if ep % 5 == 0 or ep > 25:
                rec, prec, f1 = evaluate(model, val_loader)
                print(f"Epoch {ep} >> R: {rec*100:.2f} | P: {prec*100:.2f} | F1: {f1*100:.2f}")
                
                # Save best model.
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save({'model': model.state_dict()}, './checkpoints/best.pth')
                    print("Saved Best!")
                    
    # --- Inference Loop ---
    elif args.mode == 'infer':
        if args.visualize:
            # Batch size 1 for visualization to handle one frame at a time.
            viz_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
            print("Starting Visualization...")
            for batch in viz_loader:
                pred_cls, pred_reg = model([b['points'] for b in batch])
                batch_boxes = decode_predictions(pred_cls, pred_reg)
                visualize_sample(batch[0]['points'], batch[0]['boxes'], batch_boxes[0])
        else:
            rec, prec, f1 = evaluate(model, val_loader)
            print(f"Result >> R: {rec*100:.2f} | P: {prec*100:.2f} | F1: {f1*100:.2f}")

if __name__ == '__main__':
    main()