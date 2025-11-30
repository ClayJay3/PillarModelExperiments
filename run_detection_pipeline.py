#!/usr/bin/env python3
import os
import sys
import argparse
import random
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# --- Dependency Checks ---
# We rely on spconv for Sparse Convolutions. This is much more memory efficient
# than dense 3D convolutions for LiDAR data.
try:
    import spconv.pytorch as spconv
except ImportError:
    print("Error: Spconv not found. Please install: pip install spconv-cu118 (or matching CUDA version)")
    sys.exit(1)

# --- Visualization Helper ---

def get_box_lineset(box, color):
    """
    Creates an Open3D LineSet for a 3D bounding box.
    
    Args:
        box (np.array): Box parameters [x, y, z, l, w, h, yaw].
        color (list): RGB color [r, g, b].
    
    Returns:
        o3d.geometry.LineSet: The wireframe box.
    """
    import open3d as o3d
    x, y, z, l, w, h, yaw = box[:7]
    
    # Define corners relative to center.
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    
    # Rotation matrix around Z-axis.
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    # Rotate and translate.
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d += np.array([[x], [y], [z]])
    
    # Define lines connecting corners.
    lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_3d.T)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set

def visualize_sample(points, gt_boxes, pred_boxes):
    """
    Visualizes the first sample in the batch using Open3D.
    Note: This pauses execution until the window is closed.
    """
    import open3d as o3d
    
    # Filter points to only show the first element of the batch (batch_idx == 0).
    # 'points' tensor shape is (N, 5), where column 4 is the batch index.
    mask = (points[:, 4] == 0) 
    pts_vis = points[mask, :3].cpu().numpy()
    
    geometries = []
    
    # Create Point Cloud.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_vis)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (len(pts_vis), 1)))
    geometries.append(pcd)
    
    # Add Ground Truth boxes (Black).
    # Note: gt_boxes is a list of tensors, so we take index 0.
    for box in gt_boxes[0]: 
        geometries.append(get_box_lineset(box.numpy(), [0, 0, 0]))
        
    # Add Predicted boxes (Red).
    for box in pred_boxes[0]: 
        geometries.append(get_box_lineset(box, [1, 0, 0]))
    
    # Add Coordinate Frame (Origin).
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0,0,0]))
    
    o3d.visualization.draw_geometries(geometries, window_name="PillarNeXt Optimized")

# --- Auto-Optimization Engine ---

def auto_tune_environment(base_config):
    """
    Detects hardware capabilities and optimizes configuration parameters.
    Adjusts Batch Size, TF32 precision, and CPU Worker count.
    
    Args:
        base_config (dict): The default configuration dictionary.
        
    Returns:
        dict: The optimized configuration dictionary.
    """
    print(f"\n{'='*20} AUTO-TUNING ENVIRONMENT {'='*20}")
    
    # 1. GPU VRAM & Batch Size Detection
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        total_mem_gb = gpu_props.total_memory / (1024**3)
        print(f"Detected GPU: {gpu_props.name} | VRAM: {total_mem_gb:.2f} GB")
        
        # Heuristic calculation for Batch Size based on 0.075m grid resolution.
        # ~0.8GB VRAM per sample + ~1.5GB Model Overhead.
        if total_mem_gb > 22:   # e.g., RTX 3090/4090
            base_config['batch_size'] = 24
        elif total_mem_gb > 14: # e.g., RTX 4080 (16GB)
            base_config['batch_size'] = 16 
        elif total_mem_gb > 10: # e.g., RTX 3080/4070 (12GB)
            base_config['batch_size'] = 8
        else:                   # e.g., RTX 3070 (8GB)
            base_config['batch_size'] = 4
            
        print(f"-> Optimized Batch Size: {base_config['batch_size']}")
        
        # 2. TF32 Support (TensorFloat-32)
        # Available on Ampere (RTX 30xx) and newer. Drastically speeds up FP32 matmul.
        if gpu_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("-> TF32 Enabled: YES (High Performance Matrix Math)")
        else:
            print("-> TF32 Enabled: NO (Hardware not supported)")
            
    else:
        print("WARNING: No GPU detected. Training will be impossibly slow.")
        base_config['batch_size'] = 2

    # 3. CPU Worker Detection
    cpu_count = os.cpu_count()
    # Reserve 2 cores for system/OS overhead, use the rest for data loading.
    workers = max(2, cpu_count - 2)
    base_config['num_workers'] = workers
    print(f"-> DataLoader Workers: {workers}")
    print("="*66 + "\n")
    return base_config

# --- Configuration ---
CONFIG = {
    'x_range': (-51.2, 51.2),
    'y_range': (-51.2, 51.2),
    'z_range': (-5.0, 3.0),
    'grid_size': 0.075,
    'lr': 0.001,
    'weight_decay': 0.01,
    # The following will be overwritten by auto_tune_environment
    'batch_size': 4, 
    'num_workers': 4,
}

# --- Utilities ---

def seed_everything(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mkdir(path): 
    """Creates a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

# --- Dataset (NuScenes) ---

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
    _NUSC_OK = True
except: 
    _NUSC_OK = False

class NuScenesDataset(Dataset):
    """
    Handles loading and parsing of NuScenes LiDAR data.
    """
    def __init__(self, dataroot, version='v1.0-mini', augment=False):
        if not _NUSC_OK: raise RuntimeError("pip install nuscenes-devkit")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.sample_tokens = [s['token'] for s in self.nusc.sample]
        self.augment = augment

    def __len__(self): 
        return len(self.sample_tokens)

    def _map_class(self, cat_name):
        """Filters for 'vehicle' class only."""
        if 'vehicle' in cat_name.lower(): return 0
        return None 

    def __getitem__(self, idx):
        """
        Loads a point cloud and associated bounding boxes.
        Transforms boxes from Global -> Ego -> Sensor coordinates.
        """
        token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', token)
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pc_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        
        # Load Points
        pc = LidarPointCloud.from_file(pc_path)
        pts = pc.points[:4, :].T.astype(np.float32)
        pts[:, 3] /= 255.0 # Normalize intensity

        # Get Calibration Data
        cs_rec = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose_rec = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # Prepare Inverse Transforms
        q_cs_inv = Quaternion(cs_rec['rotation']).inverse
        t_cs = np.array(cs_rec['translation'])
        q_pose_inv = Quaternion(pose_rec['rotation']).inverse
        t_pose = np.array(pose_rec['translation'])

        # Transform Boxes
        boxes = []
        for ann_t in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_t)
            cls_id = self._map_class(ann['category_name'])
            if cls_id is None: continue
            
            box_glob = np.array(ann['translation'])
            # Global -> Ego
            box_ego = q_pose_inv.rotate(box_glob - t_pose)
            # Ego -> Sensor
            box_sens = q_cs_inv.rotate(box_ego - t_cs)
            
            # Rotation
            q_box = Quaternion(ann['rotation'])
            yaw, _, _ = (q_cs_inv * q_pose_inv * q_box).yaw_pitch_roll
            l, w, h = ann['size'][1], ann['size'][0], ann['size'][2]
            
            # Spatial Filter
            if (box_sens[0] < CONFIG['x_range'][0] or box_sens[0] > CONFIG['x_range'][1] or 
                box_sens[1] < CONFIG['y_range'][0] or box_sens[1] > CONFIG['y_range'][1]):
                continue 
            boxes.append([box_sens[0], box_sens[1], box_sens[2], l, w, h, yaw, int(cls_id)])
            
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 8), dtype=np.float32)

        # Data Augmentation (Flip, Rotate, Scale)
        if self.augment:
            if np.random.rand() > 0.5: 
                pts[:, 1] = -pts[:, 1]; boxes[:, 1] = -boxes[:, 1]; boxes[:, 6] = -boxes[:, 6]
            if np.random.rand() > 0.5: 
                pts[:, 0] = -pts[:, 0]; boxes[:, 0] = -boxes[:, 0]; boxes[:, 6] = np.pi - boxes[:, 6]
            rot = np.random.uniform(-0.78, 0.78)
            c, s = np.cos(rot), np.sin(rot)
            mat = np.array([[c, -s], [s, c]])
            pts[:, :2] = np.dot(pts[:, :2], mat.T)
            if len(boxes) > 0:
                boxes[:, :2] = np.dot(boxes[:, :2], mat.T)
                boxes[:, 6] += rot
            scale = np.random.uniform(0.95, 1.05)
            pts[:, :3] *= scale
            if len(boxes) > 0: boxes[:, :6] *= scale

        # Return standard dictionary (to be collated later)
        return {'points': pts, 'boxes': boxes}

# --- OPTIMIZED COLLATION ---

def fast_collate_fn(batch):
    """
    Vectorized Collation Function.
    
    Instead of returning a list of point clouds (which forces the GPU to loop),
    this function stacks ALL points from the batch into a single (Total_Points, 5) tensor.
    The 5th column is the 'batch_idx', which allows the model to distinguish samples.
    
    Args:
        batch (list): List of dicts from __getitem__.
    
    Returns:
        dict: Batched data with 'points' as a single massive tensor.
    """
    batched_pts_list = []
    batched_boxes_list = []
    
    for i, sample in enumerate(batch):
        pts = sample['points']
        # Create a column vector filled with the index 'i'
        batch_idx = np.full((len(pts), 1), i, dtype=np.float32)
        # Horizontal stack: [x, y, z, i] + [batch_idx] -> [x, y, z, i, batch_idx]
        pts_with_idx = np.hstack([pts, batch_idx])
        
        batched_pts_list.append(torch.from_numpy(pts_with_idx))
        batched_boxes_list.append(torch.from_numpy(sample['boxes']))
        
    return {
        # Concatenate all lists into one tensor. This operation happens on CPU
        # but prepares the data for extremely fast GPU consumption.
        'points': torch.cat(batched_pts_list, dim=0), 
        'boxes': batched_boxes_list, # Boxes remain a list (variable number of boxes per sample)
        'batch_size': len(batch)
    }

# --- Vectorized Encoder ---

class SparsePillarEncoder(nn.Module):
    """
    Converts raw point clouds into a sparse tensor using vector operations.
    Eliminates all Python loops from the forward pass.
    """
    def __init__(self, out_c=32):
        super().__init__()
        self.grid_size = CONFIG['grid_size']
        self.x_range = CONFIG['x_range']
        self.y_range = CONFIG['y_range']
        self.nx = int((self.x_range[1] - self.x_range[0]) / self.grid_size)
        self.ny = int((self.y_range[1] - self.y_range[0]) / self.grid_size)
        self.spatial_shape = [self.ny, self.nx]
        
        # Point Feature Encoder (PointNet-style MLP)
        self.mlp = nn.Sequential(
            nn.Linear(9, out_c),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, batch_dict):
        """
        Vectorized Forward Pass.
        
        Args:
            batch_dict (dict): Contains 'points' (N, 5) tensor.
            
        Returns:
            spconv.SparseConvTensor: Ready for the backbone.
        """
        # Load the giant tensor to GPU. 'non_blocking' overlaps transfer with computation.
        pts_t = batch_dict['points'].cuda(non_blocking=True)
        batch_size = batch_dict['batch_size']
        
        # 1. Vectorized Filter: Keep points within grid range.
        keep = (pts_t[:,0] >= self.x_range[0]) & (pts_t[:,0] < self.x_range[1]) & \
               (pts_t[:,1] >= self.y_range[0]) & (pts_t[:,1] < self.y_range[1])
        pts_t = pts_t[keep]
        
        # Handle empty batch case (rare but possible).
        if len(pts_t) == 0:
             return spconv.SparseConvTensor(
                torch.zeros(1, 32).cuda(), torch.zeros(1, 3).int().cuda(), 
                self.spatial_shape, batch_size)

        # 2. Voxel Coordinate Calculation.
        coor_x = ((pts_t[:, 0] - self.x_range[0]) / self.grid_size).long()
        coor_y = ((pts_t[:, 1] - self.y_range[0]) / self.grid_size).long()
        batch_idx = pts_t[:, 4].long() # Retrieve batch index column

        # Physical center of the calculated voxel.
        c_x = coor_x.float() * self.grid_size + self.x_range[0] + self.grid_size/2
        c_y = coor_y.float() * self.grid_size + self.y_range[0] + self.grid_size/2

        # 3. Feature Decoration.
        # [x, y, z, intensity, x_offset, y_offset, z, x, y]
        features = torch.cat([
            pts_t[:, :4], 
            pts_t[:, :2] - torch.stack([c_x, c_y], dim=1), 
            pts_t[:, 2:3], 
            pts_t[:, :2]
        ], dim=1)
        
        features = self.mlp(features)
        
        # 4. Sparse Pillar Aggregation (The "Scatter" Step).
        # We need to group points that fall into the same pillar across the whole batch.
        # We create a unique hash key: batch_idx * (Area) + y * Width + x
        keys = batch_idx * (self.ny * self.nx) + coor_y * self.nx + coor_x
        
        # Sort points by this key so points in the same pillar are adjacent in memory.
        order = torch.argsort(keys)
        keys = keys[order]
        features = features[order]
        
        # Identify unique pillars and map original points to these pillars.
        unique_keys, inverse = torch.unique_consecutive(keys, return_inverse=True)
        
        # Max Pooling via Split:
        # Split the features based on how many points share the same key, then take max.
        feat_chunks = torch.split(features, torch.bincount(inverse).tolist())
        max_feats = torch.stack([chunk.max(dim=0)[0] for chunk in feat_chunks])
        
        # Recover the 3D indices (batch, y, x) for the unique pillars.
        # We take the index of the first point in every unique group.
        unique_mask = torch.cat([torch.tensor([1], device=pts_t.device, dtype=torch.bool), keys[1:] != keys[:-1]])
        
        indices = torch.stack([
            batch_idx[order][unique_mask], # Batch ID
            coor_y[order][unique_mask],    # Y
            coor_x[order][unique_mask]     # X
        ], dim=1).int()

        return spconv.SparseConvTensor(max_feats, indices, self.spatial_shape, batch_size)

# --- Standard Backbone & Head ---

class SparseBasicBlock(spconv.SparseModule):
    """Standard ResNet BasicBlock adapted for Sparse Convolutions."""
    def __init__(self, in_c, out_c, stride=1, indice_key=None):
        super().__init__()
        # SubMConv2d keeps sparsity (doesn't dilate features into empty space).
        self.conv1 = spconv.SubMConv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = spconv.SubMConv2d(out_c, out_c, 3, stride=1, padding=1, bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = spconv.SparseSequential(
                spconv.SubMConv2d(in_c, out_c, 1, stride=stride, bias=False, indice_key=indice_key),
                nn.BatchNorm1d(out_c)
            )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))
        return out

class SparsePillarNeXtBackbone(nn.Module):
    """
    Extracts features at multiple scales (Stride 1, 2, 4, 8).
    """
    def __init__(self):
        super().__init__()
        self.stage1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, indice_key='res1'),
            SparseBasicBlock(32, 32, indice_key='res1')
        )
        # SparseConv2d (not SubM) is used to downsample and increase receptive field.
        self.conv2 = spconv.SparseConv2d(32, 64, 3, 2, 1, bias=False)
        self.stage2 = spconv.SparseSequential(
            nn.BatchNorm1d(64), nn.ReLU(),
            SparseBasicBlock(64, 64, indice_key='res2'),
            SparseBasicBlock(64, 64, indice_key='res2')
        )
        self.conv3 = spconv.SparseConv2d(64, 128, 3, 2, 1, bias=False)
        self.stage3 = spconv.SparseSequential(
            nn.BatchNorm1d(128), nn.ReLU(),
            SparseBasicBlock(128, 128, indice_key='res3'),
            SparseBasicBlock(128, 128, indice_key='res3')
        )
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
        return x.dense() # Convert to dense tensor for the head

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, in_c, 1, bias=False), nn.BatchNorm2d(in_c), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_c, in_c, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(in_c), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_c, in_c, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(in_c), nn.ReLU())
        self.project = nn.Sequential(nn.Conv2d(in_c * 3, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU())
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return self.project(torch.cat([x1, x2, x3], dim=1))

class UpsamplingHead(nn.Module):
    """Upsamples features and predicts heatmaps/regressions."""
    def __init__(self, in_c, n_classes=1):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.conv_cls = nn.Conv2d(128, n_classes, 1)
        self.conv_reg = nn.Conv2d(128, 8, 1)
        self.conv_cls.bias.data.fill_(-4.6) # Initialize bias to prevent instability
    def forward(self, x):
        x = self.up(x)
        return torch.sigmoid(self.conv_cls(x)), self.conv_reg(x)

class SparsePillarNeXt(nn.Module):
    """Full Model Wrapper."""
    def __init__(self):
        super().__init__()
        self.encoder = SparsePillarEncoder()
        self.backbone = SparsePillarNeXtBackbone()
        self.neck = ASPP(self.backbone.out_c, 256)
        self.head = UpsamplingHead(256)
    def forward(self, batch_dict):
        x = self.encoder(batch_dict)
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(x)

# --- Targets & Loss ---

def gaussian_2d(shape, sigma=1):
    """Generates 2D Gaussian kernel for heatmap generation."""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    """Projects Gaussian kernel onto the heatmap at the object center."""
    radius = int(radius)
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(gaussian[radius - top:radius + bottom, radius - left:radius + right]).to(heatmap.device)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def build_targets(gt_boxes, feature_shape, device):
    """Constructs ground truth heatmaps and regression targets."""
    B, _, H, W = feature_shape
    hm = torch.zeros(B, 1, H, W, device=device)
    reg = torch.zeros(B, 8, H, W, device=device)
    mask = torch.zeros(B, 1, H, W, device=device)
    stride = 4
    grid_size = CONFIG['grid_size'] * stride
    
    for b in range(B):
        for box in gt_boxes[b]:
            x, y, z, l, w, h, yaw, cls = box
            cx, cy = (x - CONFIG['x_range'][0]) / grid_size, (y - CONFIG['y_range'][0]) / grid_size
            ix, iy = int(cx), int(cy)
            if 0 <= ix < W and 0 <= iy < H:
                radius = max(2, int(max(l, w) / grid_size / 2))
                draw_umich_gaussian(hm[b, 0], (ix, iy), radius)
                mask[b, 0, iy, ix] = 1.0
                reg[b, 0, iy, ix] = float(cx - ix)
                reg[b, 1, iy, ix] = float(cy - iy)
                reg[b, 2, iy, ix] = float(z)
                # Use Log for dimensions to handle scale variance
                reg[b, 3, iy, ix] = float(math.log(max(l, 0.01)))
                reg[b, 4, iy, ix] = float(math.log(max(w, 0.01)))
                reg[b, 5, iy, ix] = float(math.log(max(h, 0.01)))
                reg[b, 6, iy, ix] = float(math.sin(yaw))
                reg[b, 7, iy, ix] = float(math.cos(yaw))
    return hm, reg, mask

def compute_loss(pred_cls, pred_reg, gt_cls, gt_reg, gt_mask):
    """Focal Loss for Classification + L1 Loss for Regression."""
    pos_inds = gt_cls.eq(1).float()
    neg_weights = torch.pow(1 - gt_cls, 4)
    pred_cls = torch.clamp(pred_cls, 1e-6, 1 - 1e-6)
    
    # Focal Loss (CenterNet variant)
    pos_loss = torch.log(pred_cls) * torch.pow(1 - pred_cls, 2) * pos_inds
    neg_loss = torch.log(1 - pred_cls) * torch.pow(pred_cls, 2) * neg_weights * gt_cls.lt(1).float()
    loss_cls = - (pos_loss.sum() + neg_loss.sum()) / max(1, pos_inds.sum())
    
    # Regression Loss
    mask = gt_mask.expand_as(pred_reg).bool()
    loss_reg = F.l1_loss(pred_reg[mask], gt_reg[mask], reduction='sum') / max(1, pos_inds.sum())
    
    return loss_cls + 2.0 * loss_reg

# --- Decoding & Eval ---

@torch.no_grad()
def decode_predictions(pred_cls, pred_reg):
    """
    Decodes heatmaps into 3D Bounding Boxes.
    Returns: List of boxes, where each box is [x, y, z, l, w, h, yaw, score]
    """
    B, _, H, W = pred_cls.shape
    stride = 4
    grid_size = CONFIG['grid_size'] * stride
    
    # MaxPool NMS to find peaks
    nms_kernel = 3
    hmax = F.max_pool2d(pred_cls, (nms_kernel, nms_kernel), stride=1, padding=(nms_kernel-1)//2)
    keep = (hmax == pred_cls).float()
    pred_cls = pred_cls * keep
    
    batch_boxes = []
    for b in range(B):
        scores = pred_cls[b, 0].view(-1)
        topk_scores, topk_inds = torch.topk(scores, 50)
        boxes = []
        for i in range(50):
            if topk_scores[i] < 0.1: continue
            idx = topk_inds[i]
            iy, ix = (idx // W).long(), (idx % W).long()
            reg = pred_reg[b, :, iy, ix]
            
            # Restore coordinates
            cx = (ix + reg[0]) * grid_size + CONFIG['x_range'][0]
            cy = (iy + reg[1]) * grid_size + CONFIG['y_range'][0]
            cz = reg[2]
            l, w, h = torch.exp(reg[3]), torch.exp(reg[4]), torch.exp(reg[5])
            yaw = torch.atan2(reg[6], reg[7])
            
            # Append box WITH score
            boxes.append([cx.item(), cy.item(), cz.item(), l.item(), w.item(), h.item(), yaw.item(), topk_scores[i].item()])
        batch_boxes.append(boxes)
    return batch_boxes

@torch.no_grad()
def evaluate(model, loader):
    """
    Calculates mAP (AP@2.0m), Precision, Recall, and F1.
    """
    model.eval()
    
    # Storage for all predictions across the dataset
    # List of dicts: {'score': float, 'is_tp': 0 or 1}
    all_predictions = [] 
    total_gt_objects = 0
    
    # Distance threshold for a True Positive (NuScenes standard-ish)
    MATCH_DIST_THRESHOLD = 2.0 

    for batch in loader:
        pred_cls, pred_reg = model(batch)
        batch_boxes = decode_predictions(pred_cls, pred_reg)
        gt_box_list = batch['boxes']
        
        for b in range(len(batch_boxes)):
            # 1. Get predictions and sort by confidence (Descending)
            preds = batch_boxes[b] # [x, y, z, l, w, h, yaw, score]
            preds.sort(key=lambda x: x[7], reverse=True)
            
            # 2. Get Ground Truth
            gt_boxes = gt_box_list[b].numpy()
            total_gt_objects += len(gt_boxes)
            
            # Keep track of which GT boxes have already been matched (Greedy matching)
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)
            
            # 3. Match Predictions to GT
            for p in preds:
                p_center = np.array(p[:2])
                score = p[7]
                
                best_dist = float('inf')
                best_gt_idx = -1
                
                # Find closest unmatched GT
                if len(gt_boxes) > 0:
                    dists = np.linalg.norm(gt_boxes[:, :2] - p_center, axis=1)
                    # Sort by distance to find closest
                    sorted_idxs = np.argsort(dists)
                    
                    for idx in sorted_idxs:
                        if not gt_matched[idx]:
                            best_dist = dists[idx]
                            best_gt_idx = idx
                            break
                
                # Determine if TP or FP
                if best_dist < MATCH_DIST_THRESHOLD:
                    gt_matched[best_gt_idx] = True
                    all_predictions.append({'score': score, 'tp': 1})
                else:
                    all_predictions.append({'score': score, 'tp': 0})

    # --- Compute Metrics ---
    
    if total_gt_objects == 0:
        return 0.0, 0.0, 0.0

    # Sort all predictions globally by score
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    
    tps = np.array([x['tp'] for x in all_predictions])
    fps = 1 - tps
    
    tp_cumsum = np.cumsum(tps)
    fp_cumsum = np.cumsum(fps)
    
    recalls = tp_cumsum / total_gt_objects
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # --- Calculate Average Precision (AP) ---
    # We use the "All Points" interpolation method (Area Under Curve)
    
    # Pad with 0 and 1 for integration
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute the precision envelope (ensure curve is monotonically decreasing)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
    # Calculate Area Under Curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    # Simple F1/Rec/Prec based on the final cumulative state (or best F1 point)
    # Here we just return the values at the lowest confidence threshold used
    final_rec = recalls[-1] if len(recalls) > 0 else 0
    final_prec = precisions[-1] if len(precisions) > 0 else 0
    final_f1 = 2 * final_prec * final_rec / max(1e-6, final_prec + final_rec)
    
    print(f"\nEvaluation Complete: {len(all_predictions)} Predictions vs {total_gt_objects} GTs")
    print(f"mAP (AP@2.0m): {ap:.4f}")

    return final_rec, final_prec, final_f1

def train(model, loader, opt, scaler, epoch):
    """
    Executes one training epoch with Automatic Mixed Precision (AMP).
    """
    model.train()
    epoch_loss = 0
    t0 = time.time()
    
    for i, batch in enumerate(loader):
        # AMP Context: Runs op in FP16 where safe, FP32 where needed.
        with autocast('cuda'):
            pred_cls, pred_reg = model(batch)
            gt_cls, gt_reg, gt_mask = build_targets(batch['boxes'], pred_cls.shape, 'cuda')
            loss = compute_loss(pred_cls, pred_reg, gt_cls, gt_reg, gt_mask)
        
        if torch.isnan(loss):
            opt.zero_grad(); continue

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt)
        scaler.update()
        epoch_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Ep {epoch} [{i}/{len(loader)}] | Loss: {loss.item():.4f} | Max Conf: {pred_cls.max().item():.4f} | Pos: {gt_mask.sum().item()} | Time: {time.time()-t0:.2f}s")
            t0 = time.time()
            
    return epoch_loss / len(loader)

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='Operation mode: train or infer')
    parser.add_argument('--dataset', default='nuscenes', help='Dataset name')
    parser.add_argument('--data_root', default='./v1.0-mini', help='Root path to dataset')
    parser.add_argument('--nusc_version', default='v1.0-mini', help='Nuscenes version')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs')
    parser.add_argument('--checkpoint', default=None, help='Path to load checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='Enable evaluation metrics')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    args = parser.parse_args()
    
    # 1. AUTO-OPTIMIZE CONFIGURATION
    # Adjusts batch size and settings based on specific GPU hardware.
    global CONFIG
    CONFIG = auto_tune_environment(CONFIG)
    
    seed_everything()
    device = torch.device('cuda')
    mkdir('./checkpoints')
    
    # 2. DATASETS
    train_ds = NuScenesDataset(args.data_root, args.nusc_version, augment=True)
    val_ds = NuScenesDataset(args.data_root, args.nusc_version, augment=False)
    
    # 3. OPTIMIZED DATALOADERS
    # pinned memory + persistent workers + prefetching = Saturated GPU.
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        collate_fn=fast_collate_fn, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True,     # Uses page-locked memory for faster Host-to-Device transfer
        prefetch_factor=2,   # Buffers upcoming batches
        persistent_workers=True # Keeps worker processes alive between epochs
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        collate_fn=fast_collate_fn, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    model = SparsePillarNeXt().to(device)
    
    # Print architecture only in training mode
    if args.mode == 'train':
        print("\n" + "="*50)
        print(f"Model Architecture: {model.__class__.__name__}")
        print("="*50)
        print(model)
        print("="*50 + "\n")

    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scaler = GradScaler('cuda')
    
    start_epoch = 1
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded {args.checkpoint}")
    
    # --- Training Loop ---
    if args.mode == 'train':
        best_f1 = 0
        for ep in range(start_epoch, args.epochs + 1):
            loss = train(model, train_loader, opt, scaler, ep)
            torch.save({'model': model.state_dict()}, './checkpoints/ckpt_last.pth')
            
            # Evaluate periodically
            if ep % 5 == 0 or ep > 25:
                rec, prec, f1 = evaluate(model, val_loader)
                print(f"Epoch {ep} >> R: {rec*100:.2f} | P: {prec*100:.2f} | F1: {f1*100:.2f}")
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save({'model': model.state_dict()}, './checkpoints/best.pth')
                    print("Saved Best!")
                    
    # --- Inference Loop ---
    elif args.mode == 'infer':
        if args.visualize:
            # We use batch size 1 for visualization to control the flow.
            viz_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=fast_collate_fn)
            print("Starting Visualization...")
            for batch in viz_loader:
                pred_cls, pred_reg = model(batch)
                batch_boxes = decode_predictions(pred_cls, pred_reg)
                # Visualize the raw points and decoded boxes
                visualize_sample(batch['points'], batch['boxes'], batch_boxes)
        else:
            rec, prec, f1 = evaluate(model, val_loader)
            print(f"Result >> R: {rec*100:.2f} | P: {prec*100:.2f} | F1: {f1*100:.2f}")

if __name__ == '__main__':
    main()