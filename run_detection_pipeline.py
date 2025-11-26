#!/usr/bin/env python3
"""
run_detection_pipeline.py

Single-file 3D detection pipeline (PointPillars).
DIAGNOSTIC VERSION: Includes deep logging to debug the '0% Recall' issue.
FIXED: Boolean casting for regression mask indices.

Usage:
  python run_detection_pipeline.py --mode train --dataset nuscenes \
      --data_root ./v1.0-mini --nusc_version v1.0-mini --epochs 30

  python run_detection_pipeline.py --mode infer --dataset nuscenes \
      --data_root ./v1.0-mini --nusc_version v1.0-mini --checkpoint ./checkpoints/best.pth --evaluate
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

# --- Configuration ---
CONFIG = {
    'x_range': (-51.2, 51.2),
    'y_range': (-51.2, 51.2),
    'z_range': (-5.0, 3.0),
    'grid_size': 0.16,  # 0.16m grid -> 640x640 input
    'batch_size': 2,    # Small batch for GPU memory
    'num_workers': 4,
    'lr': 0.003,
    'weight_decay': 0.01,
}

# --- Utilities ---
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mkdir(path): os.makedirs(path, exist_ok=True)

# --- Dataset (NuScenes) ---
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
    _NUSC_OK = True
except: _NUSC_OK = False

class NuScenesDataset(Dataset):
    def __init__(self, dataroot, version='v1.0-mini', augment=False):
        if not _NUSC_OK: raise RuntimeError("pip install nuscenes-devkit")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.sample_tokens = [s['token'] for s in self.nusc.sample]
        self.augment = augment

    def __len__(self): return len(self.sample_tokens)

    def _map_class(self, cat_name):
        # Simplify to just CARS for now to guarantee convergence
        if 'vehicle' in cat_name.lower(): return 0
        return None 

    def __getitem__(self, idx):
        token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', token)
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        
        # Load Points
        pc_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        pc = LidarPointCloud.from_file(pc_path)
        pts = pc.points[:4, :].T.astype(np.float32)
        pts[:, 3] /= 255.0 # Normalize intensity

        # Load Boxes (Global -> Sensor)
        cs_rec = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose_rec = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # Pre-compute inverse transforms
        q_cs_inv = Quaternion(cs_rec['rotation']).inverse
        t_cs = np.array(cs_rec['translation'])
        q_pose_inv = Quaternion(pose_rec['rotation']).inverse
        t_pose = np.array(pose_rec['translation'])

        boxes = []
        for ann_t in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_t)
            cls_id = self._map_class(ann['category_name'])
            if cls_id is None: continue
            
            # Geometry: Global -> Ego -> Sensor
            box_glob = np.array(ann['translation'])
            box_ego = q_pose_inv.rotate(box_glob - t_pose)
            box_sens = q_cs_inv.rotate(box_ego - t_cs)
            
            # Rotation: Global -> Sensor
            q_box = Quaternion(ann['rotation'])
            yaw, _, _ = (q_cs_inv * q_pose_inv * q_box).yaw_pitch_roll
            
            # w,l,h -> l,w,h (standard detection format)
            l, w, h = ann['size'][1], ann['size'][0], ann['size'][2]
            # Check if the box center is strictly inside our defined grid range.
            # CONFIG['x_range'] is (-51.2, 51.2)
            if (box_sens[0] < CONFIG['x_range'][0] or box_sens[0] > CONFIG['x_range'][1] or 
                box_sens[1] < CONFIG['y_range'][0] or box_sens[1] > CONFIG['y_range'][1]):
                continue # Skip this box, it is "off-screen"
                
            boxes.append([box_sens[0], box_sens[1], box_sens[2], l, w, h, yaw, int(cls_id)])
            
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 8), dtype=np.float32)

        # Augmentation (Flip/Rotate/Scale)
        if self.augment:
            if np.random.rand() > 0.5: # Flip X
                pts[:, 1] = -pts[:, 1]; boxes[:, 1] = -boxes[:, 1]; boxes[:, 6] = -boxes[:, 6]
            if np.random.rand() > 0.5: # Flip Y
                pts[:, 0] = -pts[:, 0]; boxes[:, 0] = -boxes[:, 0]; boxes[:, 6] = np.pi - boxes[:, 6]
            
            rot = np.random.uniform(-0.78, 0.78) # +/- 45 deg
            c, s = np.cos(rot), np.sin(rot)
            mat = np.array([[c, -s], [s, c]])
            pts[:, :2] = np.dot(pts[:, :2], mat.T)
            if len(boxes) > 0:
                boxes[:, :2] = np.dot(boxes[:, :2], mat.T)
                boxes[:, 6] += rot
            
            scale = np.random.uniform(0.95, 1.05)
            pts[:, :3] *= scale
            if len(boxes) > 0: boxes[:, :6] *= scale

        return {'points': pts, 'boxes': boxes}

def collate_fn(batch): return batch

# --- Model Components ---

class PillarEncoder(nn.Module):
    def __init__(self, out_c=64):
        super().__init__()
        self.grid_size = CONFIG['grid_size']
        self.x_range = CONFIG['x_range']
        self.y_range = CONFIG['y_range']
        self.nx = int((self.x_range[1] - self.x_range[0]) / self.grid_size)
        self.ny = int((self.y_range[1] - self.y_range[0]) / self.grid_size)
        
        # MLP for Point Features
        self.mlp = nn.Sequential(
            nn.Linear(9, out_c),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, points_list):
        device = next(self.parameters()).device
        batch_grids = []
        
        for pts in points_list:
            # 1. Voxelize
            keep = (pts[:,0] >= self.x_range[0]) & (pts[:,0] < self.x_range[1]) & \
                   (pts[:,1] >= self.y_range[0]) & (pts[:,1] < self.y_range[1])
            pts = pts[keep]
            
            if len(pts) == 0:
                batch_grids.append(torch.zeros(64, self.ny, self.nx, device=device))
                continue
                
            pts_t = torch.from_numpy(pts).to(device)
            
            # Coords
            coor_x = ((pts_t[:, 0] - self.x_range[0]) / self.grid_size).long()
            coor_y = ((pts_t[:, 1] - self.y_range[0]) / self.grid_size).long()
            
            # Centers
            c_x = coor_x.float() * self.grid_size + self.x_range[0] + self.grid_size/2
            c_y = coor_y.float() * self.grid_size + self.y_range[0] + self.grid_size/2
            
            # Features [x, y, z, i, x-xc, y-yc, z, x, y]
            feats = torch.cat([
                pts_t, 
                pts_t[:, :2] - torch.stack([c_x, c_y], dim=1),
                pts_t[:, 2:3],
                pts_t[:, :2]
            ], dim=1)
            
            # 2. Embed
            feats = self.mlp(feats)
            
            # 3. Scatter (Max Pooling)
            indices = coor_y * self.nx + coor_x
            # Sort
            order = torch.argsort(indices)
            indices = indices[order]
            feats = feats[order]
            
            unique_idx, counts = torch.unique_consecutive(indices, return_counts=True)
            
            # Max Pool Trick
            feat_chunks = torch.split(feats, counts.tolist())
            max_feats = torch.stack([chunk.max(dim=0)[0] for chunk in feat_chunks])
            
            # Map to Grid
            grid = torch.zeros(64, self.ny * self.nx, dtype=max_feats.dtype, device=device)
            grid[:, unique_idx] = max_feats.T
            batch_grids.append(grid.view(64, self.ny, self.nx))
            
        return torch.stack(batch_grids)

class Backbone(nn.Module):
    """Standard PointPillars Backbone (SECT) - Stride 2 Output"""
    def __init__(self):
        super().__init__()
        # Downsample 1 (Stride 1 -> 1)
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        )
        # Downsample 2 (Stride 1 -> 2)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
        )
        # Downsample 3 (Stride 2 -> 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
        )
        
        # Upsample (Neck)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(64, 128, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU()) # 1->1
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 128, 2, 2, bias=False), nn.BatchNorm2d(128), nn.ReLU()) # 2->1
        self.up3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 4, bias=False), nn.BatchNorm2d(128), nn.ReLU()) # 4->1
        
        self.out_c = 128 + 128 + 128

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        u1 = self.up1(x1)
        u2 = self.up2(x2)
        u3 = self.up3(x3)
        return torch.cat([u1, u2, u3], dim=1)

class DetectionHead(nn.Module):
    def __init__(self, in_c, n_classes=1):
        super().__init__()
        self.conv_cls = nn.Conv2d(in_c, n_classes, 1)
        self.conv_reg = nn.Conv2d(in_c, 8, 1) # dx, dy, dz, w, l, h, sin, cos
        
        # Initialization
        self.conv_cls.bias.data.fill_(-4.6) # Focal loss init
        
    def forward(self, x):
        cls = torch.sigmoid(self.conv_cls(x))
        reg = self.conv_reg(x)
        return cls, reg

class PointPillars(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PillarEncoder()
        self.backbone = Backbone()
        self.head = DetectionHead(self.backbone.out_c)
        self.stride = 1 

    def forward(self, x):
        return self.head(self.backbone(self.encoder(x)))

# --- Loss & Targets ---

def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
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
    """
    Builds dense heatmap and regression targets.
    """
    B, _, H, W = feature_shape
    
    hm = torch.zeros(B, 1, H, W, device=device)
    reg = torch.zeros(B, 8, H, W, device=device)
    mask = torch.zeros(B, 1, H, W, device=device)
    
    grid_size = CONFIG['grid_size'] 
    
    for b in range(B):
        for box in gt_boxes[b]:
            # Box: x, y, z, l, w, h, yaw
            x, y, z, l, w, h, yaw, cls = box
            
            # Grid Coords
            cx = (x - CONFIG['x_range'][0]) / grid_size
            cy = (y - CONFIG['y_range'][0]) / grid_size
            
            ix, iy = int(cx), int(cy)
            
            if 0 <= ix < W and 0 <= iy < H:
                # Heatmap (Gaussian Splat)
                radius = max(2, int(max(l, w) / grid_size / 2))
                draw_umich_gaussian(hm[b, 0], (ix, iy), radius)
                
                # Regression
                mask[b, 0, iy, ix] = 1.0
                
                # Offsets (0-1)
                reg[b, 0, iy, ix] = float(cx - ix)
                reg[b, 1, iy, ix] = float(cy - iy)
                reg[b, 2, iy, ix] = float(z)
                reg[b, 3, iy, ix] = float(math.log(max(l, 0.01)))
                reg[b, 4, iy, ix] = float(math.log(max(w, 0.01)))
                reg[b, 5, iy, ix] = float(math.log(max(h, 0.01)))
                reg[b, 6, iy, ix] = float(math.sin(yaw))
                reg[b, 7, iy, ix] = float(math.cos(yaw))

    return hm, reg, mask

def compute_loss(pred_cls, pred_reg, gt_cls, gt_reg, gt_mask):
    # Focal Loss
    pos_inds = gt_cls.eq(1).float()
    neg_inds = gt_cls.lt(1).float()
    neg_weights = torch.pow(1 - gt_cls, 4)
    
    pred_cls = torch.clamp(pred_cls, 1e-6, 1 - 1e-6)
    pos_loss = torch.log(pred_cls) * torch.pow(1 - pred_cls, 2) * pos_inds
    neg_loss = torch.log(1 - pred_cls) * torch.pow(pred_cls, 2) * neg_weights * neg_inds
    
    num_pos = pos_inds.sum()
    loss_cls = - (pos_loss.sum() + neg_loss.sum()) / max(1, num_pos)
    
    # Regression Loss (L1)
    # FIX: Cast to bool for indexing
    mask = gt_mask.expand_as(pred_reg).bool()
    loss_reg = F.l1_loss(pred_reg[mask], gt_reg[mask], reduction='sum') / max(1, num_pos)
    
    return loss_cls + 2.0 * loss_reg

# --- Main Loop ---

def train(model, loader, opt, scaler, epoch):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(loader):
        with autocast('cuda'):
            pred_cls, pred_reg = model([b['points'] for b in batch])
            gt_cls, gt_reg, gt_mask = build_targets([b['boxes'] for b in batch], pred_cls.shape, 'cuda')
            loss = compute_loss(pred_cls, pred_reg, gt_cls, gt_reg, gt_mask)
        
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(opt)
        scaler.update()
        
        epoch_loss += loss.item()
        
        if i % 10 == 0:
            # DIAGNOSTIC: Print max confidence
            max_conf = pred_cls.max().item()
            print(f"Ep {epoch} | Loss: {loss.item():.4f} | Max Conf: {max_conf:.4f} | Pos: {gt_mask.sum()}")
            
    return epoch_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    matches = 0
    total_gt = 0
    total_pred = 0
    
    for batch in loader:
        pred_cls, pred_reg = model([b['points'] for b in batch])
        
        B, _, H, W = pred_cls.shape
        
        # Simple NMS
        nms_kernel = 3
        pad = (nms_kernel - 1) // 2
        hmax = F.max_pool2d(pred_cls, (nms_kernel, nms_kernel), stride=1, padding=pad)
        keep = (hmax == pred_cls).float()
        pred_cls = pred_cls * keep
        
        # Decode
        for b in range(B):
            scores = pred_cls[b, 0].view(-1)
            # DIAGNOSTIC: Lower threshold to 0.01 to see if ANYTHING is learned
            topk_scores, topk_inds = torch.topk(scores, 50)
            
            preds_box = []
            for i in range(50):
                if topk_scores[i] < 0.05: continue # Threshold
                
                idx = topk_inds[i]
                iy = (idx // W).long()
                ix = (idx % W).long()
                
                reg = pred_reg[b, :, iy, ix]
                
                cx = (ix + reg[0]) * CONFIG['grid_size'] + CONFIG['x_range'][0]
                cy = (iy + reg[1]) * CONFIG['grid_size'] + CONFIG['y_range'][0]
                
                preds_box.append([cx.item(), cy.item()])
                
            # Match
            gt_boxes = batch[b]['boxes']
            total_gt += len(gt_boxes)
            total_pred += len(preds_box)
            
            if len(gt_boxes) > 0 and len(preds_box) > 0:
                gt_xy = gt_boxes[:, :2]
                pred_xy = np.array(preds_box)
                dists = np.linalg.norm(gt_xy[:, None] - pred_xy[None, :], axis=2)
                
                # Lenient matching (2.0m)
                has_match = np.any(dists < 2.0, axis=1)
                matches += has_match.sum()
                
                # DIAGNOSTIC: Print first match check
                if matches == 0 and len(preds_box) > 0:
                    print(f"DEBUG: GT: {gt_xy[0]} | PRED: {pred_xy[0]} | Dist: {dists[0,0]:.2f}")

    rec = matches / max(1, total_gt)
    prec = matches / max(1, total_pred)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    return rec, prec, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--dataset', default='nuscenes')
    parser.add_argument('--data_root', default='./v1.0-mini')
    parser.add_argument('--nusc_version', default='v1.0-mini')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    
    seed_everything()
    device = torch.device('cuda')
    mkdir('./checkpoints')
    
    train_ds = NuScenesDataset(args.data_root, args.nusc_version, augment=True)
    val_ds = NuScenesDataset(args.data_root, args.nusc_version, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    model = PointPillars().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scaler = GradScaler('cuda')
    
    start_epoch = 1
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded {args.checkpoint}")
    
    if args.mode == 'train':
        best_f1 = 0
        for ep in range(start_epoch, args.epochs + 1):
            loss = train(model, train_loader, opt, scaler, ep)
            
            torch.save({'model': model.state_dict()}, './checkpoints/ckpt_last.pth')
            
            if ep % 5 == 0 or ep > 20:
                rec, prec, f1 = evaluate(model, val_loader)
                print(f"Epoch {ep} >> R: {rec*100:.2f} | P: {prec*100:.2f} | F1: {f1*100:.2f}")
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save({'model': model.state_dict()}, './checkpoints/best.pth')
                    print("Saved Best!")
    
    elif args.mode == 'infer':
        rec, prec, f1 = evaluate(model, val_loader)
        print(f"Result >> R: {rec*100:.2f} | P: {prec*100:.2f} | F1: {f1*100:.2f}")

if __name__ == '__main__':
    main()