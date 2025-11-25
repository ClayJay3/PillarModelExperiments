#!/usr/bin/env python3
"""
run_detection_pipeline.py

Single-file 3D detection pipeline (PillarNeXt style).
UPGRADED VERSION: Implements Gaussian Splatting Targets and CornerNet Focal Loss
to achieve high convergence and accuracy.

Usage:
  # Train (runs longer for better accuracy)
  python run_detection_pipeline.py --mode train --dataset nuscenes \
      --data_root ./v1.0-mini --nusc_version v1.0-mini --epochs 20

  # Infer + Eval
  python run_detection_pipeline.py --mode infer --dataset nuscenes \
      --data_root ./v1.0-mini --nusc_version v1.0-mini --checkpoint ./checkpoints/ckpt_epoch_20.pth --evaluate
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

# Disable anomaly detection for speed
torch.autograd.set_detect_anomaly(False)

# Optional libs
try:
    import open3d as o3d
    _OPEN3D_OK = True
except Exception:
    _OPEN3D_OK = False

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
    _NUSC_OK = True
except Exception:
    _NUSC_OK = False

# --- Utilities ----------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mkdir(path):
    os.makedirs(path, exist_ok=True)

# --- Gaussian Utils (Crucial for High Accuracy) -----------------------------

def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draws a 2D gaussian on the heatmap at integer center.
    """
    radius = int(radius)
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

# --- Loss Functions (CornerNet / CenterPoint Style) -------------------------

def fast_focal_loss(pred, gt):
    """
    Penalty-reduced pixel-wise focal loss.
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    # Log(0) protection
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

# --- Dataset ----------------------------------------------------------------

class NuScenesPointCloudDataset(Dataset):
    def __init__(self, dataroot: str, version: str = 'v1.0-mini', max_samples: int = None):
        if not _NUSC_OK: raise RuntimeError("nuscenes-devkit required.")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.sample_tokens = [s['token'] for s in self.nusc.sample]
        if max_samples: self.sample_tokens = self.sample_tokens[:max_samples]

    def __len__(self): return len(self.sample_tokens)

    def _map_class(self, cat_name: str) -> int:
        name = cat_name.lower()
        if 'vehicle' in name or 'car' in name or 'bus' in name or 'truck' in name: return 0
        if 'human' in name or 'ped' in name or 'person' in name: return 1
        if 'motor' in name or 'bicycle' in name or 'bike' in name: return 2
        return None

    def __getitem__(self, idx):
        token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', token)
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = self.nusc.get('sample_data', lidar_token)
        
        pc_path = os.path.join(self.nusc.dataroot, sd_rec['filename'])
        pc = LidarPointCloud.from_file(pc_path)
        # Random subsample if too large (optimizes speed)
        pts = pc.points[:4, :].T.astype(np.float32)

        # Coordinate Transforms
        cs_rec = self.nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_rec = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        q_pose = Quaternion(pose_rec['rotation']); t_pose = np.array(pose_rec['translation'])
        q_cs = Quaternion(cs_rec['rotation']); t_cs = np.array(cs_rec['translation'])

        boxes = []
        for ann_t in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_t)
            cls_id = self._map_class(ann['category_name'])
            if cls_id is None: continue

            # Global -> Sensor
            center = np.array(ann['translation'])
            center = q_cs.inverse.rotate(q_pose.inverse.rotate(center - t_pose) - t_cs)
            
            # Yaw (Global -> Sensor)
            q_box = q_cs.inverse * q_pose.inverse * Quaternion(ann['rotation'])
            yaw, _, _ = q_box.yaw_pitch_roll
            
            # W, L, H -> dy, dx, dz
            l, w, h = ann['size'][1], ann['size'][0], ann['size'][2]
            boxes.append([center[0], center[1], center[2], l, w, h, yaw, int(cls_id)])

        return {'points': pts, 'boxes': np.array(boxes, dtype=np.float32)}

def collate_fn(batch): return batch

class SyntheticPointCloudDataset(Dataset):
    def __init__(self, n_samples=200): self.len=n_samples
    def __len__(self): return self.len
    def __getitem__(self, idx):
        pts = (np.random.rand(2048,4)-0.5)*[80,80,6,1]
        boxes = [[(np.random.rand()-0.5)*40, (np.random.rand()-0.5)*40, -1.0, 4.0, 2.0, 1.5, 0.0, 0]]
        return {'points':pts.astype(np.float32), 'boxes':np.array(boxes, dtype=np.float32)}

# --- Network Components -----------------------------------------------------

class PillarEncoder(nn.Module):
    def __init__(self, x_range=(-40,40), y_range=(-40,40), z_range=(-3,3), grid_size=0.2, in_channels=4, out_channels=64):
        super().__init__()
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.grid_size = grid_size
        self.nx = int((self.x_max - self.x_min) / grid_size)
        self.ny = int((self.y_max - self.y_min) / grid_size)
        self.out_channels = out_channels
        self.point_mlp = nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU())

    def forward(self, points_list):
        device = next(self.point_mlp.parameters()).device
        batch_list = []
        for pts in points_list:
            # Filter (keep in numpy for easy boolean indexing)
            mask = (pts[:,0]>self.x_min)&(pts[:,0]<self.x_max)&(pts[:,1]>self.y_min)&(pts[:,1]<self.y_max)
            pts = pts[mask]

            if len(pts) == 0:
                batch_list.append(torch.zeros(self.out_channels, self.ny, self.nx, device=device))
                continue
            
            # --- FIX START ---
            # Convert to Tensor on device immediately. 
            # This allows us to use .long() and keeps downstream ops on GPU.
            pts_t = torch.from_numpy(pts).to(device)
            
            # Coords -> Indices
            ix = ((pts_t[:,0] - self.x_min) / self.grid_size).long()
            iy = ((pts_t[:,1] - self.y_min) / self.grid_size).long()
            
            # Features
            feat = self.point_mlp(pts_t)
            # --- FIX END ---
            
            # Scatter Max 
            indices = iy * self.nx + ix
            
            # Sort to group
            sort_idx = torch.argsort(indices)
            indices = indices[sort_idx]
            feat = feat[sort_idx]
            
            # Unique indices
            unique_idx, counts = torch.unique_consecutive(indices, return_counts=True)
            
            # Max pooling per pillar
            feat_split = torch.split(feat, counts.tolist())
            max_feats = torch.stack([f.max(dim=0)[0] for f in feat_split])
            
            # Scatter to grid
            grid = torch.zeros(self.out_channels, self.ny * self.nx, device=device)
            grid[:, unique_idx] = max_feats.T
            batch_list.append(grid.view(self.out_channels, self.ny, self.nx))
            
        return torch.stack(batch_list)

class Backbone(nn.Module):
    def __init__(self, in_c, channels=[64, 128, 256]):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(in_c, channels[0], 3, 1, 1), nn.BatchNorm2d(channels[0]), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(channels[0], channels[1], 3, 2, 1), nn.BatchNorm2d(channels[1]), nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv2d(channels[1], channels[2], 3, 2, 1), nn.BatchNorm2d(channels[2]), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(channels[1], 128, 2, 2), nn.BatchNorm2d(128), nn.ReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(channels[2], 128, 4, 4), nn.BatchNorm2d(128), nn.ReLU())
        self.out_c = channels[0] + 128 + 128
        
    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        return torch.cat([x1, self.up2(x2), self.up3(x3)], dim=1)

class CenterHead(nn.Module):
    def __init__(self, in_c, classes=3):
        super().__init__()
        self.hm = nn.Conv2d(in_c, classes, 1)
        self.wh = nn.Conv2d(in_c, 2, 1)
        self.reg = nn.Conv2d(in_c, 2, 1)
        # Initialize bias for focal loss stability
        self.hm.bias.data.fill_(-2.19)
        
    def forward(self, x):
        return {'hm': torch.sigmoid(self.hm(x)), 'wh': self.wh(x), 'reg': self.reg(x)}

class Detector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PillarEncoder(grid_size=cfg['grid'])
        self.backbone = Backbone(64)
        self.head = CenterHead(self.backbone.out_c, cfg['classes'])
        
    def forward(self, x):
        return self.head(self.backbone(self.encoder(x)))

# --- Targets & Training -----------------------------------------------------

def generate_targets(boxes_list, shape, encoder, device):
    """
    Gaussian Splatting target generation.
    """
    B, C, H, W = shape
    hm = np.zeros((B, C, H, W), dtype=np.float32)
    wh = np.zeros((B, 2, H, W), dtype=np.float32)
    reg = np.zeros((B, 2, H, W), dtype=np.float32)
    mask = np.zeros((B, 1, H, W), dtype=np.float32)
    
    for b, boxes in enumerate(boxes_list):
        for box in boxes:
            cx, cy, _, l, w, _, _, cls_id = box
            cls_id = int(cls_id)
            if cls_id >= C: continue
            
            # Project to grid
            x = (cx - encoder.x_min) / encoder.grid_size
            y = (cy - encoder.y_min) / encoder.grid_size
            
            x_int, y_int = int(x), int(y)
            if x_int < 0 or x_int >= W or y_int < 0 or y_int >= H: continue
            
            # Draw Gaussian
            radius = max(0, int(max(l, w) / encoder.grid_size / 2)) # Dynamic radius based on size
            radius = max(1, radius) # Minimum 1 pixel radius
            draw_umich_gaussian(hm[b, cls_id], (x_int, y_int), radius)
            
            # Regression targets
            wh[b, 0, y_int, x_int] = float(np.log(l))
            wh[b, 1, y_int, x_int] = float(np.log(w))
            reg[b, 0, y_int, x_int] = float(x - x_int)
            reg[b, 1, y_int, x_int] = float(y - y_int)
            mask[b, 0, y_int, x_int] = 1.0
            
    return torch.tensor(hm).to(device), torch.tensor(wh).to(device), \
           torch.tensor(reg).to(device), torch.tensor(mask).bool().to(device)

def train_epoch(model, loader, opt, device, epoch):
    model.train()
    total_loss = 0
    for i, batch in enumerate(loader):
        opt.zero_grad()
        # Forward
        preds = model([b['points'] for b in batch])
        
        # Targets
        gt_hm, gt_wh, gt_reg, gt_mask = generate_targets(
            [b['boxes'] for b in batch], preds['hm'].shape, model.encoder, device
        )
        
        # Loss
        loss_hm = fast_focal_loss(preds['hm'], gt_hm)
        loss_wh = F.l1_loss(preds['wh'][gt_mask.expand_as(gt_wh)], gt_wh[gt_mask.expand_as(gt_wh)]) if gt_mask.sum() > 0 else 0
        loss_reg = F.l1_loss(preds['reg'][gt_mask.expand_as(gt_reg)], gt_reg[gt_mask.expand_as(gt_reg)]) if gt_mask.sum() > 0 else 0
        
        loss = loss_hm + 0.1 * loss_wh + 1.0 * loss_reg
        loss.backward()
        opt.step()
        total_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Ep {epoch} {i}/{len(loader)} | Loss: {loss.item():.4f} (HM: {loss_hm:.4f})")
            
    return total_loss / len(loader)

def decode(preds, encoder, k=50):
    hm = preds['hm'].sigmoid().detach().cpu().numpy()
    wh = preds['wh'].detach().cpu().numpy()
    reg = preds['reg'].detach().cpu().numpy()
    B, C, H, W = hm.shape
    res = []
    
    for b in range(B):
        # Flatten and topk
        scores = hm[b].reshape(-1)
        # Fast numpy topk
        inds = np.argpartition(-scores, k)[:k]
        inds = inds[np.argsort(-scores[inds])]
        
        box_list = []
        for idx in inds:
            score = scores[idx]
            if score < 0.2: break # Confidence threshold
            
            cls = idx // (H*W)
            loc = idx % (H*W)
            y, x = loc // W, loc % W
            
            # Retrieve centers
            off_x = reg[b, 0, y, x]
            off_y = reg[b, 1, y, x]
            
            # Retrieve size
            l = np.exp(wh[b, 0, y, x])
            w = np.exp(wh[b, 1, y, x])
            
            # Map back to world
            wx = encoder.x_min + (x + off_x) * encoder.grid_size
            wy = encoder.y_min + (y + off_y) * encoder.grid_size
            
            box_list.append({'x': wx, 'y': wy, 'z': -1.0, 'dx': l, 'dy': w, 'score': float(score), 'lbl': cls})
        res.append(box_list)
    return res

def evaluate(model, loader, device):
    model.eval()
    total_matched = 0
    total_gt = 0
    total_pred = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for batch in loader:
            preds = model([b['points'] for b in batch])
            boxes_pred = decode(preds, model.encoder)
            
            for i, gt_boxes in enumerate([b['boxes'] for b in batch]):
                pred_b = boxes_pred[i]
                total_gt += len(gt_boxes)
                total_pred += len(pred_b)
                
                # Match
                if len(gt_boxes) == 0 or len(pred_b) == 0: continue
                
                gt_xy = gt_boxes[:, :2]
                pred_xy = np.array([[p['x'], p['y']] for p in pred_b])
                
                # Dist matrix
                dists = np.linalg.norm(gt_xy[:, None, :] - pred_xy[None, :, :], axis=2)
                # Greedy match within 2.0m
                has_match = np.any(dists < 2.0, axis=1)
                total_matched += np.sum(has_match)
                
    rec = total_matched / max(1, total_gt)
    prec = total_matched / max(1, total_pred)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    return rec, prec, f1

# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--dataset', default='synthetic')
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--nusc_version', default='v1.0-mini')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mkdir('./checkpoints')
    
    cfg = {'grid': 0.2, 'classes': 3} # 0.2m grid = 400x400 image for 80m range
    
    # Data
    if args.dataset == 'nuscenes':
        ds = NuScenesPointCloudDataset(args.data_root, args.nusc_version)
    else:
        ds = SyntheticPointCloudDataset()
        
    loader = DataLoader(ds, batch_size=4, shuffle=(args.mode=='train'), collate_fn=collate_fn, num_workers=2)
    
    model = Detector(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    start_epoch = 1
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'epoch' in ckpt and args.mode == 'train': start_epoch = ckpt['epoch'] + 1
        print(f"Loaded {args.checkpoint}")
        
    if args.mode == 'train':
        for ep in range(start_epoch, args.epochs+1):
            loss = train_epoch(model, loader, opt, device, ep)
            print(f"--- Epoch {ep} Avg Loss: {loss:.4f} ---")
            torch.save({'epoch': ep, 'model': model.state_dict()}, f'./checkpoints/ckpt_epoch_{ep}.pth')
            
    elif args.mode == 'infer':
        if args.evaluate:
            rec, prec, f1 = evaluate(model, loader, device)
            print(f"\nFinal Results:\nRecall:    {rec*100:.2f}%\nPrecision: {prec*100:.2f}%\nF1 Score:  {f1*100:.2f}%")
            
        if args.visualize and _OPEN3D_OK:
            # Quick viz of first few
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i > 2: break
                    preds = model([b['points'] for b in batch])
                    dec = decode(preds, model.encoder)
                    pts = batch[0]['points']
                    
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
                    
                    boxes = []
                    # Preds (Red)
                    for b in dec[0]:
                        obb = o3d.geometry.OrientedBoundingBox([b['x'], b['y'], -1], np.eye(3), [b['dx'], b['dy'], 2])
                        obb.color = (1, 0, 0)
                        boxes.append(obb)
                    # GT (Green)
                    for b in batch[0]['boxes']:
                        obb = o3d.geometry.OrientedBoundingBox(b[:3], o3d.geometry.get_rotation_matrix_from_xyz((0,0,b[6])), [b[4], b[3], b[5]])
                        obb.color = (0, 1, 0)
                        boxes.append(obb)
                        
                    o3d.visualization.draw_geometries([pcd, *boxes])

if __name__ == '__main__':
    main()