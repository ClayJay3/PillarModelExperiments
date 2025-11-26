#!/usr/bin/env python3
"""
run_detection_pipeline.py

Single-file 3D detection pipeline.
MODEL: PillarNeXt Architecture Clone (Dense Implementation).
Structure: PillarEncoder -> 4-Stage ResNet -> ASPP Neck -> Upsampling CenterHead.

Usage:
  python run_detection_pipeline.py --mode train --dataset nuscenes \
      --data_root ./v1.0-mini --nusc_version v1.0-mini --epochs 40

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

# --- Configuration ---
CONFIG = {
    'x_range': (-51.2, 51.2),
    'y_range': (-51.2, 51.2),
    'z_range': (-5.0, 3.0),
    'grid_size': 0.16,  # 640x640 input
    'batch_size': 2,
    'num_workers': 4,
    'lr': 0.001, # Lower learning rate for deeper network
    'weight_decay': 0.01,
}

# --- Utilities ---
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mkdir(path): os.makedirs(path, exist_ok=True)

# --- Visualizer ---
def get_box_lineset(box, color):
    import open3d as o3d
    x, y, z, l, w, h, yaw = box[:7]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d = np.dot(R, corners_3d)
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    corners_3d = corners_3d.T
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_3d)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
    return line_set

def visualize_sample(points, gt_boxes, pred_boxes):
    import open3d as o3d
    geoms = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (points.shape[0], 1)))
    geoms.append(pcd)
    for box in gt_boxes: geoms.append(get_box_lineset(box, [0, 0, 0]))
    for box in pred_boxes: geoms.append(get_box_lineset(box, [1, 0, 0]))
    xr, yr, zr = CONFIG['x_range'], CONFIG['y_range'], CONFIG['z_range']
    l, w, h = xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]
    geoms.append(get_box_lineset([(xr[0]+xr[1])/2, (yr[0]+yr[1])/2, (zr[0]+zr[1])/2, l, w, h, 0], [0, 0, 1]))
    o3d.visualization.draw_geometries(geoms, window_name="PillarNeXt Clone")

# --- Dataset ---
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
        if 'vehicle' in cat_name.lower(): return 0
        return None 

    def __getitem__(self, idx):
        token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', token)
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pc_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        pc = LidarPointCloud.from_file(pc_path)
        pts = pc.points[:4, :].T.astype(np.float32)
        pts[:, 3] /= 255.0 

        cs_rec = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose_rec = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        q_cs_inv = Quaternion(cs_rec['rotation']).inverse
        t_cs = np.array(cs_rec['translation'])
        q_pose_inv = Quaternion(pose_rec['rotation']).inverse
        t_pose = np.array(pose_rec['translation'])

        boxes = []
        for ann_t in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_t)
            cls_id = self._map_class(ann['category_name'])
            if cls_id is None: continue
            box_glob = np.array(ann['translation'])
            box_ego = q_pose_inv.rotate(box_glob - t_pose)
            box_sens = q_cs_inv.rotate(box_ego - t_cs)
            q_box = Quaternion(ann['rotation'])
            yaw, _, _ = (q_cs_inv * q_pose_inv * q_box).yaw_pitch_roll
            l, w, h = ann['size'][1], ann['size'][0], ann['size'][2]
            
            # Filter range
            if (box_sens[0] < CONFIG['x_range'][0] or box_sens[0] > CONFIG['x_range'][1] or 
                box_sens[1] < CONFIG['y_range'][0] or box_sens[1] > CONFIG['y_range'][1]):
                continue 
            boxes.append([box_sens[0], box_sens[1], box_sens[2], l, w, h, yaw, int(cls_id)])
            
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 8), dtype=np.float32)

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

        return {'points': pts, 'boxes': boxes}

def collate_fn(batch): return batch

# --- Model: PillarNeXt Clone ---

class PillarEncoder(nn.Module):
    def __init__(self, out_c=64):
        super().__init__()
        self.grid_size = CONFIG['grid_size']
        self.x_range = CONFIG['x_range']
        self.y_range = CONFIG['y_range']
        self.nx = int((self.x_range[1] - self.x_range[0]) / self.grid_size)
        self.ny = int((self.y_range[1] - self.y_range[0]) / self.grid_size)
        self.mlp = nn.Sequential(nn.Linear(9, out_c), nn.BatchNorm1d(out_c), nn.ReLU(inplace=True))

    def forward(self, points_list):
        device = next(self.parameters()).device
        batch_grids = []
        for pts in points_list:
            keep = (pts[:,0] >= self.x_range[0]) & (pts[:,0] < self.x_range[1]) & \
                   (pts[:,1] >= self.y_range[0]) & (pts[:,1] < self.y_range[1])
            pts = pts[keep]
            if len(pts) == 0:
                batch_grids.append(torch.zeros(64, self.ny, self.nx, device=device))
                continue
            pts_t = torch.from_numpy(pts).to(device)
            coor_x = ((pts_t[:, 0] - self.x_range[0]) / self.grid_size).long()
            coor_y = ((pts_t[:, 1] - self.y_range[0]) / self.grid_size).long()
            c_x = coor_x.float() * self.grid_size + self.x_range[0] + self.grid_size/2
            c_y = coor_y.float() * self.grid_size + self.y_range[0] + self.grid_size/2
            feats = torch.cat([pts_t, pts_t[:, :2] - torch.stack([c_x, c_y], dim=1), pts_t[:, 2:3], pts_t[:, :2]], dim=1)
            feats = self.mlp(feats)
            indices = coor_y * self.nx + coor_x
            order = torch.argsort(indices)
            indices = indices[order]
            feats = feats[order]
            unique_idx, counts = torch.unique_consecutive(indices, return_counts=True)
            feat_chunks = torch.split(feats, counts.tolist())
            max_feats = torch.stack([chunk.max(dim=0)[0] for chunk in feat_chunks])
            grid = torch.zeros(64, self.ny * self.nx, dtype=max_feats.dtype, device=device)
            grid[:, unique_idx] = max_feats.T
            batch_grids.append(grid.view(64, self.ny, self.nx))
        return torch.stack(batch_grids)

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))
    def forward(self, x):
        ident = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: ident = self.downsample(x)
        return F.relu(out + ident)

class PillarNeXtBackbone(nn.Module):
    """
    Matches diagram: 4 Stages of Residual Blocks
    """
    def __init__(self):
        super().__init__()
        # Stage 1: H, W, C (Stride 1)
        self.stage1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )
        # Stage 2: H/2, W/2, 2C (Stride 2)
        self.stage2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1)
        )
        # Stage 3: H/4, W/4, 4C (Stride 2)
        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1)
        )
        # Stage 4: H/8, W/8, 4C (Stride 2) - Note diagram says 4C, typically doubles, we'll use 512
        self.stage4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1)
        )
        self.out_c = 512

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

class ASPP(nn.Module):
    """
    Matches diagram: Parallel convolutions with different dilations
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # Dilation rates from diagram: 6, 12, 18
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, in_c, 1, bias=False), nn.BatchNorm2d(in_c), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_c, in_c, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(in_c), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_c, in_c, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(in_c), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_c, in_c, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(in_c), nn.ReLU())
        
        # Concat -> Reduce
        self.project = nn.Sequential(
            nn.Conv2d(in_c * 4, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.project(out)

class UpsamplingHead(nn.Module):
    """
    Matches diagram: Deconv -> Final Heads
    """
    def __init__(self, in_c, n_classes=1):
        super().__init__()
        # Diagram shows Deconv to upsample. 
        # Input is H/8. We want at least H/4 or H/2 for detection.
        # Let's upsample x2 to get H/4 resolution.
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Task Heads
        self.conv_cls = nn.Conv2d(256, n_classes, 1)
        self.conv_reg = nn.Conv2d(256, 8, 1)
        
        self.conv_cls.bias.data.fill_(-4.6)

    def forward(self, x):
        x = self.up(x)
        return torch.sigmoid(self.conv_cls(x)), self.conv_reg(x)

class PillarNeXtClone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PillarEncoder()
        self.backbone = PillarNeXtBackbone() # 4-Stage ResNet
        self.neck = ASPP(self.backbone.out_c, 256) # ASPP
        self.head = UpsamplingHead(256) # Deconv Head
        
        # Calculate total stride:
        # Encoder (1) -> Stage 1 (1) -> Stage 2 (2) -> Stage 3 (4) -> Stage 4 (8) -> ASPP (8) -> Head Up (4)
        self.stride = 4 

    def forward(self, x):
        x = self.encoder(x)
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(x)

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
    B, _, H, W = feature_shape
    hm = torch.zeros(B, 1, H, W, device=device)
    reg = torch.zeros(B, 8, H, W, device=device)
    mask = torch.zeros(B, 1, H, W, device=device)
    
    # Effective stride is 4 (0.16 * 4 = 0.64m resolution)
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
                reg[b, 3, iy, ix] = float(math.log(max(l, 0.01)))
                reg[b, 4, iy, ix] = float(math.log(max(w, 0.01)))
                reg[b, 5, iy, ix] = float(math.log(max(h, 0.01)))
                reg[b, 6, iy, ix] = float(math.sin(yaw))
                reg[b, 7, iy, ix] = float(math.cos(yaw))
    return hm, reg, mask

def compute_loss(pred_cls, pred_reg, gt_cls, gt_reg, gt_mask):
    pos_inds = gt_cls.eq(1).float()
    neg_weights = torch.pow(1 - gt_cls, 4)
    pred_cls = torch.clamp(pred_cls, 1e-6, 1 - 1e-6)
    pos_loss = torch.log(pred_cls) * torch.pow(1 - pred_cls, 2) * pos_inds
    neg_loss = torch.log(1 - pred_cls) * torch.pow(pred_cls, 2) * neg_weights * gt_cls.lt(1).float()
    loss_cls = - (pos_loss.sum() + neg_loss.sum()) / max(1, pos_inds.sum())
    mask = gt_mask.expand_as(pred_reg).bool()
    loss_reg = F.l1_loss(pred_reg[mask], gt_reg[mask], reduction='sum') / max(1, pos_inds.sum())
    return loss_cls + 2.0 * loss_reg

# --- Training ---

def train(model, loader, opt, scaler, epoch):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(loader):
        with autocast('cuda'):
            pred_cls, pred_reg = model([b['points'] for b in batch])
            gt_cls, gt_reg, gt_mask = build_targets([b['boxes'] for b in batch], pred_cls.shape, 'cuda')
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
            print(f"Ep {epoch} | Loss: {loss.item():.4f} | Max Conf: {pred_cls.max().item():.4f} | Pos: {gt_mask.sum()}")
    return epoch_loss / len(loader)

@torch.no_grad()
def decode_predictions(pred_cls, pred_reg):
    B, _, H, W = pred_cls.shape
    stride = 4
    grid_size = CONFIG['grid_size'] * stride
    
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
            cx = (ix + reg[0]) * grid_size + CONFIG['x_range'][0]
            cy = (iy + reg[1]) * grid_size + CONFIG['y_range'][0]
            cz = reg[2]
            l, w, h = torch.exp(reg[3]), torch.exp(reg[4]), torch.exp(reg[5])
            yaw = torch.atan2(reg[6], reg[7])
            boxes.append([cx.item(), cy.item(), cz.item(), l.item(), w.item(), h.item(), yaw.item()])
        batch_boxes.append(boxes)
    return batch_boxes

@torch.no_grad()
def evaluate(model, loader):
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
                dists = np.linalg.norm(gt_xy[:, None] - pred_xy[None, :], axis=2)
                matches += np.any(dists < 2.0, axis=1).sum()
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
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    seed_everything()
    device = torch.device('cuda')
    mkdir('./checkpoints')
    
    train_ds = NuScenesDataset(args.data_root, args.nusc_version, augment=True)
    val_ds = NuScenesDataset(args.data_root, args.nusc_version, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    model = PillarNeXtClone().to(device)
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
            if ep % 5 == 0 or ep > 25:
                rec, prec, f1 = evaluate(model, val_loader)
                print(f"Epoch {ep} >> R: {rec*100:.2f} | P: {prec*100:.2f} | F1: {f1*100:.2f}")
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save({'model': model.state_dict()}, './checkpoints/best.pth')
                    print("Saved Best!")
    elif args.mode == 'infer':
        if args.visualize:
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