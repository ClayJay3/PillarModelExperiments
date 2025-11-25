#!/usr/bin/env python3
"""
run_detection_pipeline.py

Single-file 3D detection pipeline inspired by PillarNeXt design (pillar encoder,
backbone, configurable necks, and configurable detection heads). This is NOT the
PillarNeXt repo — it's a new-from-scratch implementation intended for experimentation.

Usage examples:
  # Quick smoke-test (synthetic data)
  python run_detection_pipeline.py --mode train --dataset synthetic --epochs 2

  # Switch neck and head
  python run_detection_pipeline.py --mode train --dataset synthetic --neck aspp --head center --epochs 2

  # To run with nuScenes:
  1) pip install nuscenes-devkit
  2) Download nuScenes dataset and put 'v1.0-mini' or 'v1.0-trainval' under --data_root
  3) python run_detection_pipeline.py --dataset nuscenes --data_root /path/to/nuscenes --mode train

  # To run with Waymo (requires Waymo data preparation and signing; Waymo dataset is very large)
  # See TODOs below and official dataset sites.

Important notes:
 - This script includes helpers for dataset download/checking but DOES NOT bypass nuScenes/Waymo auth requirements.
 - For quick iteration, use --dataset synthetic which will generate random point clouds and boxes so you can iterate on neck/head.
 - The code is intentionally modular: change NeckClass or HeadClass objects to experiment.
 - This is a research / experimental script — not production level code. Use GPUs for real training.

References (for architecture choices / motivation):
 - PillarNeXt paper (CVPR 2023): provided by user. :contentReference[oaicite:2]{index=2}
 - Project proposal (detection head experiments): provided by user. :contentReference[oaicite:3]{index=3}
"""

import os
import argparse
import random
import math
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.autograd.set_detect_anomaly(True)

# --- Utilities ----------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mkdir(path):
    os.makedirs(path, exist_ok=True)

# --- Dataset helpers ---------------------------------------------------------

class SyntheticPointCloudDataset(Dataset):
    """
    Tiny synthetic dataset to test pipeline quickly.
    Each sample: Nx4 point cloud (x,y,z,intensity) + list of boxes
    Boxes format: [x, y, z, dx, dy, dz, yaw] (center + dims + yaw)
    """
    def __init__(self, n_samples=200, points_per_sample=2048, max_objects=6, rng_seed=0):
        self.n_samples = n_samples
        self.points_per_sample = points_per_sample
        self.max_objects = max_objects
        self.rng = np.random.RandomState(rng_seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # generate random points in range [-40,40] x [-40,40] x [-3,3]
        pts = (self.rng.rand(self.points_per_sample, 4) - 0.5) * np.array([80, 80, 6, 1.0])
        n_obj = self.rng.randint(1, self.max_objects + 1)
        boxes = []
        for i in range(n_obj):
            x = (self.rng.rand() - 0.5) * 60
            y = (self.rng.rand() - 0.5) * 60
            z = (self.rng.rand() - 0.5) * 2
            dx = 1.5 + self.rng.rand() * 4.0
            dy = 0.5 + self.rng.rand() * 2.0
            dz = 1.0 + self.rng.rand() * 2.5
            yaw = (self.rng.rand() - 0.5) * math.pi
            cls = self.rng.randint(0, 3)  # 3 classes
            boxes.append([x, y, z, dx, dy, dz, yaw, cls])
        sample = {
            'points': pts.astype(np.float32),
            'boxes': np.array(boxes, dtype=np.float32)
        }
        return sample

def collate_fn(batch):
    # Very simple collate: return lists
    return batch

# --- Pillar encoder (simple) -------------------------------------------------

class PillarEncoder(nn.Module):
    """
    Convert point cloud (N x 4) into a pseudo-image (C x H x W).
    This is a super-simplified pillar encoder: it bins points into a BEV grid,
    computes per-pillar max-pooled features (and mean), and outputs a pseudo-image.

    This is intentionally simple to be easy to modify.
    """
    def __init__(self, x_range=(-40,40), y_range=(-40,40), z_range=(-3,3),
                 grid_size=0.15, in_channels=4, out_channels=64):
        super().__init__()
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.grid_size = grid_size
        self.nx = int((self.x_max - self.x_min) / grid_size)
        self.ny = int((self.y_max - self.y_min) / grid_size)
        self.out_channels = out_channels
        # small MLP for point features -> per-point embedding
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, out_channels),
            nn.ReLU()
        )

    def forward(self, points_list: List[np.ndarray]):
        """
        points_list: list of (N_i, 4) numpy arrays
        returns: torch.FloatTensor of shape (B, C, ny, nx)

        Implementation:
         - Try to use torch_scatter.scatter_max (safe + fast if installed).
         - Fallback: group by unique linear indices and compute per-group max (safe, slower).
        """
        device = next(self.point_mlp.parameters()).device
        use_torch_scatter = False
        try:
            # Attempt to import scatter_max from torch_scatter (preferred)
            from torch_scatter import scatter_max
            use_torch_scatter = True
        except Exception:
            use_torch_scatter = False

        batch = []
        for pts in points_list:
            coords = pts[:, :3]  # x,y,z
            ix = ((coords[:, 0] - self.x_min) / self.grid_size).astype(np.int64)
            iy = ((coords[:, 1] - self.y_min) / self.grid_size).astype(np.int64)
            valid_mask = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny) & \
                         (coords[:, 2] >= self.z_min) & (coords[:, 2] <= self.z_max)
            ix = ix[valid_mask]; iy = iy[valid_mask]; pts_valid = pts[valid_mask]

            if pts_valid.shape[0] == 0:
                batch.append(torch.zeros(self.out_channels, self.ny, self.nx, device=device))
                continue

            pts_feat = torch.from_numpy(pts_valid.astype(np.float32)).to(device)
            feat = self.point_mlp(pts_feat)  # M x C

            linear_idx = torch.from_numpy((iy * self.nx + ix).astype(np.int64)).to(device)  # (M,)

            num_cells = self.ny * self.nx

            if use_torch_scatter:
                # fast path using torch_scatter.scatter_max
                # feat: M x C -> transpose to C x M to scatter per-channel, but scatter_max works on last dim; we'll scatter along dim=0 grouping rows
                # scatter_max expects src shaped (M, C) and index (M,), returns (out, arg)
                # We'll compute out (num_cells x C) and transpose to (C x num_cells)
                # Note: torch_scatter.scatter_max outputs shape (num_cells, C) when dim=0 and index in [0,num_cells)
                out_max, _ = scatter_max(feat, linear_idx, dim=0, dim_size=num_cells)  # out_max: (num_cells, C)
                # transpose to C x num_cells and replace any -inf with 0
                bev_flat = out_max.transpose(0, 1).contiguous()  # C x num_cells
                # scatter_max uses -inf for empty buckets; replace them with 0
                bev_flat[bev_flat == float('-inf')] = 0.0
                bev = bev_flat.view(self.out_channels, self.ny, self.nx)
                batch.append(bev)
            else:
                # safe fallback (pure PyTorch). Group by unique indices and compute per-group max.
                # This iterates only over unique occupied cells (not every empty cell).
                unique_idx, inverse = torch.unique(linear_idx, return_inverse=True)
                # unique_idx: (K,), K = number of occupied cells
                # We'll compute per-unique-group max
                K = unique_idx.shape[0]
                C = self.out_channels
                # collect max per group
                max_per_group = []
                for k in range(K):
                    mask_k = (inverse == k)
                    group_feats = feat[mask_k]  # nk x C
                    # max along dim 0 -> (C,)
                    max_k = group_feats.max(dim=0)[0]
                    max_per_group.append(max_k)
                # stack into (C, K)
                if K > 0:
                    stacked = torch.stack(max_per_group, dim=1)  # C x K
                else:
                    stacked = torch.empty(C, 0, device=device)
                # build final flat bev of shape (C, num_cells)
                # initialize zeros for all cells (these are the empty ones)
                bev_flat = torch.zeros(C, num_cells, device=device)
                # place the stacked values into the correct columns
                # unique_idx is indices of columns that are occupied
                bev_flat[:, unique_idx.long()] = stacked
                bev = bev_flat.view(C, self.ny, self.nx)
                batch.append(bev)

        out = torch.stack(batch, dim=0)  # B x C x ny x nx
        return out

# --- Backbone (simple ResNet-like conv) --------------------------------------

class SimpleBackbone(nn.Module):
    """
    A small convolutional backbone operating on BEV pseudo-image.
    Depth and width are configurable.
    """
    def __init__(self, in_channels=64, channels=[64, 128, 256], downsample=[2,2,2]):
        super().__init__()
        layers = []
        cur = in_channels
        for idx, ch in enumerate(channels):
            layers.append(nn.Conv2d(cur, ch, kernel_size=3, stride=downsample[idx], padding=1, bias=False))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(inplace=True))
            # add a couple of residual-ish convs
            layers.append(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(inplace=True))
            cur = ch
        self.net = nn.Sequential(*layers)
        self.out_channels = cur

    def forward(self, x):
        return self.net(x)  # B x C x H' x W'

# --- Necks (ASPP, FPN-lite, Plain) -------------------------------------------

class NeckPlain(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.out_channels = out_channels
    def forward(self, x):
        return self.conv(x)

class NeckASPP(nn.Module):
    """
    Simple ASPP (atrous spatial pyramid pooling) adapted to 2D feature maps.
    """
    def __init__(self, in_channels, out_channels, rates=(1,6,12,18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.project = nn.Sequential(
            nn.Conv2d(len(rates)*out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.out_channels = out_channels

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        cat = torch.cat(outs, dim=1)
        return self.project(cat)

class NeckFPNLite(nn.Module):
    """
    Very simplified FPN-like neck for single-stage flow. Accepts single input (we
    don't implement multi-scale backbone outputs here) — uses conv -> upsample as
    a simple fusion to emulate feature pyramid behavior.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.out_channels = out_channels

    def forward(self, x):
        r = self.reduce(x)
        return self.upsample(r)

# --- Detection heads ---------------------------------------------------------

class CenterHead(nn.Module):
    """
    A center-based detection head (in the spirit of CenterPoint/CenterNet).
    Predicts heatmap, size, z, orientation, and class.
    """
    def __init__(self, in_channels, num_classes=3, feat_channels=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        # outputs
        self.hm = nn.Conv2d(feat_channels, num_classes, 1)   # heatmap per class
        self.wh = nn.Conv2d(feat_channels, 2, 1)             # box sizes (dx, dy)
        self.z = nn.Conv2d(feat_channels, 1, 1)              # z center
        self.h = nn.Conv2d(feat_channels, 1, 1)              # dz
        self.reg = nn.Conv2d(feat_channels, 2, 1)            # local offset
        self.rot = nn.Conv2d(feat_channels, 2, 1)            # sin/cos of yaw
        self.num_classes = num_classes

    def forward(self, x):
        f = self.conv(x)
        hm = torch.sigmoid(self.hm(f))
        wh = F.relu(self.wh(f))
        z = self.z(f)
        dz = F.relu(self.h(f))
        reg = self.reg(f)
        rot = self.rot(f)
        out = {
            'hm': hm, 'wh': wh, 'z': z, 'dz': dz, 'reg': reg, 'rot': rot
        }
        return out

class AnchorHead(nn.Module):
    """
    A simple anchor-head that predicts objectness and regressions per anchor location.
    This is intentionally minimal to allow easy experimentation.
    """
    def __init__(self, in_channels, num_anchors=2, num_classes=3, feat_channels=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(feat_channels, num_anchors * num_classes, 1)
        self.reg = nn.Conv2d(feat_channels, num_anchors * 7, 1)  # dx,dy,dz,w,l,h,yaw
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, x):
        f = self.conv(x)
        cls_logits = self.cls(f)
        reg = self.reg(f)
        return {'cls': cls_logits, 'reg': reg}

# --- Losses (very simple placeholders) ---------------------------------------

def focal_loss(pred, gt):
    # placeholder simple focal-like loss for heatmap
    return F.binary_cross_entropy(pred, gt)

def l1_loss(pred, gt, mask=None):
    if mask is None:
        return F.l1_loss(pred, gt)
    else:
        return (F.l1_loss(pred * mask, gt * mask, reduction='sum') / (mask.sum() + 1e-6))

# --- Simple trainer & evaluator ----------------------------------------------

class DetectorModel(nn.Module):
    def __init__(self, encoder, backbone, neck, head):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, points_list):
        # encoder expects list of numpy arrays
        x = self.encoder(points_list)    # B x C x H x W
        x = self.backbone(x)             # B x C' x H' x W'
        x = self.neck(x)                 # B x C'' x H'' x W''
        out = self.head(x)
        return out

def train_one_epoch(model: DetectorModel, dataloader: DataLoader, optimizer, device, epoch, cfg):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(dataloader):
        # batch is list of samples
        points = [b['points'] for b in batch]
        # forward
        out = model(points)
        # Build synthetic targets if using synthetic dataset
        # For real dataset, you'd build heatmaps, regression targets here.
        # Here compute a toy loss: L2 on random target maps so pipeline runs.
        device = next(model.parameters()).device
        # generate dummy targets of same spatial shape as outputs
        if 'hm' in out:
            hm = out['hm']
            target_hm = torch.rand_like(hm).to(device)
            loss = F.mse_loss(hm, target_hm)
            # add other preds for more stable training
            if 'wh' in out:
                target_wh = torch.rand_like(out['wh']).to(device)
                loss = loss + 0.1 * F.mse_loss(out['wh'], target_wh)
        else:
            # anchor head
            cls = out['cls']
            reg = out['reg']
            loss = 0.1 * F.mse_loss(cls, torch.rand_like(cls).to(device)) + 0.1 * F.mse_loss(reg, torch.rand_like(reg).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i+1) % 10 == 0:
            print(f"Epoch {epoch} iter {i+1}/{len(dataloader)} loss={loss.item():.4f}")
    return total_loss / len(dataloader)

def run_inference(model: DetectorModel, dataloader: DataLoader, device, max_batches=5):
    model.eval()
    results = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            points = [b['points'] for b in batch]
            out = model(points)
            # Here we do a super simple dummy decoding:
            if 'hm' in out:
                hm = out['hm'].cpu().numpy()
                # pick top-k peaks per class
                b, c, h, w = hm.shape
                topk = 5
                for bi in range(b):
                    picks = []
                    for cls in range(c):
                        flat = hm[bi, cls].ravel()
                        idxs = np.argpartition(-flat, topk)[:topk]
                        ys = idxs // w; xs = idxs % w
                        for y, x in zip(ys, xs):
                            picks.append({'class': cls, 'x_idx': int(x), 'y_idx': int(y)})
                    results.append({'batch_idx': i, 'picks': picks})
            else:
                results.append({'batch_idx': i, 'note': 'anchor head inference placeholder'})
    return results

# --- Dataset download / prepare helpers (instructions & checks) --------------

def check_nuscenes_data(data_root: str) -> bool:
    """
    Check for a nuScenes dataset at data_root. If not found, print instructions.
    """
    # The user should install nuscenes-devkit and download dataset from nuScenes.
    # We only check for expected folders.
    expected = ['v1.0-mini', 'v1.0-trainval']
    for name in expected:
        if os.path.isdir(os.path.join(data_root, name)):
            print(f"Found nuScenes dataset at {os.path.join(data_root, name)}")
            return True
    print("nuScenes dataset not found under data_root. To use nuScenes:")
    print(" 1) pip install nuscenes-devkit")
    print(" 2) Register at https://www.nuscenes.org/ and download v1.0-mini (small) or v1.0-trainval.")
    print(" 3) Place the dataset dir under --data_root, e.g. /data/nuscenes/v1.0-mini")
    return False

def check_waymo_data(data_root: str) -> bool:
    """
    Check for Waymo dataset paths. Waymo requires registration and special tooling.
    """
    # Waymo open dataset requires special TFRecord files or converted formats.
    cand = os.path.join(data_root, 'waymo_open_dataset')
    if os.path.exists(cand):
        print(f"Found Waymo data at {cand}")
        return True
    print("Waymo dataset not found under data_root. To use Waymo:")
    print(" 1) Register and follow Waymo Open Dataset instructions (https://waymo.com/open/download)")
    print(" 2) Download tfrecord files and use the official reader or convert to pytorch-friendly format.")
    return False

# --- Argparsing / main -------------------------------------------------------

def build_model(cfg, device):
    encoder = PillarEncoder(grid_size=cfg['grid_size'], out_channels=cfg['encoder_out']).to(device)
    backbone = SimpleBackbone(in_channels=cfg['encoder_out'], channels=cfg['backbone_channels']).to(device)
    # choose neck
    neck_type = cfg['neck']
    neck_out = cfg['neck_out_channels']
    if neck_type == 'aspp':
        neck = NeckASPP(backbone.out_channels, neck_out).to(device)
    elif neck_type == 'fpn':
        neck = NeckFPNLite(backbone.out_channels, neck_out).to(device)
    else:
        neck = NeckPlain(backbone.out_channels, neck_out).to(device)
    # choose head
    if cfg['head'] == 'center':
        head = CenterHead(neck.out_channels, num_classes=cfg['num_classes']).to(device)
    else:
        head = AnchorHead(neck.out_channels, num_anchors=cfg['num_anchors'], num_classes=cfg['num_classes']).to(device)
    model = DetectorModel(encoder, backbone, neck, head)
    return model

def main():
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'infer'], default='train')
    parser.add_argument('--dataset', choices=['synthetic', 'nuscenes', 'waymo'], default='synthetic')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--neck', choices=['plain','aspp','fpn'], default='aspp')
    parser.add_argument('--head', choices=['center','anchor'], default='center')
    parser.add_argument('--grid_size', type=float, default=0.15)
    parser.add_argument('--encoder_out', type=int, default=64)
    parser.add_argument('--backbone_channels', nargs='+', type=int, default=[64,128,256])
    parser.add_argument('--neck_out_channels', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    opt = parser.parse_args()

    mkdir(opt.save_dir)

    cfg = {
        'grid_size': opt.grid_size,
        'encoder_out': opt.encoder_out,
        'backbone_channels': opt.backbone_channels,
        'neck': opt.neck,
        'neck_out_channels': opt.neck_out_channels,
        'head': opt.head,
        'num_classes': opt.num_classes,
        'num_anchors': 2
    }

    device = torch.device(opt.device)
    print("Building model. Neck:", opt.neck, "Head:", opt.head, "Device:", device)
    model = build_model(cfg, device)
    print(model)

    # dataset selection
    if opt.dataset == 'synthetic':
        dataset = SyntheticPointCloudDataset(n_samples=256, points_per_sample=2048)
    elif opt.dataset == 'nuscenes':
        if not check_nuscenes_data(opt.data_root):
            print("Falling back to synthetic dataset for now.")
            dataset = SyntheticPointCloudDataset(n_samples=256, points_per_sample=2048)
        else:
            # TODO: provide a wrapper using nuscenes-devkit to create dataloader
            raise NotImplementedError("nuScenes loader not implemented in this script. See header instructions.")
    elif opt.dataset == 'waymo':
        if not check_waymo_data(opt.data_root):
            print("Falling back to synthetic dataset for now.")
            dataset = SyntheticPointCloudDataset(n_samples=256, points_per_sample=2048)
        else:
            # TODO: implement Waymo reader (requires TFRecords conversion)
            raise NotImplementedError("Waymo loader not implemented in this script. See header instructions.")
    else:
        raise ValueError("Unknown dataset")

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    if opt.mode == 'train':
        for epoch in range(1, opt.epochs + 1):
            avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch, cfg)
            print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")
            # save checkpoint
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'cfg': cfg
            }
            torch.save(ckpt, os.path.join(opt.save_dir, f'ckpt_epoch_{epoch}.pth'))
        print("Training complete.")
    elif opt.mode == 'infer':
        results = run_inference(model, dataloader, device)
        print("Inference results (sample):")
        for r in results[:10]:
            print(r)
    else:
        raise ValueError("Unknown mode")

if __name__ == '__main__':
    main()
