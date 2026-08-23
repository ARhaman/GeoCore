"""Pretrained single-view control for the registered mineral segmentation study.

This experiment changes only the encoder from the v2 compact U-Net to an
ImageNet-pretrained ResNet-34. It preserves v2's source-level normalized manifest,
512x640 resizing, 448x448 mineral-centred crops, class-weighted CE plus Dice loss,
augmentations, optimizer, scheduler, seed, and evaluation metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision import models

NAMES = ["background", "olivine", "pyroxene", "plagioclase", "alkali_feldspar", "quartz", "biotite", "muscovite", "hornblende"]
C = len(NAMES)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def load_weights(path: str) -> torch.Tensor:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and isinstance(payload.get("ce_weights"), dict):
        values = [payload["ce_weights"][name] for name in NAMES]
    elif isinstance(payload, dict) and isinstance(payload.get("weights"), list):
        values = payload["weights"]
    else:
        raise ValueError("Weights must be `ce_weights` by class name or an explicit ordered `weights` list.")
    if len(values) != C:
        raise ValueError(f"Expected {C} weights, found {len(values)}")
    values = [float(v) for v in values]
    if all(abs(v - 1.0) < 1e-6 for v in values):
        raise ValueError("All-one weights are forbidden for this rare-mineral control.")
    return torch.tensor(values, dtype=torch.float32)


class DS(Dataset):
    """V2 input protocol with the only change being inverse-frequency mineral crop centres."""
    def __init__(self, rows, size, train, seedv, sampling_weights, rare_crop_probability):
        self.rows = rows; self.w, self.h = size; self.train = train; self.seed = seedv
        self.sampling_weights = np.asarray(sampling_weights, dtype=np.float64)
        self.rare_crop_probability = rare_crop_probability
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r["image_path"]) as im: im = im.convert("RGB")
        with Image.open(r["mask_path"]) as ma: ma = ma.convert("L")
        im = im.resize((self.w, self.h), Image.Resampling.BILINEAR)
        ma = ma.resize((self.w, self.h), Image.Resampling.NEAREST)
        rng = random.Random(self.seed + i)
        if self.train:
            if rng.random() < .5: im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT); ma = ma.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random() < .35: im = ImageEnhance.Contrast(im).enhance(.75 + .5 * rng.random())
            cw, ch = min(448, self.w), min(448, self.h)
            mask_arr = np.asarray(ma)
            present = np.unique(mask_arr[(mask_arr > 0) & (mask_arr < C)])
            if len(present) and rng.random() < self.rare_crop_probability:
                probabilities = self.sampling_weights[present]
                selected = int(rng.choices(list(present), weights=list(probabilities), k=1)[0])
                yy, xx = np.argwhere(mask_arr == selected)[rng.randrange(int((mask_arr == selected).sum()))]
                x = max(0, min(self.w - cw, int(xx) - cw // 2)); y = max(0, min(self.h - ch, int(yy) - ch // 2))
            else:
                x = rng.randint(0, self.w - cw); y = rng.randint(0, self.h - ch)
            im = im.crop((x, y, x + cw, y + ch)); ma = ma.crop((x, y, x + cw, y + ch))
        x = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        x = (x - MEAN.squeeze(0)) / STD.squeeze(0)
        y = torch.from_numpy(np.asarray(ma, dtype=np.int64))
        return x, y, r["source_id"]


def dl(rows, args, train, sampling_weights):
    return DataLoader(DS(rows, (args.width, args.height), train, args.seed, sampling_weights, args.rare_crop_probability), args.batch_size, shuffle=train, num_workers=0, pin_memory=True)


class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True), nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
    def forward(self, x): return self.layers(x)


class PretrainedResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.pretrained = True
        except Exception as exc:
            print(f"Warning: pretrained weights unavailable ({exc}); falling back to random ResNet-34.")
            resnet = models.resnet34(weights=None)
            self.pretrained = False
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.e1, self.e2, self.e3, self.e4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.d3 = Block(512 + 256, 256); self.d2 = Block(256 + 128, 128); self.d1 = Block(128 + 64, 64)
        self.head = nn.Sequential(Block(64, 64), nn.Conv2d(64, C, 1))
    def forward(self, x):
        h, w = x.shape[-2:]
        z = self.stem(x); a = self.e1(z); b = self.e2(a); c = self.e3(b); d = self.e4(c)
        z = F.interpolate(d, size=c.shape[-2:], mode="bilinear", align_corners=False); z = self.d3(torch.cat([z, c], 1))
        z = F.interpolate(z, size=b.shape[-2:], mode="bilinear", align_corners=False); z = self.d2(torch.cat([z, b], 1))
        z = F.interpolate(z, size=a.shape[-2:], mode="bilinear", align_corners=False); z = self.d1(torch.cat([z, a], 1))
        z = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(z)


def loss_fn(logits, y, w):
    ce = F.cross_entropy(logits, y, weight=w)
    p = logits.softmax(1); oh = F.one_hot(y, C).permute(0, 3, 1, 2).float(); losses = []
    for k in range(1, C):
        a = p[:, k].flatten(1); b = oh[:, k].flatten(1)
        losses.append(1 - (2 * (a * b).sum(1) + 1) / (a.sum(1) + b.sum(1) + 1))
    return ce + torch.stack(losses).mean()


def metrics(model, loader, dev):
    cm = torch.zeros((C, C), dtype=torch.long); model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            pred = model(x.to(dev)).argmax(1).cpu()
            cm += torch.bincount(y.reshape(-1) * C + pred.reshape(-1), minlength=C * C).reshape(C, C)
    iou, dice = [], []
    for k in range(C):
        tp = cm[k, k].float(); union = cm[k].sum() + cm[:, k].sum() - tp; total = cm[k].sum() + cm[:, k].sum()
        iou.append(float(tp / union) if union else 0.0); dice.append(float(2 * tp / total) if total else 0.0)
    return {"per_class_iou": dict(zip(NAMES, iou)), "per_class_dice": dict(zip(NAMES, dice)), "mean_iou_all": float(np.mean(iou)), "mean_iou_minerals": float(np.mean(iou[1:])), "mean_dice_minerals": float(np.mean(dice[1:])), "confusion_matrix": cm.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--weights", required=True); p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=60); p.add_argument("--batch-size", type=int, default=2); p.add_argument("--height", type=int, default=512); p.add_argument("--width", type=int, default=640); p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--seed", type=int, default=2026); p.add_argument("--rare-crop-probability", type=float, default=0.85)
    args = p.parse_args(); seed(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for this controlled pretrained experiment.")
    dev = torch.device("cuda"); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).open(encoding="utf-8", newline="") as f: rows = list(csv.DictReader(f))
    grouped = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    weights = load_weights(args.weights).to(dev)
    sampling_weights = weights.detach().cpu().tolist()
    train, val, test = dl(grouped["train"], args, True, sampling_weights), dl(grouped["val"], args, False, sampling_weights), dl(grouped["test"], args, False, sampling_weights)
    model = PretrainedResUNet().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True); best = -1.0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for x, y, _ in train:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=True): loss = loss_fn(model(x), y, weights)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total += float(loss)
        sched.step(); validation = metrics(model, val, dev)
        record = {"epoch": epoch, "loss": total / max(1, len(train)), "val": validation}; history.append(record); print(json.dumps(record))
        if validation["mean_iou_minerals"] > best:
            best = validation["mean_iou_minerals"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "classes": NAMES, "weights": weights.tolist(), "pretrained": model.pretrained}, out / "best.pt")
    checkpoint = torch.load(out / "best.pt", map_location=dev, weights_only=False); model.load_state_dict(checkpoint["model"])
    final = {"device": "cuda", "architecture": "imagenet_pretrained_resnet34_unet", "pretrained_encoder": bool(checkpoint["pretrained"]), "best_epoch": checkpoint["epoch"], "classes": NAMES, "weights": checkpoint["weights"], "weights_source": str(args.weights), "same_protocol_as": "rare_crop_v2_except_inverse_frequency_crop_centres", "rare_crop_probability": args.rare_crop_probability, "test": metrics(model, test, dev)}
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8"); (out / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8"); print(json.dumps(final, indent=2))


if __name__ == "__main__": main()
