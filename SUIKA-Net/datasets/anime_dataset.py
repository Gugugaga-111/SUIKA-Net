import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


class AnimeCharacterDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        root: str,
        split: Optional[str] = None,
        head_box_file: Optional[str] = None,
        mask_root: Optional[str] = None,
        require_head_box: bool = True,
        require_mask: bool = True,
        transform_global=None,
        transform_head=None,
        transform_mask=None,
    ) -> None:
        self.root = root
        self.mask_root = mask_root
        self.require_head_box = require_head_box
        self.require_mask = require_mask
        self.transform_global = transform_global
        self.transform_head = transform_head
        self.transform_mask = transform_mask

        self.samples = self._load_samples(csv_file, split)
        self.labels = [int(s["label"]) for s in self.samples]
        self.label_names = self._build_label_names(self.samples)
        self.class_counts = self._build_class_counts(self.labels)
        self.head_boxes = self._load_head_boxes(head_box_file)

    @staticmethod
    def _load_samples(csv_file: str, split: Optional[str]) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_split = "split" in fieldnames
            for row in reader:
                if split is not None and has_split and row.get("split", "") != split:
                    continue
                samples.append(row)

        if not samples:
            split_msg = f" and split={split}" if split else ""
            raise RuntimeError(f"No samples found in {csv_file}{split_msg}.")
        return samples

    @staticmethod
    def _build_label_names(samples: List[Dict[str, str]]) -> Dict[int, str]:
        label_names: Dict[int, str] = {}
        for s in samples:
            label = int(s["label"])
            name = s.get("label_name", str(label))
            if label not in label_names:
                label_names[label] = name
        return label_names

    @staticmethod
    def _build_class_counts(labels: List[int]) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for y in labels:
            counts[y] = counts.get(y, 0) + 1
        return counts

    @staticmethod
    def _load_head_boxes(head_box_file: Optional[str]) -> Dict[str, Any]:
        if not head_box_file or not os.path.exists(head_box_file):
            return {}
        with open(head_box_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _to_abs(root: str, maybe_rel_path: str) -> str:
        if os.path.isabs(maybe_rel_path):
            return maybe_rel_path
        return os.path.join(root, maybe_rel_path)

    @staticmethod
    def _normalize_key(p: str) -> str:
        return p.replace("\\", "/")

    def _find_head_box(self, file_path: str) -> Optional[Tuple[int, int, int, int]]:
        norm_path = self._normalize_key(file_path)
        base = os.path.basename(norm_path)
        stem = os.path.splitext(base)[0]
        candidates = [norm_path, base, stem]

        for key in candidates:
            if key not in self.head_boxes:
                continue
            raw = self.head_boxes[key]
            if isinstance(raw, dict):
                x1 = _safe_int(raw.get("x1", 0))
                y1 = _safe_int(raw.get("y1", 0))
                x2 = _safe_int(raw.get("x2", 0))
                y2 = _safe_int(raw.get("y2", 0))
            elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
                x1, y1, x2, y2 = (_safe_int(raw[i]) for i in range(4))
            else:
                continue
            return x1, y1, x2, y2
        return None

    def _load_image(self, file_path: str) -> Image.Image:
        path_abs = self._to_abs(self.root, file_path)
        return Image.open(path_abs).convert("RGB")

    def _crop_head(self, img: Image.Image, file_path: str) -> Image.Image:
        w, h = img.size
        box = self._find_head_box(file_path)
        if box is not None:
            x1, y1, x2, y2 = box
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(1, min(w, x2))
            y2 = max(1, min(h, y2))
            if x2 > x1 and y2 > y1:
                return img.crop((x1, y1, x2, y2))

        if self.require_head_box:
            raise KeyError(f"Missing or invalid head box for {file_path}")

        # Optional non-strict mode fallback.
        fx1 = int(0.20 * w)
        fy1 = int(0.00 * h)
        fx2 = int(0.80 * w)
        fy2 = int(0.60 * h)
        return img.crop((fx1, fy1, fx2, fy2))

    def _load_mask(self, img: Image.Image, sample: Dict[str, str]) -> Optional[Image.Image]:
        if not self.mask_root:
            if self.require_mask:
                raise FileNotFoundError("mask_root is required but not provided")
            return None

        file_path = sample["file_path"]
        stem = os.path.splitext(file_path)[0]
        candidates = [
            os.path.join(self.mask_root, stem + ".png"),
            os.path.join(self.mask_root, os.path.basename(stem) + ".png"),
        ]
        if sample.get("mask_path"):
            candidates.insert(0, self._to_abs(self.root, sample["mask_path"]))

        for p in candidates:
            p_abs = self._to_abs(self.root, p)
            if os.path.exists(p_abs):
                return Image.open(p_abs).convert("L").resize(img.size, Image.BILINEAR)
        if self.require_mask:
            raise FileNotFoundError(f"Mask not found for {file_path} under {self.mask_root}")
        return None

    @staticmethod
    def _make_mask_prompt(img: Image.Image, mask: Optional[Image.Image]) -> Image.Image:
        if mask is None:
            blurred = img.filter(ImageFilter.GaussianBlur(radius=6))
            return Image.blend(img, blurred, alpha=0.20)

        img_np = np.asarray(img).astype(np.float32)
        mask_np = np.asarray(mask).astype(np.float32) / 255.0
        if mask_np.ndim == 2:
            mask_np = np.expand_dims(mask_np, axis=-1)
        blurred_np = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=8))).astype(np.float32)

        mixed = img_np * mask_np + blurred_np * (1.0 - mask_np)
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)
        return Image.fromarray(mixed)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        file_path = sample["file_path"]
        label = int(sample["label"])
        label_name = sample.get("label_name", str(label))

        img = self._load_image(file_path)
        img_head = self._crop_head(img, file_path)
        mask = self._load_mask(img, sample)
        img_mask = self._make_mask_prompt(img, mask)

        x_global = self.transform_global(img) if self.transform_global else img
        x_head = self.transform_head(img_head) if self.transform_head else img_head
        x_mask = self.transform_mask(img_mask) if self.transform_mask else img_mask

        return {
            "global": x_global,
            "head": x_head,
            "mask": x_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
            "file_path": file_path,
            "label_name": label_name,
        }


def build_train_transforms(
    img_size: int = 224,
    use_randaugment: bool = True,
) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    global_ops: List[Any] = [
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if use_randaugment:
        global_ops.append(transforms.RandAugment())
    global_ops.extend([transforms.ToTensor(), normalize])

    head_ops: List[Any] = [
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        normalize,
    ]

    mask_ops: List[Any] = [
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ]

    return transforms.Compose(global_ops), transforms.Compose(head_ops), transforms.Compose(mask_ops)


def build_eval_transforms(
    img_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    basic = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return basic, basic, basic
