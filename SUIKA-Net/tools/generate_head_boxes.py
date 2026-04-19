#!/usr/bin/env python3
import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


@dataclass
class HeadBoxResult:
    box: Tuple[int, int, int, int]
    conf: float
    source_weight: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strict head boxes for all images in meta.csv.")
    parser.add_argument("--csv-file", type=str, default="data/meta.csv")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--out-json", type=str, default="data/head_boxes.json")
    parser.add_argument(
        "--weights",
        type=str,
        default=(
            "weights/head_detector/yolov8x.pt,"
            "weights/head_detector/yolov8l.pt,"
            "weights/head_detector/yolov8m.pt,"
            "weights/head_detector/yolov8s.pt,"
            "weights/head_detector/yolov8n.pt"
        ),
        help="Comma-separated YOLO face detector weights.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--imgsz-list", type=str, default="640,960")
    parser.add_argument("--conf-thres", type=float, default=0.15)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--expand-ratio", type=float, default=1.25)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--tta-flip", action="store_true", default=True)
    parser.add_argument("--no-tta-flip", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--save-report", type=str, default="data/head_boxes_report.json")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_path(root: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(root, p)


def load_image_paths(csv_file: str, max_images: Optional[int]) -> List[str]:
    out: List[str] = []
    seen = set()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row["file_path"]
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
            if max_images is not None and len(out) >= max_images:
                break
    if not out:
        raise RuntimeError(f"No images found in {csv_file}")
    return out


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda:0"
    raise RuntimeError("CUDA is unavailable. Head detection is expected to run on CUDA.")


def clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1_i = max(0, min(w - 1, int(round(x1))))
    y1_i = max(0, min(h - 1, int(round(y1))))
    x2_i = max(1, min(w, int(round(x2))))
    y2_i = max(1, min(h, int(round(y2))))
    if x2_i <= x1_i:
        x2_i = min(w, x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(h, y1_i + 1)
    return x1_i, y1_i, x2_i, y2_i


def expand_box(
    box: Sequence[float],
    w: int,
    h: int,
    ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1) * ratio
    bh = (y2 - y1) * ratio
    nx1 = cx - bw * 0.5
    ny1 = cy - bh * 0.5
    nx2 = cx + bw * 0.5
    ny2 = cy + bh * 0.5
    return clip_box(nx1, ny1, nx2, ny2, w=w, h=h)


def pick_best_box(
    boxes_xyxy: np.ndarray,
    boxes_conf: np.ndarray,
    img_w: int,
    img_h: int,
    expand_ratio: float,
) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
    if boxes_xyxy.size == 0:
        return None
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    score = boxes_conf * np.power(np.clip(areas, 1.0, None), 0.25)
    idx = int(np.argmax(score))
    box = expand_box(boxes_xyxy[idx], w=img_w, h=img_h, ratio=expand_ratio)
    conf = float(boxes_conf[idx])
    return box, conf


def detect_single_image(
    abs_image_path: str,
    models: Sequence[Tuple[str, YOLO]],
    device: str,
    imgsz_list: Sequence[int],
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    expand_ratio: float,
    tta_flip: bool,
) -> Optional[HeadBoxResult]:
    with Image.open(abs_image_path).convert("RGB") as img:
        img_w, img_h = img.size
        img_np = np.asarray(img)

    candidates: List[HeadBoxResult] = []
    for weight_path, model in models:
        for imgsz in imgsz_list:
            result = model.predict(
                source=abs_image_path,
                device=device,
                imgsz=imgsz,
                conf=conf_thres,
                iou=iou_thres,
                max_det=max_det,
                verbose=False,
            )[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
                boxes_conf = result.boxes.conf.detach().cpu().numpy()
                picked = pick_best_box(
                    boxes_xyxy=boxes_xyxy,
                    boxes_conf=boxes_conf,
                    img_w=img_w,
                    img_h=img_h,
                    expand_ratio=expand_ratio,
                )
                if picked is not None:
                    box, conf = picked
                    candidates.append(HeadBoxResult(box=box, conf=conf, source_weight=weight_path))

            if tta_flip:
                flipped = img_np[:, ::-1, :].copy()
                result_flip = model.predict(
                    source=flipped,
                    device=device,
                    imgsz=imgsz,
                    conf=conf_thres,
                    iou=iou_thres,
                    max_det=max_det,
                    verbose=False,
                )[0]
                if result_flip.boxes is not None and len(result_flip.boxes) > 0:
                    boxes_xyxy = result_flip.boxes.xyxy.detach().cpu().numpy()
                    boxes_conf = result_flip.boxes.conf.detach().cpu().numpy()
                    # map flipped boxes back to original coordinate system
                    boxes_back = boxes_xyxy.copy()
                    boxes_back[:, 0] = img_w - boxes_xyxy[:, 2]
                    boxes_back[:, 2] = img_w - boxes_xyxy[:, 0]
                    picked_flip = pick_best_box(
                        boxes_xyxy=boxes_back,
                        boxes_conf=boxes_conf,
                        img_w=img_w,
                        img_h=img_h,
                        expand_ratio=expand_ratio,
                    )
                    if picked_flip is not None:
                        box, conf = picked_flip
                        candidates.append(HeadBoxResult(box=box, conf=conf, source_weight=weight_path))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x.conf, reverse=True)
    return candidates[0]


def main() -> None:
    args = parse_args()
    strict = args.strict and (not args.allow_missing)
    root_abs = os.path.abspath(args.root)
    csv_file_abs = os.path.abspath(args.csv_file)
    out_json_abs = os.path.abspath(args.out_json)
    report_abs = os.path.abspath(args.save_report)

    ensure_dir(os.path.dirname(out_json_abs))
    ensure_dir(os.path.dirname(report_abs))

    weight_paths = [os.path.abspath(w.strip()) for w in args.weights.split(",") if w.strip()]
    if not weight_paths:
        raise RuntimeError("No detector weights provided.")
    for w in weight_paths:
        if not os.path.exists(w):
            raise FileNotFoundError(f"Detector weight not found: {w}")

    device = choose_device(args.device)
    tta_flip = bool(args.tta_flip) and (not bool(args.no_tta_flip))
    imgsz_list = [int(x.strip()) for x in args.imgsz_list.split(",") if x.strip()]
    if not imgsz_list:
        raise RuntimeError("imgsz_list is empty")
    print(f"[setup] device={device}")
    print(f"[setup] imgsz_list={imgsz_list}, tta_flip={tta_flip}")
    print(f"[setup] loading {len(weight_paths)} detector weights...")

    models = [(w, YOLO(w)) for w in weight_paths]
    image_paths = load_image_paths(csv_file_abs, max_images=args.max_images)
    print(f"[setup] images={len(image_paths)}")

    existing: Dict[str, object] = {}
    if os.path.exists(out_json_abs) and (not args.overwrite):
        with open(out_json_abs, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            existing = raw

    out: Dict[str, object] = dict(existing)
    missing: List[str] = []
    detector_usage: Dict[str, int] = {os.path.basename(w): 0 for w in weight_paths}

    for idx, rel_path in enumerate(image_paths, start=1):
        if rel_path in out and (not args.overwrite):
            continue

        abs_path = resolve_path(root_abs, rel_path)
        if not os.path.exists(abs_path):
            missing.append(rel_path)
            continue

        result = detect_single_image(
            abs_image_path=abs_path,
            models=models,
            device=device,
            imgsz_list=imgsz_list,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            max_det=args.max_det,
            expand_ratio=args.expand_ratio,
            tta_flip=tta_flip,
        )
        if result is None:
            missing.append(rel_path)
        else:
            box = result.box
            out[rel_path] = {
                "x1": box[0],
                "y1": box[1],
                "x2": box[2],
                "y2": box[3],
                "conf": round(float(result.conf), 6),
                "source_weight": os.path.basename(result.source_weight),
            }
            detector_usage[os.path.basename(result.source_weight)] += 1

        if idx % 50 == 0 or idx == len(image_paths):
            print(
                f"[progress] {idx}/{len(image_paths)} "
                f"done, detected={len(out)}, missing={len(missing)}"
            )

    with open(out_json_abs, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    report = {
        "total_images": len(image_paths),
        "detected": len(out),
        "missing": len(missing),
        "strict": strict,
        "detector_usage": detector_usage,
        "missing_examples": missing[:50],
    }
    with open(report_abs, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[done] head boxes saved: {out_json_abs}")
    print(f"[done] report saved: {report_abs}")
    if missing:
        print(f"[warn] missing detections: {len(missing)}")
        if strict:
            raise RuntimeError(
                f"Strict mode enabled and {len(missing)} images have no detected head box. "
                f"See {report_abs}"
            )


if __name__ == "__main__":
    main()
