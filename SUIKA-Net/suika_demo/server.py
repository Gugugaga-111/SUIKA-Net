#!/usr/bin/env python3
import argparse
import cgi
import csv
import io
import json
import mimetypes
import os
import random
import site
import sys
import time
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_eval_transforms
from models import AnimeNet, PrototypeBank
from utils.io import load_yaml, resolve_device


@dataclass
class HeadBoxResult:
    box: Tuple[int, int, int, int]
    conf: float


def list_nvidia_lib_dirs() -> List[str]:
    dirs: List[str] = []
    for sp in site.getsitepackages():
        nvidia_root = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for name in os.listdir(nvidia_root):
            lib_dir = os.path.join(nvidia_root, name, "lib")
            if os.path.isdir(lib_dir):
                dirs.append(lib_dir)

    out: List[str] = []
    seen = set()
    for d in dirs:
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


def patch_ld_library_path(extra_dirs: Sequence[str]) -> None:
    old = os.environ.get("LD_LIBRARY_PATH", "")
    old_parts = [p for p in old.split(":") if p]
    for d in reversed(extra_dirs):
        if d not in old_parts:
            old_parts.insert(0, d)
    os.environ["LD_LIBRARY_PATH"] = ":".join(old_parts)


def resolve_path(root: Path, maybe_rel_path: str) -> Path:
    p = Path(maybe_rel_path)
    if p.is_absolute():
        return p
    return root / p


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
) -> Optional[HeadBoxResult]:
    if boxes_xyxy.size == 0:
        return None
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    score = boxes_conf * np.power(np.clip(areas, 1.0, None), 0.25)
    idx = int(np.argmax(score))
    box = expand_box(boxes_xyxy[idx], w=img_w, h=img_h, ratio=expand_ratio)
    conf = float(boxes_conf[idx])
    return HeadBoxResult(box=box, conf=conf)


def extract_image_name(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    parts = stem.split("_", 2)
    if len(parts) == 3:
        return parts[2]
    return stem


def extract_pixiv_id(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    parts = stem.split("_", 2)
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[1]
    return ""


class SuikaEngine:
    def __init__(
        self,
        repo_root: Path,
        config_path: Path,
        checkpoint_path: Path,
        device_arg: Optional[str],
        head_weights: Sequence[Path],
        u2net_home: Path,
        u2net_model: str,
        seed: int,
    ) -> None:
        self.repo_root = repo_root
        self.cfg = load_yaml(str(config_path))

        wanted_device = device_arg if device_arg else self.cfg.get("device", "auto")
        self.device = resolve_device(wanted_device)
        if self.device.type != "cuda":
            raise RuntimeError("SUIKA-Net demo requires CUDA and does not fall back to CPU.")
        self.device_str = f"cuda:{self.device.index if self.device.index is not None else torch.cuda.current_device()}"

        stage_name = self.cfg["training"]["stage"]
        stage_cfg = self.cfg["stages"][stage_name]
        self.views: List[str] = list(stage_cfg.get("views", ["global"]))

        data_cfg = self.cfg["data"]
        self.data_root = resolve_path(repo_root, data_cfg["root"]).resolve()
        csv_file = resolve_path(repo_root, data_cfg["csv_file"]).resolve()
        label_map_file = self.data_root / "label_map.json"
        if not label_map_file.exists():
            raise FileNotFoundError(f"label map not found: {label_map_file}")

        self.require_head_box = bool(data_cfg.get("require_head_box", True))
        self.require_mask = bool(data_cfg.get("require_mask", True))

        self.rng = random.Random(seed)

        self.label_map = self._load_label_map(label_map_file)
        self.gallery_by_label = self._load_gallery_index(csv_file)

        img_size = int(self.cfg["model"]["img_size"])
        self.img_size = img_size
        self.transform_global, self.transform_head, self.transform_mask = build_eval_transforms(img_size=img_size)

        self.model = self._build_model(checkpoint_path)
        self.proto_bank = self._build_proto_bank()

        self.detectors = self._build_detectors(head_weights)
        self.expand_ratio = 1.25
        self.det_conf = 0.15
        self.det_iou = 0.5
        self.det_imgsz = 960
        self.det_max_det = 20

        self.mask_session, self.mask_remove = self._build_mask_session(u2net_home, u2net_model)

    def _load_label_map(self, path: Path) -> Dict[int, str]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out: Dict[int, str] = {}
        for k, v in raw.items():
            out[int(k)] = str(v)
        return out

    def _load_gallery_index(self, csv_file: Path) -> Dict[int, List[Dict[str, str]]]:
        out: Dict[int, List[Dict[str, str]]] = {}
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(row["label"])
                out.setdefault(label, []).append(row)
        return out

    def _build_model(self, checkpoint_path: Path) -> AnimeNet:
        num_classes = len(self.label_map)
        model_cfg = self.cfg["model"]
        model = AnimeNet(
            model_name=model_cfg["name"],
            num_classes=num_classes,
            emb_dim=int(model_cfg["emb_dim"]),
            # Checkpoint contains full backbone weights; avoid unnecessary remote download.
            pretrained=False,
            proj_hidden_dim=model_cfg.get("proj_hidden_dim"),
            drop_rate=float(model_cfg.get("drop_rate", 0.0)),
            use_mixstyle=bool(model_cfg.get("use_mixstyle", False)),
            mixstyle_p=float(model_cfg.get("mixstyle_p", 0.5)),
            mixstyle_alpha=float(model_cfg.get("mixstyle_alpha", 0.3)),
            view_weights=model_cfg.get("view_weights"),
        ).to(self.device)

        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self.ckpt = ckpt
        return model

    def _build_proto_bank(self) -> Optional[PrototypeBank]:
        stage_name = self.cfg["training"]["stage"]
        stage_cfg = self.cfg["stages"][stage_name]
        use_prototype = bool(stage_cfg.get("use_prototype", True))
        if not use_prototype:
            return None

        num_classes = len(self.label_map)
        proto = PrototypeBank(
            num_classes=num_classes,
            emb_dim=int(self.cfg["model"]["emb_dim"]),
            momentum=float(self.cfg["loss"].get("prototype_momentum", 0.9)),
        ).to(self.device)
        if "prototype_bank" in self.ckpt:
            proto.load_state_dict(self.ckpt["prototype_bank"])
        proto.eval()
        return proto

    def _build_detectors(self, head_weights: Sequence[Path]) -> List[YOLO]:
        models: List[YOLO] = []
        for weight in head_weights:
            if not weight.exists():
                raise FileNotFoundError(f"head detector weight not found: {weight}")
            models.append(YOLO(str(weight)))
        return models

    def _build_mask_session(self, u2net_home: Path, u2net_model: str):
        os.environ["U2NET_HOME"] = str(u2net_home)
        patch_ld_library_path(list_nvidia_lib_dirs())

        import onnxruntime as ort
        from rembg import new_session, remove

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError("onnxruntime CUDAExecutionProvider is unavailable.")

        session = new_session(u2net_model, providers=["CUDAExecutionProvider"])
        active = session.inner_session.get_providers()
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError("rembg session did not enable CUDAExecutionProvider.")
        return session, remove

    @staticmethod
    def _normalize_round(round_idx: int) -> int:
        return max(0, int(round_idx))

    @staticmethod
    def _build_seeded_rng(seed: int, label_id: int, round_idx: int) -> random.Random:
        mixed = int(seed) + int(label_id) * 9176 + int(round_idx) * 1000003
        return random.Random(mixed)

    @staticmethod
    def _sampling_meta(seed: Optional[int], round_idx: int) -> Dict[str, Any]:
        return {
            "fixed_seed": seed is not None,
            "seed": seed,
            "round": int(round_idx),
        }

    def _detect_head(self, image: Image.Image) -> Image.Image:
        image_np = np.asarray(image)
        h, w = image_np.shape[:2]

        candidates: List[HeadBoxResult] = []
        for det in self.detectors:
            result = det.predict(
                source=image_np,
                device=self.device_str,
                imgsz=self.det_imgsz,
                conf=self.det_conf,
                iou=self.det_iou,
                max_det=self.det_max_det,
                verbose=False,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
            boxes_conf = result.boxes.conf.detach().cpu().numpy()
            picked = pick_best_box(
                boxes_xyxy=boxes_xyxy,
                boxes_conf=boxes_conf,
                img_w=w,
                img_h=h,
                expand_ratio=self.expand_ratio,
            )
            if picked is not None:
                candidates.append(picked)

        if candidates:
            candidates.sort(key=lambda x: x.conf, reverse=True)
            x1, y1, x2, y2 = candidates[0].box
            return image.crop((x1, y1, x2, y2))

        if self.require_head_box:
            raise RuntimeError("Head detector failed to produce a valid box for the query image.")

        fx1 = int(0.20 * w)
        fy1 = int(0.00 * h)
        fx2 = int(0.80 * w)
        fy2 = int(0.60 * h)
        return image.crop((fx1, fy1, fx2, fy2))

    def _infer_mask(self, image: Image.Image) -> Optional[Image.Image]:
        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            mask_bytes = self.mask_remove(
                buf.getvalue(),
                session=self.mask_session,
                only_mask=True,
                post_process_mask=True,
            )
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L").resize(image.size, Image.BILINEAR)
            return mask_img
        except Exception as exc:
            if self.require_mask:
                raise RuntimeError("Mask generation failed for query image.") from exc
            return None

    def _make_mask_prompt(self, image: Image.Image, mask: Optional[Image.Image]) -> Image.Image:
        if mask is None:
            blurred = image.filter(ImageFilter.GaussianBlur(radius=6))
            return Image.blend(image, blurred, alpha=0.20)

        img_np = np.asarray(image).astype(np.float32)
        mask_np = np.asarray(mask).astype(np.float32) / 255.0
        if mask_np.ndim == 2:
            mask_np = np.expand_dims(mask_np, axis=-1)
        blurred_np = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=8))).astype(np.float32)

        mixed = img_np * mask_np + blurred_np * (1.0 - mask_np)
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)
        return Image.fromarray(mixed)

    def _sample_gallery(
        self,
        label_id: int,
        k: int,
        seed: Optional[int] = None,
        round_idx: int = 0,
    ) -> List[Dict[str, str]]:
        pool = self.gallery_by_label.get(label_id, [])
        if not pool:
            return []

        count = min(max(1, k), len(pool))
        if seed is None:
            chosen = self.rng.sample(pool, count)
        else:
            rng = self._build_seeded_rng(seed=seed, label_id=label_id, round_idx=round_idx)
            chosen = rng.sample(pool, count)

        out: List[Dict[str, str]] = []
        for row in chosen:
            rel_path = row["file_path"]
            file_name = os.path.basename(rel_path)
            pixiv_id = row.get("pixiv_id", "") or extract_pixiv_id(file_name)
            out.append(
                {
                    "file_path": rel_path,
                    "file_name": file_name,
                    "image_name": extract_image_name(file_name),
                    "pixiv_id": pixiv_id,
                    "image_url": f"/api/image?path={urllib.parse.quote(rel_path)}",
                }
            )
        return out

    def sample_gallery(
        self,
        label_id: int,
        num_gallery: int,
        seed: Optional[int] = None,
        round_idx: int = 0,
    ) -> Dict[str, Any]:
        if label_id not in self.label_map:
            raise ValueError(f"Unknown label_id: {label_id}")

        started = time.perf_counter()
        round_norm = self._normalize_round(round_idx)
        gallery = self._sample_gallery(label_id=label_id, k=num_gallery, seed=seed, round_idx=round_norm)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return {
            "label_id": label_id,
            "character_name": self.label_map[label_id],
            "gallery": gallery,
            "meta": {
                "latency_ms": elapsed_ms,
                "sampling": self._sampling_meta(seed=seed, round_idx=round_norm),
            },
        }

    @torch.no_grad()
    def predict(
        self,
        image_bytes: bytes,
        num_gallery: int = 8,
        seed: Optional[int] = None,
        round_idx: int = 0,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        head = self._detect_head(image)
        mask = self._infer_mask(image)
        mask_prompt = self._make_mask_prompt(image, mask)

        xg = self.transform_global(image).unsqueeze(0).to(self.device, non_blocking=True)
        xh = self.transform_head(head).unsqueeze(0).to(self.device, non_blocking=True) if "head" in self.views else None
        xm = (
            self.transform_mask(mask_prompt).unsqueeze(0).to(self.device, non_blocking=True)
            if "mask" in self.views
            else None
        )

        with torch.autocast(device_type=self.device.type, enabled=True):
            out = self.model(xg, xh, xm)
            logits = out["logits"]
            prob = torch.softmax(logits, dim=-1)

        conf, pred = prob.max(dim=-1)
        label_id = int(pred.item())
        confidence = float(conf.item())

        topk = min(5, prob.shape[1])
        top_probs, top_idx = torch.topk(prob[0], k=topk)
        top_predictions: List[Dict[str, Any]] = []
        for i in range(topk):
            idx = int(top_idx[i].item())
            top_predictions.append(
                {
                    "label_id": idx,
                    "character_name": self.label_map.get(idx, str(idx)),
                    "confidence": float(top_probs[i].item()),
                }
            )

        proto_similarity = None
        if self.proto_bank is not None:
            sim = self.proto_bank.similarity(out["z_fuse"])
            proto_similarity = float(sim[0, label_id].item())

        round_norm = self._normalize_round(round_idx)
        gallery = self._sample_gallery(label_id=label_id, k=num_gallery, seed=seed, round_idx=round_norm)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "label_id": label_id,
            "character_name": self.label_map.get(label_id, str(label_id)),
            "confidence": confidence,
            "prototype_similarity": proto_similarity,
            "top_predictions": top_predictions,
            "gallery": gallery,
            "meta": {
                "latency_ms": elapsed_ms,
                "device": self.device_str,
                "views": self.views,
                "sampling": self._sampling_meta(seed=seed, round_idx=round_norm),
            },
        }

    @torch.no_grad()
    def warmup(self) -> None:
        print("[warmup] starting GPU warmup...")
        started = time.perf_counter()

        dummy = torch.zeros((1, 3, self.img_size, self.img_size), dtype=torch.float32, device=self.device)
        xh = dummy if "head" in self.views else None
        xm = dummy if "mask" in self.views else None

        with torch.autocast(device_type=self.device.type, enabled=True):
            out = self.model(dummy, xh, xm)
            _ = torch.softmax(out["logits"], dim=-1)

        if self.proto_bank is not None:
            _ = self.proto_bank.similarity(out["z_fuse"])

        warm_np = np.zeros((self.det_imgsz, self.det_imgsz, 3), dtype=np.uint8)
        for det in self.detectors:
            _ = det.predict(
                source=warm_np,
                device=self.device_str,
                imgsz=self.det_imgsz,
                conf=self.det_conf,
                iou=self.det_iou,
                max_det=self.det_max_det,
                verbose=False,
            )

        warm_img = Image.fromarray(np.full((self.img_size, self.img_size, 3), 127, dtype=np.uint8), mode="RGB")
        _ = self._infer_mask(warm_img)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        print(f"[warmup] done in {elapsed_ms:.1f} ms")


class SuikaRequestHandler(BaseHTTPRequestHandler):
    engine: Optional[SuikaEngine] = None
    static_root: Optional[Path] = None
    max_upload_bytes: int = 16 * 1024 * 1024
    max_gallery: int = 100
    default_gallery: int = 10

    @staticmethod
    def _parse_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(text)

    @staticmethod
    def _parse_non_negative_int(value: Any, default: int = 0) -> int:
        if value is None:
            return max(0, int(default))
        text = str(value).strip()
        if not text:
            return max(0, int(default))
        return max(0, int(text))

    @staticmethod
    def _clamp_gallery(num_gallery: int, max_gallery: int, default_gallery: int) -> int:
        if num_gallery <= 0:
            return default_gallery
        return max(1, min(max_gallery, int(num_gallery)))

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _serve_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._send_error(404, "Not found")
            return

        mime, _ = mimetypes.guess_type(str(file_path))
        if not mime:
            mime = "application/octet-stream"

        with open(file_path, "rb") as f:
            data = f.read()

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _resolve_static_path(self, raw_path: str) -> Optional[Path]:
        assert self.static_root is not None
        path = urllib.parse.unquote(raw_path)
        rel = path.lstrip("/")
        candidate = (self.static_root / rel).resolve()
        if not str(candidate).startswith(str(self.static_root.resolve())):
            return None
        return candidate

    def _resolve_dataset_image(self, rel_path: str) -> Optional[Path]:
        assert self.engine is not None
        rel = Path(rel_path)
        if rel.is_absolute() or ".." in rel.parts:
            return None
        candidate = (self.engine.data_root / rel).absolute()
        return candidate

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/health":
            assert self.engine is not None
            self._json(
                200,
                {
                    "ok": True,
                    "device": self.engine.device_str,
                    "views": self.engine.views,
                    "num_classes": len(self.engine.label_map),
                },
            )
            return

        if parsed.path == "/api/gallery":
            assert self.engine is not None
            try:
                qs = urllib.parse.parse_qs(parsed.query)
                label_id = int(qs.get("label_id", [""])[0])
                raw_num_gallery = self._parse_non_negative_int(
                    qs.get("num_gallery", [self.default_gallery])[0],
                    default=self.default_gallery,
                )
                num_gallery = self._clamp_gallery(
                    num_gallery=raw_num_gallery,
                    max_gallery=self.max_gallery,
                    default_gallery=self.default_gallery,
                )
                seed = self._parse_optional_int(qs.get("seed", [None])[0])
                round_idx = self._parse_non_negative_int(qs.get("round", [0])[0], default=0)
                result = self.engine.sample_gallery(
                    label_id=label_id,
                    num_gallery=num_gallery,
                    seed=seed,
                    round_idx=round_idx,
                )
            except ValueError as exc:
                self._send_error(400, str(exc))
                return
            except Exception as exc:
                self._send_error(500, str(exc))
                return

            self._json(200, result)
            return

        if parsed.path == "/api/image":
            qs = urllib.parse.parse_qs(parsed.query)
            rel_path = qs.get("path", [""])[0]
            if not rel_path:
                self._send_error(400, "Missing 'path' query parameter")
                return
            target = self._resolve_dataset_image(rel_path)
            if target is None:
                self._send_error(400, "Invalid path")
                return
            self._serve_file(target)
            return

        if parsed.path in ("/", ""):
            assert self.static_root is not None
            self._serve_file(self.static_root / "index.html")
            return

        target = self._resolve_static_path(parsed.path)
        if target is None:
            self._send_error(400, "Invalid path")
            return
        self._serve_file(target)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/predict":
            self._send_error(404, "Not found")
            return

        assert self.engine is not None

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._send_error(400, "Empty request body")
            return
        if content_length > self.max_upload_bytes + (1 * 1024 * 1024):
            self._send_error(413, f"Request too large (>{self.max_upload_bytes} bytes)")
            return

        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._send_error(400, "Content-Type must be multipart/form-data")
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": ctype,
                "CONTENT_LENGTH": str(content_length),
            },
        )

        image_field = form["image"] if "image" in form else None
        if image_field is None or not getattr(image_field, "file", None):
            self._send_error(400, "Missing image field")
            return

        image_bytes = image_field.file.read()
        if not image_bytes:
            self._send_error(400, "Uploaded image is empty")
            return
        if len(image_bytes) > self.max_upload_bytes:
            self._send_error(413, f"Image exceeds {self.max_upload_bytes} bytes")
            return

        try:
            raw_num_gallery = self._parse_non_negative_int(form.getvalue("num_gallery"), default=self.default_gallery)
            num_gallery = self._clamp_gallery(
                num_gallery=raw_num_gallery,
                max_gallery=self.max_gallery,
                default_gallery=self.default_gallery,
            )
            seed = self._parse_optional_int(form.getvalue("seed"))
            round_idx = self._parse_non_negative_int(form.getvalue("round"), default=0)
        except ValueError as exc:
            self._send_error(400, str(exc))
            return

        try:
            result = self.engine.predict(
                image_bytes=image_bytes,
                num_gallery=num_gallery,
                seed=seed,
                round_idx=round_idx,
            )
        except Exception as exc:
            self._send_error(500, str(exc))
            return

        self._json(200, result)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[http] {self.address_string()} - {fmt % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SUIKA-Net local demo server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    parser.add_argument("--config", type=str, default="configs/tuned_v3_top20x500_testselect.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/stage_c/tuned_v3_top20x500_testselect/best.pt",
    )
    parser.add_argument(
        "--head-weights",
        type=str,
        default="weights/head_detector/yolov8s.pt",
        help="Comma-separated detector weights",
    )
    parser.add_argument("--u2net-home", type=str, default="weights/rembg")
    parser.add_argument("--u2net-model", type=str, default="u2net")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-upload-mb", type=int, default=16)
    parser.add_argument("--default-gallery", type=int, default=10)
    parser.add_argument("--max-gallery", type=int, default=100)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip startup warmup. By default warmup is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    config_path = resolve_path(repo_root, args.config).resolve()
    checkpoint_path = resolve_path(repo_root, args.checkpoint).resolve()
    static_root = (repo_root / "suika_demo").resolve()
    u2net_home = resolve_path(repo_root, args.u2net_home).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not static_root.exists():
        raise FileNotFoundError(f"Static dir not found: {static_root}")

    head_weights: List[Path] = []
    for item in args.head_weights.split(","):
        p = item.strip()
        if not p:
            continue
        head_weights.append(resolve_path(repo_root, p).resolve())
    if not head_weights:
        raise RuntimeError("No head detector weights were provided.")

    print("[startup] initializing SUIKA-Net demo engine...")
    engine = SuikaEngine(
        repo_root=repo_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device_arg=args.device,
        head_weights=head_weights,
        u2net_home=u2net_home,
        u2net_model=args.u2net_model,
        seed=args.seed,
    )
    print(
        "[startup] engine ready: "
        f"device={engine.device_str}, classes={len(engine.label_map)}, views={engine.views}"
    )

    if not args.no_warmup:
        engine.warmup()

    SuikaRequestHandler.engine = engine
    SuikaRequestHandler.static_root = static_root
    SuikaRequestHandler.max_upload_bytes = args.max_upload_mb * 1024 * 1024
    SuikaRequestHandler.default_gallery = max(1, args.default_gallery)
    SuikaRequestHandler.max_gallery = max(1, args.max_gallery)

    server = ThreadingHTTPServer((args.host, args.port), SuikaRequestHandler)
    print(f"[startup] serving http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
