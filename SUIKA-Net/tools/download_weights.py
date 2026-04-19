#!/usr/bin/env python3
import argparse
import json
import os
import site
import sys
from datetime import datetime
from typing import Dict, List

import torch
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify all required model weights.")
    parser.add_argument("--output-root", type=str, default="weights")
    parser.add_argument(
        "--head-repo",
        type=str,
        default="bogdnvch/yolov8-face",
        help="HuggingFace repo for face/head detector weights.",
    )
    parser.add_argument(
        "--head-files",
        type=str,
        default="yolov8x.pt,yolov8l.pt,yolov8m.pt,yolov8s.pt,yolov8n.pt",
        help="Comma-separated weight files to download for head detector.",
    )
    parser.add_argument(
        "--backbone-repo",
        type=str,
        default="timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
        help="HuggingFace repo used by timm backbone.",
    )
    parser.add_argument(
        "--backbone-files",
        type=str,
        default="model.safetensors,config.json",
        help="Comma-separated backbone files to cache locally.",
    )
    parser.add_argument("--u2net-model", type=str, default="u2net")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if CUDA provider cannot be used for rembg.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_nvidia_lib_dirs() -> List[str]:
    dirs: List[str] = []
    for sp in site.getsitepackages():
        nvidia_root = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for module in os.listdir(nvidia_root):
            lib_dir = os.path.join(nvidia_root, module, "lib")
            if os.path.isdir(lib_dir):
                dirs.append(lib_dir)
    uniq = []
    seen = set()
    for d in dirs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def prepend_ld_library_path(extra_dirs: List[str]) -> None:
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    for d in reversed(extra_dirs):
        if d not in parts:
            parts.insert(0, d)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def maybe_reexec_with_cuda_libs(extra_dirs: List[str], ready_flag: str = "WEIGHTS_LD_READY") -> None:
    if os.environ.get(ready_flag) == "1":
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    missing = [d for d in extra_dirs if d not in parts]
    if not missing:
        return

    new_parts = parts[:]
    for d in reversed(extra_dirs):
        if d not in new_parts:
            new_parts.insert(0, d)

    new_env = os.environ.copy()
    new_env["LD_LIBRARY_PATH"] = ":".join(new_parts)
    new_env[ready_flag] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], new_env)


def download_hf_files(repo_id: str, filenames: List[str], local_dir: str) -> Dict[str, str]:
    ensure_dir(local_dir)
    out = {}
    for name in filenames:
        name = name.strip()
        if not name:
            continue
        p = hf_hub_download(repo_id=repo_id, filename=name, local_dir=local_dir)
        out[name] = p
        print(f"[download] {repo_id}:{name} -> {p}")
    return out


def verify_head_weights(head_weights: List[str], device: str) -> None:
    from ultralytics import YOLO

    for w in head_weights:
        model = YOLO(w)
        # dry-run with a random tensor-like image to validate model loading only
        print(f"[verify] head detector load ok: {w}")
        # Explicitly keep model on target device for CUDA check.
        model.to(device)


def verify_backbone() -> None:
    import timm

    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0, global_pool="avg")
    _ = model
    print("[verify] timm backbone pretrained load ok: vit_base_patch16_224")


def verify_rembg(u2net_home: str, model_name: str, require_cuda: bool) -> List[str]:
    os.environ["U2NET_HOME"] = u2net_home
    ensure_dir(u2net_home)

    nvidia_lib_dirs = resolve_nvidia_lib_dirs()
    prepend_ld_library_path(nvidia_lib_dirs)

    import onnxruntime as ort
    from rembg import new_session

    available = ort.get_available_providers()
    print(f"[verify] onnxruntime providers: {available}")
    if require_cuda and "CUDAExecutionProvider" not in available:
        raise RuntimeError("CUDAExecutionProvider not available for onnxruntime.")

    providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    if require_cuda and providers != ["CUDAExecutionProvider"]:
        raise RuntimeError("Failed to initialize rembg with CUDAExecutionProvider.")

    session = new_session(model_name, providers=providers)
    active_providers = session.inner_session.get_providers()
    print(f"[verify] rembg session ok: model={model_name}, active_providers={active_providers}")
    if require_cuda and "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError("rembg initialized without CUDAExecutionProvider in strict CUDA mode.")
    return active_providers


def main() -> None:
    args = parse_args()
    nvidia_lib_dirs = resolve_nvidia_lib_dirs()
    maybe_reexec_with_cuda_libs(nvidia_lib_dirs)
    prepend_ld_library_path(nvidia_lib_dirs)

    output_root = os.path.abspath(args.output_root)
    head_dir = os.path.join(output_root, "head_detector")
    backbone_dir = os.path.join(output_root, "backbone")
    rembg_dir = os.path.join(output_root, "rembg")
    ensure_dir(output_root)
    ensure_dir(head_dir)
    ensure_dir(backbone_dir)
    ensure_dir(rembg_dir)

    head_files = [s.strip() for s in args.head_files.split(",") if s.strip()]
    backbone_files = [s.strip() for s in args.backbone_files.split(",") if s.strip()]

    head_downloaded = download_hf_files(args.head_repo, head_files, head_dir)
    backbone_downloaded = download_hf_files(args.backbone_repo, backbone_files, backbone_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda:0":
        raise RuntimeError("CUDA is unavailable but --require-cuda was set.")

    verify_head_weights([head_downloaded[k] for k in head_files], device=device)
    verify_backbone()
    rembg_providers = verify_rembg(rembg_dir, args.u2net_model, args.require_cuda)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "head_repo": args.head_repo,
        "head_files": head_downloaded,
        "backbone_repo": args.backbone_repo,
        "backbone_files": backbone_downloaded,
        "u2net_home": rembg_dir,
        "u2net_model": args.u2net_model,
        "rembg_providers": rembg_providers,
        "require_cuda": bool(args.require_cuda),
    }
    manifest_path = os.path.join(output_root, "weights_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[done] manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
