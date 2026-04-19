#!/usr/bin/env python3
import argparse
import csv
import io
import os
import site
import sys
from typing import List, Optional

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strict foreground masks for all images.")
    parser.add_argument("--csv-file", type=str, default="data/meta.csv")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--out-root", type=str, default="data/masks")
    parser.add_argument("--u2net-home", type=str, default="weights/rembg")
    parser.add_argument("--model", type=str, default="u2net")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--post-process-mask", action="store_true", default=True)
    parser.add_argument("--binary-threshold", type=int, default=127)
    parser.add_argument("--report-file", type=str, default="data/masks_report.json")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_path(root: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(root, p)


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
    uniq = []
    seen = set()
    for d in dirs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def patch_ld_library_path(extra_dirs: List[str]) -> None:
    old = os.environ.get("LD_LIBRARY_PATH", "")
    old_parts = [p for p in old.split(":") if p]
    for d in reversed(extra_dirs):
        if d not in old_parts:
            old_parts.insert(0, d)
    os.environ["LD_LIBRARY_PATH"] = ":".join(old_parts)


def maybe_reexec_with_cuda_libs(extra_dirs: List[str], ready_flag: str = "MASKS_LD_READY") -> None:
    if os.environ.get(ready_flag) == "1":
        return

    old = os.environ.get("LD_LIBRARY_PATH", "")
    old_parts = [p for p in old.split(":") if p]
    missing = [d for d in extra_dirs if d not in old_parts]
    if not missing:
        return

    new_parts = old_parts[:]
    for d in reversed(extra_dirs):
        if d not in new_parts:
            new_parts.insert(0, d)

    new_env = os.environ.copy()
    new_env["LD_LIBRARY_PATH"] = ":".join(new_parts)
    new_env[ready_flag] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], new_env)


def load_paths(csv_file: str, max_images: Optional[int]) -> List[str]:
    out: List[str] = []
    seen = set()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = row["file_path"]
            if fp in seen:
                continue
            seen.add(fp)
            out.append(fp)
            if max_images is not None and len(out) >= max_images:
                break
    return out


def save_mask_png(mask_img: Image.Image, out_path: str, threshold: int) -> None:
    mask_l = mask_img.convert("L")
    arr = np.asarray(mask_l)
    arr = (arr >= threshold).astype(np.uint8) * 255
    out_img = Image.fromarray(arr, mode="L")
    ensure_dir(os.path.dirname(out_path))
    out_img.save(out_path)


def main() -> None:
    args = parse_args()
    strict = args.strict and (not args.allow_missing)
    root_abs = os.path.abspath(args.root)
    csv_abs = os.path.abspath(args.csv_file)
    out_root_abs = os.path.abspath(args.out_root)
    report_abs = os.path.abspath(args.report_file)
    u2net_home_abs = os.path.abspath(args.u2net_home)

    ensure_dir(out_root_abs)
    ensure_dir(os.path.dirname(report_abs))
    ensure_dir(u2net_home_abs)

    # Must be set before rembg/onnxruntime import.
    os.environ["U2NET_HOME"] = u2net_home_abs
    nvidia_lib_dirs = list_nvidia_lib_dirs()
    maybe_reexec_with_cuda_libs(nvidia_lib_dirs)
    patch_ld_library_path(nvidia_lib_dirs)

    import json
    import onnxruntime as ort
    from rembg import new_session, remove

    providers = ort.get_available_providers()
    print(f"[setup] onnxruntime providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError("CUDAExecutionProvider is unavailable; strict full pipeline requires CUDA mask inference.")

    session = new_session(args.model, providers=["CUDAExecutionProvider"])
    active_providers = session.inner_session.get_providers()
    print(f"[setup] rembg session ready: model={args.model}, active_providers={active_providers}")
    if "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            "rembg session did not activate CUDAExecutionProvider. "
            "Check CUDA runtime libs and LD_LIBRARY_PATH."
        )

    paths = load_paths(csv_abs, max_images=args.max_images)
    print(f"[setup] images={len(paths)}")

    failed: List[str] = []
    generated = 0
    skipped = 0
    for idx, rel_path in enumerate(paths, start=1):
        src_abs = resolve_path(root_abs, rel_path)
        stem = os.path.splitext(rel_path)[0]
        out_abs = os.path.join(out_root_abs, stem + ".png")

        if os.path.exists(out_abs) and (not args.overwrite):
            skipped += 1
            continue

        try:
            with open(src_abs, "rb") as f:
                img_bytes = f.read()
            mask_bytes = remove(
                img_bytes,
                session=session,
                only_mask=True,
                post_process_mask=bool(args.post_process_mask),
            )
            mask_img = Image.open(io.BytesIO(mask_bytes))
            save_mask_png(mask_img, out_abs, threshold=args.binary_threshold)
            generated += 1
        except Exception:
            failed.append(rel_path)

        if idx % 25 == 0 or idx == len(paths):
            print(
                f"[progress] {idx}/{len(paths)} processed, generated={generated}, "
                f"skipped={skipped}, failed={len(failed)}"
            )

    report = {
        "total_images": len(paths),
        "generated": generated,
        "skipped_existing": skipped,
        "failed": len(failed),
        "strict": strict,
        "failed_examples": failed[:50],
        "out_root": out_root_abs,
        "u2net_home": u2net_home_abs,
    }
    with open(report_abs, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[done] mask report saved: {report_abs}")
    if failed:
        print(f"[warn] failed masks: {len(failed)}")
        if strict:
            raise RuntimeError(
                f"Strict mode enabled and {len(failed)} masks failed to generate. See {report_abs}"
            )


if __name__ == "__main__":
    main()
