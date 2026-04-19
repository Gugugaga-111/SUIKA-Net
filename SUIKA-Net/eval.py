#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from datasets import AnimeCharacterDataset, build_eval_transforms
from models import AnimeNet, PrototypeBank
from utils.io import load_yaml, resolve_device
from utils.metrics import accuracy_topk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate anime character recognition network.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--tau-prob", type=float, default=0.5, help="Open-set probability threshold.")
    parser.add_argument("--tau-sim", type=float, default=0.35, help="Open-set prototype similarity threshold.")
    parser.add_argument("--disable-open-set", action="store_true")
    parser.add_argument("--save-preds", type=str, default=None)
    return parser.parse_args()


def resolve_path(root: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(root, p)


def get_stage_params(cfg: dict) -> Dict[str, object]:
    stage_name = cfg["training"]["stage"]
    stage_cfg = cfg["stages"][stage_name]
    return {
        "stage": stage_name,
        "views": list(stage_cfg.get("views", ["global"])),
        "use_prototype": bool(stage_cfg.get("use_prototype", True)),
    }


@torch.no_grad()
def run_eval(
    model: AnimeNet,
    proto_bank: Optional[PrototypeBank],
    loader: DataLoader,
    device: torch.device,
    views,
    tau_prob: float,
    tau_sim: float,
    disable_open_set: bool,
):
    model.eval()

    total = 0
    top1_sum = 0.0
    top5_sum = 0.0
    open_correct = 0
    unknown_count = 0
    rows = []

    for batch in loader:
        xg = batch["global"].to(device, non_blocking=True)
        xh = batch["head"].to(device, non_blocking=True) if "head" in views else None
        xm = batch["mask"].to(device, non_blocking=True) if "mask" in views else None
        y = batch["label"].to(device, non_blocking=True)

        out = model(xg, xh, xm)
        logits = out["logits"]
        prob = torch.softmax(logits, dim=-1)
        max_prob, pred = prob.max(dim=-1)

        bs = y.size(0)
        total += bs
        b_top1, b_top5 = accuracy_topk(logits, y, topk=(1, min(5, logits.size(1))))
        top1_sum += b_top1 * bs / 100.0
        top5_sum += b_top5 * bs / 100.0

        sim = None
        max_sim = None
        if proto_bank is not None:
            sim = proto_bank.similarity(out["z_fuse"])
            max_sim = sim.max(dim=-1).values

        pred_open = pred.clone()
        if not disable_open_set and max_sim is not None:
            unknown = (max_prob < tau_prob) | (max_sim < tau_sim)
            pred_open[unknown] = -1
            unknown_count += int(unknown.sum().item())

        open_correct += int((pred_open == y).sum().item())

        for i in range(bs):
            rows.append(
                {
                    "file_path": batch["file_path"][i],
                    "label": int(y[i].item()),
                    "pred": int(pred[i].item()),
                    "pred_open": int(pred_open[i].item()),
                    "conf": float(max_prob[i].item()),
                    "sim": float(max_sim[i].item()) if max_sim is not None else "",
                }
            )

    closed_top1 = 100.0 * top1_sum / max(total, 1)
    closed_top5 = 100.0 * top5_sum / max(total, 1)
    open_acc = 100.0 * open_correct / max(total, 1)
    unknown_rate = 100.0 * unknown_count / max(total, 1)
    metrics = {
        "num_samples": total,
        "closed_set_top1": closed_top1,
        "closed_set_top5": closed_top5,
        "open_set_accuracy": open_acc,
        "open_set_unknown_rate": unknown_rate,
    }
    return metrics, rows


def main() -> None:
    args = parse_args()
    repo_root = os.path.abspath(os.path.dirname(__file__))
    cfg = load_yaml(resolve_path(repo_root, args.config))
    if args.device:
        cfg["device"] = args.device
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    device = resolve_device(cfg.get("device", "auto"))
    stage_params = get_stage_params(cfg)

    data_root = resolve_path(repo_root, cfg["data"]["root"])
    csv_file = resolve_path(repo_root, cfg["data"]["csv_file"])
    head_box_file = resolve_path(repo_root, cfg["data"]["head_box_file"])
    mask_root = resolve_path(repo_root, cfg["data"]["mask_root"])

    tg, th, tm = build_eval_transforms(img_size=int(cfg["model"]["img_size"]))
    dataset = AnimeCharacterDataset(
        csv_file=csv_file,
        root=data_root,
        split=args.split,
        head_box_file=head_box_file,
        mask_root=mask_root,
        require_head_box=bool(cfg["data"].get("require_head_box", True)),
        require_mask=bool(cfg["data"].get("require_mask", True)),
        transform_global=tg,
        transform_head=th,
        transform_mask=tm,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        drop_last=False,
        persistent_workers=int(cfg["data"]["num_workers"]) > 0,
    )

    num_classes = max(dataset.labels) + 1
    model = AnimeNet(
        model_name=cfg["model"]["name"],
        num_classes=num_classes,
        emb_dim=int(cfg["model"]["emb_dim"]),
        pretrained=bool(cfg["model"].get("pretrained", True)),
        proj_hidden_dim=cfg["model"].get("proj_hidden_dim"),
        drop_rate=float(cfg["model"].get("drop_rate", 0.0)),
        use_mixstyle=bool(cfg["model"].get("use_mixstyle", False)),
        mixstyle_p=float(cfg["model"].get("mixstyle_p", 0.5)),
        mixstyle_alpha=float(cfg["model"].get("mixstyle_alpha", 0.3)),
        view_weights=cfg["model"].get("view_weights"),
    ).to(device)

    checkpoint_path = resolve_path(repo_root, args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    proto_bank = None
    if stage_params["use_prototype"]:
        proto_bank = PrototypeBank(
            num_classes=num_classes,
            emb_dim=int(cfg["model"]["emb_dim"]),
            momentum=float(cfg["loss"]["prototype_momentum"]),
        ).to(device)
        if "prototype_bank" in ckpt:
            proto_bank.load_state_dict(ckpt["prototype_bank"])

    metrics, rows = run_eval(
        model=model,
        proto_bank=proto_bank,
        loader=loader,
        device=device,
        views=stage_params["views"],
        tau_prob=float(args.tau_prob),
        tau_sim=float(args.tau_sim),
        disable_open_set=bool(args.disable_open_set),
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.save_preds:
        save_path = resolve_path(repo_root, args.save_preds)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["file_path", "label", "pred", "pred_open", "conf", "sim"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[eval] saved predictions to {save_path}")


if __name__ == "__main__":
    main()
