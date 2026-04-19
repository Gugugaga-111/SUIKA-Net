#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Optional

import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from datasets import AnimeCharacterDataset, build_eval_transforms, build_train_transforms
from losses import build_class_weights, compute_loss
from models import AnimeNet, PrototypeBank
from utils.io import ensure_dir, load_yaml, resolve_device, seed_everything
from utils.metrics import AccuracyMeter
from utils.sampler import build_class_balanced_sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anime character recognition network.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--stage", type=str, default=None, choices=["stage_a", "stage_b", "stage_c"])
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run one short epoch for sanity check.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps per epoch.")
    return parser.parse_args()


def deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def resolve_path(root: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(root, p)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.stage:
        cfg["training"]["stage"] = args.stage
    if args.device:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.max_steps is not None:
        cfg["training"]["max_steps_per_epoch"] = args.max_steps

    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir
    elif args.stage:
        cfg["output"]["dir"] = os.path.join("outputs", args.stage)

    if args.dry_run:
        cfg["training"]["epochs"] = 1
        cfg["training"]["max_steps_per_epoch"] = 5
        cfg["output"]["dir"] = os.path.join(cfg["output"]["dir"], "dry_run")
    return cfg


def build_dataloaders(cfg: dict, repo_root: str):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    data_root = resolve_path(repo_root, data_cfg["root"])
    csv_file = resolve_path(repo_root, data_cfg["csv_file"])
    head_box_file = resolve_path(repo_root, data_cfg["head_box_file"])
    mask_root = resolve_path(repo_root, data_cfg["mask_root"])

    t_g, t_h, t_m = build_train_transforms(
        img_size=model_cfg["img_size"],
        use_randaugment=bool(data_cfg.get("use_randaugment", True)),
    )
    v_g, v_h, v_m = build_eval_transforms(img_size=model_cfg["img_size"])

    train_set = AnimeCharacterDataset(
        csv_file=csv_file,
        root=data_root,
        split=data_cfg.get("train_split", "train"),
        head_box_file=head_box_file,
        mask_root=mask_root,
        require_head_box=bool(data_cfg.get("require_head_box", True)),
        require_mask=bool(data_cfg.get("require_mask", True)),
        transform_global=t_g,
        transform_head=t_h,
        transform_mask=t_m,
    )
    val_set = AnimeCharacterDataset(
        csv_file=csv_file,
        root=data_root,
        split=data_cfg.get("val_split", "val"),
        head_box_file=head_box_file,
        mask_root=mask_root,
        require_head_box=bool(data_cfg.get("require_head_box", True)),
        require_mask=bool(data_cfg.get("require_mask", True)),
        transform_global=v_g,
        transform_head=v_h,
        transform_mask=v_m,
    )

    if train_cfg.get("use_class_balanced_sampler", False):
        sampler = build_class_balanced_sampler(
            labels=train_set.labels,
            power=float(train_cfg.get("sampler_power", 1.0)),
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 8)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=True,
        persistent_workers=int(data_cfg.get("num_workers", 8)) > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 8)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=False,
        persistent_workers=int(data_cfg.get("num_workers", 8)) > 0,
    )
    return train_set, val_set, train_loader, val_loader


def get_stage_params(cfg: dict) -> Dict[str, object]:
    stage = cfg["training"]["stage"]
    stage_cfg = cfg["stages"][stage]
    out = {
        "stage": stage,
        "views": list(stage_cfg.get("views", ["global"])),
        "lambda_view": float(stage_cfg.get("lambda_view", cfg["loss"]["lambda_view"])),
        "lambda_proto": float(stage_cfg.get("lambda_proto", cfg["loss"]["lambda_proto"])),
        "use_prototype": bool(stage_cfg.get("use_prototype", True)),
    }
    return out


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer):
    epochs = int(cfg["training"]["epochs"])
    min_lr = float(cfg["scheduler"]["min_lr"])
    warmup_epochs = int(cfg["scheduler"].get("warmup_epochs", 0))
    warmup_start_factor = float(cfg["scheduler"].get("warmup_start_factor", 0.1))

    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs,
            eta_min=min_lr,
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=max(epochs - warmup_epochs, 1),
        eta_min=min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def move_batch_to_device(batch: dict, device: torch.device, views):
    xg = batch["global"].to(device, non_blocking=True)
    xh = batch["head"].to(device, non_blocking=True) if "head" in views else None
    xm = batch["mask"].to(device, non_blocking=True) if "mask" in views else None
    y = batch["label"].to(device, non_blocking=True)
    return xg, xh, xm, y


def train_one_epoch(
    model: AnimeNet,
    proto_bank: Optional[PrototypeBank],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    ce_weight: Optional[torch.Tensor],
    stage_params: Dict[str, object],
    cfg: dict,
    epoch: int,
) -> dict:
    model.train()
    meter = AccuracyMeter()
    loss_track = {"cls_loss": 0.0, "view_loss": 0.0, "proto_loss": 0.0}

    max_steps = cfg["training"].get("max_steps_per_epoch")
    print_freq = int(cfg["training"].get("print_freq", 20))
    grad_clip = float(cfg["optimizer"].get("grad_clip_norm", 0.0))
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"

    for step, batch in enumerate(loader):
        xg, xh, xm, y = move_batch_to_device(batch, device, stage_params["views"])

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(xg, xh, xm)
            loss, loss_dict = compute_loss(
                outputs=outputs,
                labels=y,
                proto_bank=proto_bank if stage_params["use_prototype"] else None,
                ce_weight=ce_weight,
                lambda_view=float(stage_params["lambda_view"]),
                lambda_proto=float(stage_params["lambda_proto"]),
                temperature=float(cfg["loss"]["prototype_temperature"]),
                label_smoothing=float(cfg["training"]["label_smoothing"]),
            )

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if proto_bank is not None and stage_params["use_prototype"]:
            with torch.no_grad():
                proto_bank.update(outputs["z_fuse"].detach(), y)

        meter.update(float(loss.detach().item()), outputs["logits"].detach(), y.detach())
        for k in loss_track:
            loss_track[k] += loss_dict[k]

        if (step + 1) % print_freq == 0:
            print(
                f"[train][epoch {epoch:03d}][step {step + 1:04d}/{len(loader):04d}] "
                f"loss={meter.avg_loss:.4f} top1={meter.top1:.2f} top5={meter.top5:.2f}"
            )

        if max_steps is not None and (step + 1) >= int(max_steps):
            break

    steps = max(step + 1, 1)
    out = meter.as_dict()
    out.update({k: v / steps for k, v in loss_track.items()})
    return out


@torch.no_grad()
def evaluate(
    model: AnimeNet,
    proto_bank: Optional[PrototypeBank],
    loader: DataLoader,
    device: torch.device,
    ce_weight: Optional[torch.Tensor],
    stage_params: Dict[str, object],
    cfg: dict,
) -> dict:
    model.eval()
    meter = AccuracyMeter()
    loss_track = {"cls_loss": 0.0, "view_loss": 0.0, "proto_loss": 0.0}

    for batch in loader:
        xg, xh, xm, y = move_batch_to_device(batch, device, stage_params["views"])
        outputs = model(xg, xh, xm)
        loss, loss_dict = compute_loss(
            outputs=outputs,
            labels=y,
            proto_bank=proto_bank if stage_params["use_prototype"] else None,
            ce_weight=ce_weight,
            lambda_view=float(stage_params["lambda_view"]),
            lambda_proto=float(stage_params["lambda_proto"]),
            temperature=float(cfg["loss"]["prototype_temperature"]),
            label_smoothing=float(cfg["training"]["label_smoothing"]),
        )
        meter.update(float(loss.detach().item()), outputs["logits"], y)
        for k in loss_track:
            loss_track[k] += loss_dict[k]

    steps = max(len(loader), 1)
    out = meter.as_dict()
    out.update({k: v / steps for k, v in loss_track.items()})
    return out


def save_checkpoint(
    path: str,
    model: AnimeNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    proto_bank: Optional[PrototypeBank],
    epoch: int,
    best_top1: float,
    cfg: dict,
) -> None:
    ckpt = {
        "epoch": epoch,
        "best_top1": best_top1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": cfg,
    }
    if proto_bank is not None:
        ckpt["prototype_bank"] = proto_bank.state_dict()
    torch.save(ckpt, path)


def main() -> None:
    args = parse_args()
    repo_root = os.path.abspath(os.path.dirname(__file__))
    cfg = load_yaml(resolve_path(repo_root, args.config))
    cfg = apply_overrides(cfg, args)

    seed_everything(int(cfg["seed"]))
    device = resolve_device(cfg.get("device", "auto"))
    print(f"[setup] device={device}")

    stage_params = get_stage_params(cfg)
    print(
        f"[setup] stage={stage_params['stage']} views={stage_params['views']} "
        f"lambda_view={stage_params['lambda_view']} lambda_proto={stage_params['lambda_proto']} "
        f"use_prototype={stage_params['use_prototype']}"
    )

    train_set, val_set, train_loader, val_loader = build_dataloaders(cfg, repo_root)
    num_classes = max(train_set.labels) + 1
    print(
        f"[data] train={len(train_set)} val={len(val_set)} classes={num_classes} "
        f"batch_size={cfg['training']['batch_size']}"
    )

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

    proto_bank = None
    if stage_params["use_prototype"]:
        proto_bank = PrototypeBank(
            num_classes=num_classes,
            emb_dim=int(cfg["model"]["emb_dim"]),
            momentum=float(cfg["loss"]["prototype_momentum"]),
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )
    scheduler = build_scheduler(cfg=cfg, optimizer=optimizer)
    scaler = GradScaler(device=device.type, enabled=bool(cfg["training"].get("amp", True)) and device.type == "cuda")

    if bool(cfg["training"].get("use_class_balanced_ce", True)):
        ce_weight = build_class_weights(train_set.class_counts, beta=float(cfg["training"]["class_balance_beta"]))
        ce_weight = ce_weight.to(device)
    else:
        ce_weight = None

    start_epoch = 0
    best_top1 = 0.0
    if args.resume:
        resume_path = resolve_path(repo_root, args.resume)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if proto_bank is not None and "prototype_bank" in ckpt:
            proto_bank.load_state_dict(ckpt["prototype_bank"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_top1 = float(ckpt.get("best_top1", 0.0))
        print(f"[resume] loaded {resume_path}, start_epoch={start_epoch}, best_top1={best_top1:.2f}")

    out_dir = resolve_path(repo_root, cfg["output"]["dir"])
    ensure_dir(out_dir)
    latest_path = os.path.join(out_dir, cfg["output"]["checkpoint_name"])
    best_path = os.path.join(out_dir, cfg["output"]["best_name"])
    history_path = os.path.join(out_dir, "history.jsonl")

    with open(os.path.join(out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    epochs = int(cfg["training"]["epochs"])
    for epoch in range(start_epoch, epochs):
        train_stats = train_one_epoch(
            model=model,
            proto_bank=proto_bank,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            ce_weight=ce_weight,
            stage_params=stage_params,
            cfg=cfg,
            epoch=epoch,
        )
        val_stats = evaluate(
            model=model,
            proto_bank=proto_bank,
            loader=val_loader,
            device=device,
            ce_weight=ce_weight,
            stage_params=stage_params,
            cfg=cfg,
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        log_obj = {
            "epoch": epoch,
            "lr": lr_now,
            "train": train_stats,
            "val": val_stats,
        }
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")

        print(
            f"[epoch {epoch:03d}] "
            f"train loss={train_stats['loss']:.4f} top1={train_stats['top1']:.2f} "
            f"val loss={val_stats['loss']:.4f} top1={val_stats['top1']:.2f} "
            f"lr={lr_now:.6e}"
        )

        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            proto_bank=proto_bank,
            epoch=epoch,
            best_top1=best_top1,
            cfg=cfg,
        )

        if val_stats["top1"] >= best_top1:
            best_top1 = val_stats["top1"]
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                proto_bank=proto_bank,
                epoch=epoch,
                best_top1=best_top1,
                cfg=cfg,
            )
            print(f"[checkpoint] saved new best to {best_path} (top1={best_top1:.2f})")

        if (epoch + 1) % int(cfg["training"]["save_every"]) == 0:
            epoch_path = os.path.join(out_dir, f"epoch_{epoch:03d}.pt")
            save_checkpoint(
                path=epoch_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                proto_bank=proto_bank,
                epoch=epoch,
                best_top1=best_top1,
                cfg=cfg,
            )

    print(f"[done] training complete. best_top1={best_top1:.2f}, output_dir={out_dir}")


if __name__ == "__main__":
    main()
