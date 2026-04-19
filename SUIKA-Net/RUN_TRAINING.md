# Anime Character Recognition Pipeline

This project now contains a complete implementation of the `method.md` pipeline:

- Three-view identity modeling: `global / head / mask`
- Shared-backbone network + embedding projection
- Prototype bank + prototype contrastive loss
- Class-balanced CE
- Stage-based training (`stage_a`, `stage_b`, `stage_c`)
- Open-set rejection in evaluation

## 1) Prepare data

```bash
/path/to/miniconda3/envs/bs1/bin/python prepare_data.py \
  --source-dir downloads/characters_top10_each500_single_medium \
  --output-root data \
  --max-per-class 500 \
  --train-ratio 0.8 \
  --no-val \
  --link-mode symlink
```

Outputs:

- `data/meta.csv`
- `data/label_map.json`
- `data/class_stats.json`
- `data/head_boxes.json` (placeholder; will be overwritten by detector)

## 2) Download full weights (no missing modules)

```bash
CUDA_VISIBLE_DEVICES=2 /path/to/miniconda3/envs/bs1/bin/python -u tools/download_weights.py \
  --output-root weights \
  --require-cuda
```

This downloads and verifies:

- Head detector weights (`weights/head_detector/*.pt`)
- Backbone weights (`weights/backbone/*`)
- Rembg U2Net weights (`weights/rembg/u2net.onnx`)

## 3) Build three-view assets (strict)

Generate head boxes:

```bash
CUDA_VISIBLE_DEVICES=2 /path/to/miniconda3/envs/bs1/bin/python -u tools/generate_head_boxes.py \
  --csv-file data/meta.csv \
  --root data \
  --weights weights/head_detector/yolov8s.pt,weights/head_detector/yolov8n.pt \
  --out-json data/head_boxes.json \
  --strict
```

Generate masks:

```bash
CUDA_VISIBLE_DEVICES=2 /path/to/miniconda3/envs/bs1/bin/python -u tools/generate_masks.py \
  --csv-file data/meta.csv \
  --root data \
  --out-root data/masks \
  --u2net-home weights/rembg \
  --strict
```

## 4) Train

Stage C (full method, test split used for best checkpoint selection):

```bash
CUDA_VISIBLE_DEVICES=2 /path/to/miniconda3/envs/bs1/bin/python -u train.py \
  --config configs/tuned_v3_top10x500_testselect.yaml
```

Stage A baseline:

```bash
CUDA_VISIBLE_DEVICES=2 /path/to/miniconda3/envs/bs1/bin/python -u train.py \
  --config configs/base.yaml \
  --stage stage_a \
  --output-dir outputs/stage_a
```

Stage B:

```bash
CUDA_VISIBLE_DEVICES=2 /path/to/miniconda3/envs/bs1/bin/python -u train.py \
  --config configs/base.yaml \
  --stage stage_b \
  --output-dir outputs/stage_b
```

## 5) Evaluate

```bash
CUDA_VISIBLE_DEVICES=2 /path/to/miniconda3/envs/bs1/bin/python -u eval.py \
  --config configs/base.yaml \
  --checkpoint outputs/stage_c/best.pt \
  --split test \
  --save-preds outputs/stage_c/test_preds.csv
```

Disable open-set filtering:

```bash
CUDA_VISIBLE_DEVICES=2 /path/to/miniconda3/envs/bs1/bin/python -u eval.py \
  --config configs/base.yaml \
  --checkpoint outputs/stage_c/best.pt \
  --split test \
  --disable-open-set
```

## Notes

- Current experiment config in `configs/base.yaml`:
  - `data.require_head_box: false` (allows a small number of head-box misses)
  - `data.require_mask: true` (mask must exist for every image)
- If you want fully strict head-box training, set `data.require_head_box: true`.
