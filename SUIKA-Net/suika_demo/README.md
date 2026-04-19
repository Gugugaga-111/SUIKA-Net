# SUIKA-Net Demo

`SUIKA-Net` 本地前后端一体 Demo：

1. 上传一张角色图
2. 后端用当前方法完整推理（`global + head + mask`）
3. 返回预测角色名
4. 基于预测角色，从当前数据集随机展示若干图
5. 每张结果展示 `pixiv id` 与图片名称

## 特性

- 前端风格参考 `mllm_demo`，但为全新配色与内容结构
- 后端直接加载本地 checkpoint，不调用外部大模型 API
- 头部检测：`YOLO`（CUDA）
- 掩码生成：`rembg + onnxruntime-gpu`（CUDAExecutionProvider）
- 分类模型：`SUIKA-Net` 三视角融合
- 支持“同角色重抽”（不重复跑分类）
- 支持固定随机种子（`seed + round`）保证重抽可复现
- 默认启动 warmup，降低首个请求延迟

## 启动

在仓库根目录执行：

```bash
/path/to/miniconda3/envs/bs1/bin/python suika_demo/server.py --host 0.0.0.0 --port 8090
```

打开：

- `http://localhost:8090`

内网访问（同网段其他机器）：

1. 查询本机内网 IP：`hostname -I`
2. 其他机器访问：`http://<你的内网IP>:8090`

## 常用参数

- `--config`：模型配置，默认 `configs/tuned_v3_top20x500_testselect.yaml`
- `--checkpoint`：模型权重，默认 `outputs/stage_c/tuned_v3_top20x500_testselect/best.pt`
- `--head-weights`：头部检测权重，支持逗号分隔多个
- `--u2net-home` / `--u2net-model`：掩码模型目录与名称
- `--default-gallery`：默认随机展示数量
- `--max-gallery`：前端可请求的最大展示数量
- `--no-warmup`：禁用启动 warmup（默认会 warmup）

## API

### `POST /api/predict`

`multipart/form-data`：

- `image`: 图片文件
- `num_gallery`: 随机返回数量（可选）
- `seed`: 固定随机种子（可选）
- `round`: 固定种子模式下的重抽轮次（可选）

返回 JSON 主要字段：

- `character_name`
- `label_id`
- `confidence`
- `top_predictions`
- `gallery[]`（含 `pixiv_id`, `image_name`, `file_name`, `image_url`）
- `meta.sampling`（固定种子/seed/round 信息）

### `GET /api/gallery?label_id=...&num_gallery=...&seed=...&round=...`

按给定角色标签返回随机样本图；用于“同角色重抽”而不重复推理。

### `GET /api/image?path=...`

按相对路径返回数据集图片，用于前端图库展示。

### `GET /api/health`

返回服务状态、设备与视角信息。
