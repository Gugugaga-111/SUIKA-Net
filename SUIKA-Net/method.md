下面给你一版**能直接开工的实现方案**，按 **PyTorch + timm** 来写，目标是：

**输入图片 → 输出角色名**
并且同时具备这几个“论文包装点”：

* 三视图身份建模：全图 / 头部 / mask-prompt
* 跨画风鲁棒：style mixing
* 长尾友好：class-balanced + prototype learning
* 细粒度识别：prototype contrastive

我先给整体结构，再给核心代码骨架。

---

# 1. 整体实现框架

## 1.1 训练流程

分 3 个阶段最稳：

### Stage A：强 baseline

先只训一个正常分类器：

* 输入：全图
* backbone：`vit_base_patch16_224` 或 `convnext_tiny`
* loss：交叉熵 / class-balanced CE
* 目的：拿到稳定 baseline

### Stage B：多视图 + 一致性

加入：

* 全图 `global`
* 头部裁剪 `head`
* mask-prompt `mask`

并加入视图一致性损失。

### Stage C：prototype + 长尾增强

加入：

* prototype bank
* prototype contrastive loss
* tail-aware reweighting / sampler

---

# 2. 数据准备

动漫角色识别的关键不是网络，而是**数据组织**。

## 2.1 数据目录建议

```bash
data/
├── images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── meta.csv
├── head_boxes.json
├── masks/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
```

## 2.2 `meta.csv` 格式

```csv
image_id,file_path,label,label_name,anime_id,style_id
1,images/000001.jpg,0,rem,rezero,style_a
2,images/000002.jpg,0,rem,rezero,style_b
3,images/000003.jpg,1,asuna,sao,style_c
```

建议至少保留：

* `label`：角色 ID
* `label_name`：角色名
* `anime_id`：作品名
* `style_id`：画风来源或域标签，可选

后面跨画风实验会用到。

---

# 3. 预处理

## 3.1 头部框

最省事做法：

* 用动漫人脸检测器离线跑一遍
* 没检测到就退化成中心裁剪

存成：

```json
{
  "000001.jpg": [x1, y1, x2, y2],
  "000002.jpg": [x1, y1, x2, y2]
}
```

## 3.2 mask-prompt

最简单的可落地版，不必真做高质量分割：

### 方案 1：框近似 mask

直接把 head box 或人物 box 扩张成软 mask。

### 方案 2：现成分割模型离线生成

用动漫人物分割模型先离线生成 mask。

### 方案 3：最省工程版

没有分割模型时，用头部框 + 高斯模糊背景，也能先跑。

mask-prompt 图像生成逻辑：

```python
foreground = img * mask
background = gaussian_blur(img) * (1 - mask)
x_mask = foreground + background
```

这已经足够像“prompted input”。

---

# 4. Dataset 实现

下面是核心 dataset 骨架。

```python
import os
import json
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

class AnimeCharacterDataset(Dataset):
    def __init__(self, csv_file, root, head_box_file=None, mask_root=None,
                 transform_global=None, transform_head=None, transform_mask=None):
        self.df = pd.read_csv(csv_file)
        self.root = root
        self.mask_root = mask_root
        self.transform_global = transform_global
        self.transform_head = transform_head
        self.transform_mask = transform_mask

        if head_box_file and os.path.exists(head_box_file):
            with open(head_box_file, "r", encoding="utf-8") as f:
                self.head_boxes = json.load(f)
        else:
            self.head_boxes = {}

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return img

    def _crop_head(self, img, file_name):
        w, h = img.size
        if file_name in self.head_boxes:
            x1, y1, x2, y2 = self.head_boxes[file_name]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                return img.crop((x1, y1, x2, y2))
        # fallback: center crop upper area
        return img.crop((w * 0.2, h * 0.0, w * 0.8, h * 0.6))

    def _load_mask(self, img, file_name):
        if self.mask_root is None:
            return None

        mask_path = os.path.join(self.mask_root, os.path.splitext(file_name)[0] + ".png")
        if not os.path.exists(mask_path):
            return None

        mask = Image.open(mask_path).convert("L").resize(img.size)
        return mask

    def _make_mask_prompt(self, img, mask):
        if mask is None:
            # fallback: whole image with weak blur background simulation
            blurred = img.filter(ImageFilter.GaussianBlur(radius=6))
            return Image.blend(img, blurred, alpha=0.2)

        img_np = np.array(img).astype(np.float32)
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_np = np.expand_dims(mask_np, axis=-1)

        blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=8))).astype(np.float32)
        out = img_np * mask_np + blurred * (1.0 - mask_np)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.root, row["file_path"])
        file_name = os.path.basename(row["file_path"])
        label = int(row["label"])

        img = self._load_image(file_path)
        img_head = self._crop_head(img, file_name)
        mask = self._load_mask(img, file_name)
        img_mask = self._make_mask_prompt(img, mask)

        x_global = self.transform_global(img) if self.transform_global else img
        x_head = self.transform_head(img_head) if self.transform_head else img_head
        x_mask = self.transform_mask(img_mask) if self.transform_mask else img_mask

        return {
            "global": x_global,
            "head": x_head,
            "mask": x_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long)
        }
```

---

# 5. 数据增强

建议分视图做不同增强。

```python
from torchvision import transforms

def build_transforms(img_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform_global = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        normalize
    ])

    transform_head = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        normalize
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    return transform_global, transform_head, transform_mask
```

---

# 6. 模型结构

建议一开始用**共享 backbone**。
三路输入都过同一个 backbone，减少参数量，也更稳定。

## 6.1 模型组成

* `backbone`
* `proj_head`：输出 embedding
* `classifier`
* `prototype bank`

## 6.2 原型定义

最简单版：

* 每类一个 prototype：`[num_classes, dim]`

稍强版：

* 每类 K 个 prototype：`[num_classes, K, dim]`

先上**每类一个 prototype**就够了。

---

# 7. Style Mixing 模块

你想包装“跨画风泛化”，最轻量的实现是 **MixStyle** 风格统计混合。

对 CNN 比较好插；ViT 也能做，但工程复杂一点。
如果你想最快落地，我建议：

* **第一版 backbone 用 ConvNeXt-T**
* 中间层插 style mixing

下面给一个简化版 style mixing：

```python
import torch
import torch.nn as nn

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()

        x_norm = (x - mu) / sig

        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]

        lam = self.beta.sample((B, 1, 1, 1)).to(x.device)
        mu_mix = mu * lam + mu2 * (1 - lam)
        sig_mix = sig * lam + sig2 * (1 - lam)

        return x_norm * sig_mix + mu_mix
```

---

# 8. 主模型代码骨架

这里给一个**共享 backbone + 三视图融合 + prototype** 的结构。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class AnimeNet(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", num_classes=1000, emb_dim=512):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, emb_dim)
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def encode(self, x):
        feat = self.backbone(x)
        emb = self.proj(feat)
        emb = F.normalize(emb, dim=-1)
        return emb

    def forward(self, x_global, x_head=None, x_mask=None):
        z_g = self.encode(x_global)

        out = {"z_global": z_g}

        if x_head is not None:
            z_h = self.encode(x_head)
            out["z_head"] = z_h
        if x_mask is not None:
            z_m = self.encode(x_mask)
            out["z_mask"] = z_m

        # 融合策略：平均
        z_list = [out["z_global"]]
        if "z_head" in out:
            z_list.append(out["z_head"])
        if "z_mask" in out:
            z_list.append(out["z_mask"])

        z_fuse = torch.stack(z_list, dim=0).mean(dim=0)
        z_fuse = F.normalize(z_fuse, dim=-1)

        logits = self.classifier(z_fuse)

        out["z_fuse"] = z_fuse
        out["logits"] = logits
        return out
```

---

# 9. Prototype Bank

最简单可用版：用 buffer 存原型，训练时 EMA 更新。

```python
class PrototypeBank(nn.Module):
    def __init__(self, num_classes, emb_dim, momentum=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.momentum = momentum
        self.register_buffer("prototypes", F.normalize(torch.randn(num_classes, emb_dim), dim=-1))

    @torch.no_grad()
    def update(self, feats, labels):
        # feats: [B, D], normalized
        for c in labels.unique():
            mask = labels == c
            feat_c = feats[mask].mean(dim=0)
            old = self.prototypes[c]
            new = self.momentum * old + (1 - self.momentum) * feat_c
            self.prototypes[c] = F.normalize(new, dim=-1)

    def similarity(self, feats):
        # feats [B, D], prototypes [C, D]
        return feats @ self.prototypes.t()
```

---

# 10. 损失函数

总损失：

[
L = L_{cls} + \lambda_1 L_{view} + \lambda_2 L_{proto}
]

再加 class-balanced 权重。

## 10.1 Class-Balanced CE

```python
def build_class_weights(cls_counts, beta=0.9999):
    counts = torch.tensor(cls_counts, dtype=torch.float32)
    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(cls_counts)
    return weights
```

## 10.2 视图一致性损失

```python
def view_consistency_loss(z1, z2):
    return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()
```

## 10.3 Prototype loss

把原型相似度当分类 logits。

```python
def prototype_loss(proto_bank, feats, labels, temperature=0.07):
    sim = proto_bank.similarity(feats) / temperature
    return F.cross_entropy(sim, labels)
```

## 10.4 总损失

```python
def compute_loss(outputs, labels, proto_bank, ce_weight=None,
                 lambda_view=0.2, lambda_proto=0.5):
    logits = outputs["logits"]
    z_fuse = outputs["z_fuse"]

    cls_loss = F.cross_entropy(logits, labels, weight=ce_weight)

    view_loss = 0.0
    n = 0
    if "z_head" in outputs:
        view_loss += view_consistency_loss(outputs["z_global"], outputs["z_head"])
        n += 1
    if "z_mask" in outputs:
        view_loss += view_consistency_loss(outputs["z_global"], outputs["z_mask"])
        n += 1
    if "z_head" in outputs and "z_mask" in outputs:
        view_loss += view_consistency_loss(outputs["z_head"], outputs["z_mask"])
        n += 1
    if n > 0:
        view_loss = view_loss / n
    else:
        view_loss = torch.tensor(0.0, device=labels.device)

    p_loss = prototype_loss(proto_bank, z_fuse, labels)

    total = cls_loss + lambda_view * view_loss + lambda_proto * p_loss
    return total, {
        "cls_loss": cls_loss.item(),
        "view_loss": float(view_loss.item()),
        "proto_loss": p_loss.item()
    }
```

---

# 11. 训练循环

```python
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, proto_bank, loader, optimizer, device, ce_weight=None):
    model.train()
    scaler = GradScaler()

    stats = {"loss": 0.0, "cls_loss": 0.0, "view_loss": 0.0, "proto_loss": 0.0}

    for batch in loader:
        xg = batch["global"].to(device)
        xh = batch["head"].to(device)
        xm = batch["mask"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(xg, xh, xm)
            loss, loss_dict = compute_loss(outputs, y, proto_bank, ce_weight=ce_weight)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            proto_bank.update(outputs["z_fuse"].detach(), y)

        stats["loss"] += loss.item()
        stats["cls_loss"] += loss_dict["cls_loss"]
        stats["view_loss"] += loss_dict["view_loss"]
        stats["proto_loss"] += loss_dict["proto_loss"]

    n = len(loader)
    for k in stats:
        stats[k] /= n
    return stats
```

---

# 12. 推理逻辑

推理时同样走三视图：

```python
@torch.no_grad()
def predict(model, img_global, img_head, img_mask):
    model.eval()
    out = model(img_global, img_head, img_mask)
    prob = torch.softmax(out["logits"], dim=-1)
    pred = prob.argmax(dim=-1)
    conf = prob.max(dim=-1).values
    return pred, conf, out["z_fuse"]
```

## 开集拒识

加一个简单阈值：

### 方法 1：softmax 阈值

* 若最大概率 < `tau1`，判未知

### 方法 2：prototype 距离阈值

* 若与最近 prototype 的相似度 < `tau2`，判未知

```python
@torch.no_grad()
def open_set_predict(model, proto_bank, xg, xh, xm, tau_prob=0.5, tau_sim=0.35):
    out = model(xg, xh, xm)
    prob = torch.softmax(out["logits"], dim=-1)
    pred = prob.argmax(dim=-1)
    max_prob = prob.max(dim=-1).values

    sim = proto_bank.similarity(out["z_fuse"])
    max_sim = sim.max(dim=-1).values

    unknown = (max_prob < tau_prob) | (max_sim < tau_sim)
    pred[unknown] = -1
    return pred
```

---

# 13. 训练配置建议

## 13.1 超参数

先用这一组：

```python
model_name = "vit_base_patch16_224"
img_size = 224
batch_size = 64
lr = 3e-4
weight_decay = 1e-4
epochs = 50
emb_dim = 512
lambda_view = 0.2
lambda_proto = 0.5
```

## 13.2 优化器

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

---

# 14. 推荐的完整训练顺序

## 第一步：baseline

只用：

* `global`
* `CE`

拿到基础 top-1。

## 第二步：+ 三视图

加：

* `global + head + mask`
* `view consistency`

看提升。

## 第三步：+ prototype

加：

* prototype bank
* prototype loss

## 第四步：+ 长尾

加：

* class-balanced CE
* class-aware sampler

## 第五步：+ style mixing

如果 backbone 是 ConvNeXt，就把 MixStyle 插到前两层。

---

# 15. 最小可用项目结构

```bash
project/
├── train.py
├── eval.py
├── models/
│   ├── anime_net.py
│   ├── prototype_bank.py
│   └── mixstyle.py
├── datasets/
│   └── anime_dataset.py
├── losses/
│   └── losses.py
├── configs/
│   └── base.yaml
└── utils/
    ├── metrics.py
    └── sampler.py
```

---

# 16. `train.py` 主入口骨架

```python
def main():
    device = "cuda"

    t_g, t_h, t_m = build_transforms(224)
    train_set = AnimeCharacterDataset(
        csv_file="data/meta.csv",
        root="data",
        head_box_file="data/head_boxes.json",
        mask_root="data/masks",
        transform_global=t_g,
        transform_head=t_h,
        transform_mask=t_m
    )
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

    num_classes = train_set.df["label"].nunique()
    model = AnimeNet(model_name="vit_base_patch16_224", num_classes=num_classes, emb_dim=512).to(device)
    proto_bank = PrototypeBank(num_classes, 512).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # class weights
    cls_counts = train_set.df["label"].value_counts().sort_index().tolist()
    ce_weight = build_class_weights(cls_counts).to(device)

    for epoch in range(50):
        stats = train_one_epoch(model, proto_bank, train_loader, optimizer, device, ce_weight=ce_weight)
        print(f"Epoch {epoch}: {stats}")
```

---

# 17. 你论文里可以怎么写创新点

直接写成这三点就够了：

## 创新点 1

**提出三视图角色身份建模框架**
联合全局外观、头部细节和 mask-prompt 提示视图，增强对背景干扰和局部细粒度属性的鲁棒性。

## 创新点 2

**提出 style-invariant prototype learning**
通过视图一致性与原型约束，使模型更关注“角色身份”而不是“作品/画风统计”。

## 创新点 3

**面向长尾动漫角色识别的 prototype-enhanced learning**
利用类原型缓解尾类样本不足造成的分类器不稳定问题。

---

# 18. 一开始最该做的简化版

别一上来全堆满。
建议你先做这个版本：

## V1

* backbone：`vit_base_patch16_224`
* 输入：`global + head + mask`
* loss：`CE + 0.2 * view_loss + 0.5 * proto_loss`
* 长尾：class-balanced CE
* 不加 MixStyle

这是最稳的。

## V2

* backbone 改 `convnext_tiny`
* 插 MixStyle
* 做跨画风实验
