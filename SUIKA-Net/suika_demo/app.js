const API_PREDICT = "/api/predict";
const API_GALLERY = "/api/gallery";
const MAX_FILE_SIZE = 16 * 1024 * 1024;

const form = document.querySelector("#predict-form");
const imageInput = document.querySelector("#image-file");
const numGalleryInput = document.querySelector("#num-gallery");
const useFixedSeedInput = document.querySelector("#use-fixed-seed");
const seedValueInput = document.querySelector("#seed-value");
const submitBtn = document.querySelector("#submit-btn");
const resampleBtn = document.querySelector("#resample-btn");
const resetBtn = document.querySelector("#reset-btn");

const previewImage = document.querySelector("#preview-image");
const previewPlaceholder = document.querySelector("#preview-placeholder");
const resultContent = document.querySelector("#result-content");
const resultTemplate = document.querySelector("#result-template");
const statusPill = document.querySelector("#status-pill");
const metaInfo = document.querySelector("#meta-info");

let currentPreviewUrl = "";
let lastResult = null;
let resampleRound = 0;
let predictLoading = false;
let resampleLoading = false;

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) {
    clearPreview();
    return;
  }

  if (!validateFile(file)) {
    imageInput.value = "";
    clearPreview();
    return;
  }

  if (currentPreviewUrl) {
    URL.revokeObjectURL(currentPreviewUrl);
  }

  currentPreviewUrl = URL.createObjectURL(file);
  previewImage.src = currentPreviewUrl;
  previewImage.style.display = "block";
  previewPlaceholder.style.display = "none";
});

useFixedSeedInput.addEventListener("change", () => {
  seedValueInput.disabled = !useFixedSeedInput.checked;
  if (useFixedSeedInput.checked && !seedValueInput.value.trim()) {
    seedValueInput.value = "42";
  }
  resampleRound = 0;
});

seedValueInput.addEventListener("input", () => {
  if (useFixedSeedInput.checked) {
    resampleRound = 0;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const imageFile = imageInput.files?.[0];
  if (!imageFile) {
    showError("请先上传一张角色图片。", "分类失败");
    return;
  }
  if (!validateFile(imageFile)) {
    return;
  }

  const numGallery = clampGalleryCount(numGalleryInput.value);

  let seed = null;
  try {
    seed = getSeedOrNull();
  } catch (error) {
    const message = error instanceof Error ? error.message : "种子参数错误。";
    showError(message, "参数错误");
    return;
  }

  predictLoading = true;
  refreshActionState();
  setStatus("loading", "分类中");

  try {
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("num_gallery", String(numGallery));
    if (seed !== null) {
      formData.append("seed", String(seed));
      formData.append("round", "0");
    }

    const response = await fetch(API_PREDICT, {
      method: "POST",
      body: formData,
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = data?.error || `请求失败（HTTP ${response.status}）`;
      throw new Error(message);
    }

    lastResult = data;
    resampleRound = 0;
    renderResult(data);
    setStatus("done", "分类完成");
    renderMetaFromPredict(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "请求失败，请稍后重试。";
    showError(message, "分类失败");
  } finally {
    predictLoading = false;
    refreshActionState();
  }
});

resampleBtn.addEventListener("click", async () => {
  if (!lastResult || !Number.isFinite(Number(lastResult.label_id))) {
    showError("请先完成一次分类，再进行同角色重抽。", "无法重抽");
    return;
  }

  const numGallery = clampGalleryCount(numGalleryInput.value);

  let seed = null;
  try {
    seed = getSeedOrNull();
  } catch (error) {
    const message = error instanceof Error ? error.message : "种子参数错误。";
    showError(message, "参数错误");
    return;
  }

  const nextRound = seed !== null ? resampleRound + 1 : 0;

  resampleLoading = true;
  refreshActionState();
  setStatus("loading", "重抽中");

  try {
    const params = new URLSearchParams();
    params.set("label_id", String(lastResult.label_id));
    params.set("num_gallery", String(numGallery));
    if (seed !== null) {
      params.set("seed", String(seed));
      params.set("round", String(nextRound));
    }

    const response = await fetch(`${API_GALLERY}?${params.toString()}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = data?.error || `请求失败（HTTP ${response.status}）`;
      throw new Error(message);
    }

    resampleRound = nextRound;
    lastResult.gallery = Array.isArray(data?.gallery) ? data.gallery : [];
    renderResult(lastResult);
    setStatus("done", "已重抽样");
    renderMetaFromResample(data, lastResult);
  } catch (error) {
    const message = error instanceof Error ? error.message : "重抽失败，请稍后重试。";
    showError(message, "重抽失败");
  } finally {
    resampleLoading = false;
    refreshActionState();
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  useFixedSeedInput.checked = false;
  seedValueInput.value = "42";
  seedValueInput.disabled = true;

  clearPreview();
  resetResult();
});

function refreshActionState() {
  const busy = predictLoading || resampleLoading;

  imageInput.disabled = busy;
  numGalleryInput.disabled = busy;
  useFixedSeedInput.disabled = busy;
  seedValueInput.disabled = busy || !useFixedSeedInput.checked;

  submitBtn.disabled = busy;
  submitBtn.textContent = predictLoading ? "分类中..." : "开始分类";

  resampleBtn.disabled = busy || !lastResult;
  resampleBtn.textContent = resampleLoading ? "重抽中..." : "同角色重抽";

  resetBtn.disabled = busy;
}

function setStatus(kind, text) {
  statusPill.className = `pill ${kind}`;
  statusPill.textContent = text;
}

function clearPreview() {
  if (currentPreviewUrl) {
    URL.revokeObjectURL(currentPreviewUrl);
    currentPreviewUrl = "";
  }
  previewImage.src = "";
  previewImage.style.display = "none";
  previewPlaceholder.style.display = "block";
}

function validateFile(file) {
  const validTypes = new Set(["image/png", "image/jpeg", "image/webp"]);
  if (!validTypes.has(file.type)) {
    showError("仅支持 PNG / JPG / WEBP 格式。", "输入不合法");
    return false;
  }
  if (file.size > MAX_FILE_SIZE) {
    showError("图片大小不能超过 16MB。", "输入不合法");
    return false;
  }
  return true;
}

function resetResult() {
  lastResult = null;
  resampleRound = 0;

  resultContent.className = "result-empty";
  resultContent.innerHTML = "<p>上传图片并点击“开始分类”后，这里会显示角色预测和随机样本结果。</p>";
  metaInfo.textContent = "";
  setStatus("idle", "等待输入");
  refreshActionState();
}

function showError(message, statusText = "分类失败") {
  setStatus("error", statusText);
  resultContent.className = "result-empty";
  resultContent.innerHTML = `<p>${escapeHtml(message)}</p>`;
  metaInfo.textContent = "";
}

function renderMetaFromPredict(data) {
  const latencyMs = Number(data?.meta?.latency_ms || 0);
  const device = data?.meta?.device || "-";
  const views = Array.isArray(data?.meta?.views) ? data.meta.views.join(" / ") : "-";
  const sampling = buildSamplingText(data?.meta?.sampling);
  metaInfo.textContent = `推理耗时：${latencyMs.toFixed(1)} ms ｜ 设备：${device} ｜ 视角：${views} ｜ 抽样：${sampling}`;
}

function renderMetaFromResample(data, result) {
  const latencyMs = Number(data?.meta?.latency_ms || 0);
  const characterName = String(result?.character_name || "未知角色");
  const sampling = buildSamplingText(data?.meta?.sampling);
  metaInfo.textContent = `重抽耗时：${latencyMs.toFixed(1)} ms ｜ 当前角色：${characterName} ｜ 抽样：${sampling}`;
}

function buildSamplingText(sampling) {
  if (!sampling || !sampling.fixed_seed) {
    return "随机模式";
  }
  const seed = sampling.seed ?? "-";
  const round = sampling.round ?? 0;
  return `固定种子 ${seed}（round ${round}）`;
}

function renderResult(data) {
  const fragment = resultTemplate.content.cloneNode(true);

  const name = String(data?.character_name || "未知角色");
  const labelId = Number(data?.label_id);
  const confidence = normalizeConfidence(data?.confidence);
  const protoSim = data?.prototype_similarity;

  fragment.querySelector("#char-name").textContent = name;
  fragment.querySelector("#label-text").textContent = `标签：${Number.isFinite(labelId) ? labelId : "-"}`;
  fragment.querySelector("#sim-text").textContent =
    protoSim === null || protoSim === undefined
      ? "Prototype Similarity：-"
      : `Prototype Similarity：${Number(protoSim).toFixed(4)}`;

  fragment.querySelector("#confidence-text").textContent = `${confidence}%`;
  fragment.querySelector("#confidence-bar").style.width = `${confidence}%`;

  const topkChips = fragment.querySelector("#topk-chips");
  renderTopK(topkChips, data?.top_predictions);

  const galleryGrid = fragment.querySelector("#gallery-grid");
  renderGallery(galleryGrid, data?.gallery);

  resultContent.className = "";
  resultContent.innerHTML = "";
  resultContent.appendChild(fragment);
}

function renderTopK(container, topPredictions) {
  container.innerHTML = "";
  const list = Array.isArray(topPredictions) ? topPredictions : [];
  if (!list.length) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = "暂无 Top-K 结果";
    container.appendChild(chip);
    return;
  }

  for (const item of list) {
    const chip = document.createElement("span");
    chip.className = "chip";
    const name = String(item?.character_name || "未知角色");
    const conf = normalizeConfidence(item?.confidence);
    chip.textContent = `${name} · ${conf}%`;
    container.appendChild(chip);
  }
}

function renderGallery(container, galleryItems) {
  container.innerHTML = "";
  const items = Array.isArray(galleryItems) ? galleryItems : [];

  if (!items.length) {
    const empty = document.createElement("p");
    empty.textContent = "暂无可展示样本。";
    container.appendChild(empty);
    return;
  }

  for (const item of items) {
    const card = document.createElement("article");
    card.className = "gallery-card";

    const img = document.createElement("img");
    img.loading = "lazy";
    img.decoding = "async";
    img.src = String(item?.image_url || "");
    img.alt = String(item?.image_name || item?.file_name || "dataset image");

    const meta = document.createElement("div");
    meta.className = "gallery-meta";

    const name = document.createElement("p");
    name.textContent = `图片名称：${String(item?.image_name || "未知")}`;

    const pixiv = document.createElement("p");
    pixiv.textContent = `Pixiv ID：${String(item?.pixiv_id || "未知")}`;

    meta.append(name, pixiv);
    card.append(img, meta);
    container.appendChild(card);
  }
}

function clampGalleryCount(rawValue) {
  const value = Number(rawValue);
  if (!Number.isFinite(value) || value <= 0) {
    return 10;
  }
  const snapped = Math.round(value / 5) * 5;
  return Math.max(5, Math.min(100, snapped));
}

function getSeedOrNull() {
  if (!useFixedSeedInput.checked) {
    return null;
  }
  const value = Number(seedValueInput.value);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error("固定随机种子必须是非负整数。");
  }
  return value;
}

function normalizeConfidence(value) {
  if (value === null || value === undefined || value === "") {
    return 0;
  }
  const num = Number(value);
  if (Number.isNaN(num)) {
    return 0;
  }
  const scaled = num <= 1 ? num * 100 : num;
  return Math.max(0, Math.min(100, Math.round(scaled)));
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

resetResult();
