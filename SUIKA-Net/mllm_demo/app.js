const FIXED_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1";
const FIXED_MODEL = "qwen3.6-plus";
const FIXED_API_KEY = "<REDACTED_API_KEY>";
const MAX_FILE_SIZE = 8 * 1024 * 1024;

const form = document.querySelector("#search-form");
const imageInput = document.querySelector("#image-file");
const extraPromptInput = document.querySelector("#extra-prompt");
const submitBtn = document.querySelector("#submit-btn");
const resetBtn = document.querySelector("#reset-btn");

const previewImage = document.querySelector("#preview-image");
const previewPlaceholder = document.querySelector("#preview-placeholder");
const resultContent = document.querySelector("#result-content");
const resultTemplate = document.querySelector("#result-template");
const statusPill = document.querySelector("#status-pill");
const metaInfo = document.querySelector("#meta-info");

let currentPreviewUrl = "";

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

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const imageFile = imageInput.files?.[0];
  const extraPrompt = extraPromptInput.value.trim();

  if (!imageFile) {
    showError("\u8bf7\u5148\u4e0a\u4f20\u4e00\u5f20\u89d2\u8272\u56fe\u7247\u3002");
    return;
  }
  if (!validateFile(imageFile)) {
    return;
  }

  setLoading(true);
  setStatus("loading", "\u8bc6\u522b\u4e2d");

  try {
    const dataUrl = await fileToDataUrl(imageFile);
    const response = await fetch(`${FIXED_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${FIXED_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: FIXED_MODEL,
        enable_thinking: false,
        temperature: 0.2,
        messages: [
          {
            role: "system",
            content:
              "\u4f60\u662f\u52a8\u6f2b\u89d2\u8272\u8bc6\u522b\u52a9\u624b\u3002\u8bf7\u4ec5\u6839\u636e\u56fe\u7247\u5185\u5bb9\u505a\u51fa\u5224\u65ad\uff0c\u4e0d\u786e\u5b9a\u65f6\u8981\u660e\u786e\u8bf4\u660e\u3002\u8bf7\u53ea\u8fd4\u56de\u5408\u6cd5 JSON\uff0c\u4e0d\u8981\u8f93\u51fa\u4efb\u4f55\u989d\u5916\u6587\u5b57\u3002",
          },
          {
            role: "user",
            content: [
              {
                type: "text",
                text: buildPrompt(extraPrompt),
              },
              {
                type: "image_url",
                image_url: { url: dataUrl },
              },
            ],
          },
        ],
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      const message =
        data?.error?.message || `\u8bf7\u6c42\u5931\u8d25\uff08HTTP ${response.status}\uff09`;
      throw new Error(message);
    }

    const rawContent = data?.choices?.[0]?.message?.content;
    if (!rawContent) {
      throw new Error("\u6a21\u578b\u672a\u8fd4\u56de\u53ef\u7528\u5185\u5bb9\uff0c\u8bf7\u91cd\u8bd5\u3002");
    }

    const parsed = extractJSON(rawContent);
    renderResult(parsed, rawContent);
    setStatus("done", "\u8bc6\u522b\u5b8c\u6210");
    showMeta(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "\u8bc6\u522b\u5931\u8d25\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5\u3002";
    showError(message);
  } finally {
    setLoading(false);
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  clearPreview();
  resultContent.className = "result-empty";
  resultContent.innerHTML = "<p>\u4e0a\u4f20\u56fe\u7247\u5e76\u70b9\u51fb\u201c\u5f00\u59cb\u8bc6\u522b\u201d\u540e\uff0c\u8fd9\u91cc\u4f1a\u663e\u793a\u8be6\u7ec6\u89d2\u8272\u6863\u6848\u3002</p>";
  metaInfo.textContent = "";
  setStatus("idle", "\u7b49\u5f85\u8f93\u5165");
});

function setLoading(loading) {
  submitBtn.disabled = loading;
  submitBtn.textContent = loading ? "\u8bc6\u522b\u4e2d..." : "\u5f00\u59cb\u8bc6\u522b";
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
    showError("\u4ec5\u652f\u6301 PNG / JPG / WEBP \u683c\u5f0f\u7684\u56fe\u7247\u3002");
    return false;
  }
  if (file.size > MAX_FILE_SIZE) {
    showError("\u56fe\u7247\u5927\u5c0f\u4e0d\u80fd\u8d85\u8fc7 8MB\u3002");
    return false;
  }
  return true;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("\u56fe\u7247\u8bfb\u53d6\u5931\u8d25\uff0c\u8bf7\u91cd\u65b0\u4e0a\u4f20\u3002"));
    reader.readAsDataURL(file);
  });
}

function buildPrompt(extraPrompt) {
  const basePrompt = [
    "\u8bf7\u8bc6\u522b\u8fd9\u5f20\u56fe\u7247\u4e2d\u7684\u52a8\u6f2b\u89d2\u8272\uff0c\u5e76\u4ec5\u8fd4\u56de JSON\uff08\u4e0d\u8981 markdown \u4ee3\u7801\u5757\uff09\u3002",
    "\u8bf7\u5c3d\u53ef\u80fd\u8fd4\u56de\u66f4\u8be6\u7ec6\u7684\u89d2\u8272\u6863\u6848\uff0c\u4f46\u4e0d\u8981\u81c6\u6d4b\uff1a\u4e0d\u786e\u5b9a\u5c31\u5199\u201c\u672a\u77e5\u201d\u6216\u7a7a\u6570\u7ec4\u3002",
    "{",
    '  "character_name": "\u89d2\u8272\u540d\uff0c\u65e0\u6cd5\u786e\u5b9a\u586b\u5199\u672a\u77e5",',
    '  "japanese_name": "\u89d2\u8272\u65e5\u6587\u540d\uff08\u6f22\u5b57/\u3075\u308a\u304c\u306a/\u30ab\u30bf\u30ab\u30ca\uff09\uff0c\u65e0\u6cd5\u786e\u5b9a\u586b\u5199\u672a\u77e5",',
    '  "anime_title": "\u6240\u5c5e\u4f5c\u54c1\u540d\uff0c\u65e0\u6cd5\u786e\u5b9a\u586b\u5199\u672a\u77e5",',
    '  "aliases": ["\u522b\u540d1", "\u522b\u540d2"],',
    '  "gender": "\u6027\u522b\u6216\u672a\u77e5",',
    '  "age_stage": "\u5e74\u9f84\u6bb5\uff0c\u4f8b\u5982\u5c11\u5e74/\u6210\u5e74/\u672a\u77e5",',
    '  "identity": "\u8eab\u4efd\u6216\u804c\u4e1a\u5b9a\u4f4d",',
    '  "camp_or_affiliation": "\u9635\u8425/\u7ec4\u7ec7/\u6240\u5c5e\u56e2\u4f53",',
    '  "appearance_points": ["\u53ef\u89c1\u5916\u89c2\u7279\u5f81A", "\u53ef\u89c1\u5916\u89c2\u7279\u5f81B"],',
    '  "description": "\u4e2d\u6587\u7b80\u4ecb\uff0c200-320\u5b57\uff0c\u5305\u542b\u8bbe\u5b9a/\u6027\u683c/\u80fd\u529b/\u4e0e\u4e3b\u7ebf\u7684\u5173\u7cfb",',
    '  "story_role": "\u89d2\u8272\u5728\u5267\u60c5\u4e2d\u7684\u5b9a\u4f4d",',
    '  "background": "\u80cc\u666f\u4e0e\u7ecf\u5386\u6458\u8981",',
    '  "personality_traits": ["\u6027\u683c\u7279\u5f811", "\u6027\u683c\u7279\u5f812"],',
    '  "abilities": ["\u80fd\u529b/\u6280\u80fd1", "\u80fd\u529b/\u6280\u80fd2"],',
    '  "relationships": ["\u4e0eXX\u7684\u5173\u7cfb\uff1a..."],',
    '  "signature_lines_or_markers": ["\u6807\u5fd7\u53f0\u8bcd/\u53e3\u5934\u7985/\u52a8\u4f5c/\u9053\u5177"],',
    '  "evidence": ["\u4f60\u8bc6\u522b\u8be5\u89d2\u8272\u7684\u56fe\u50cf\u4f9d\u636e1", "\u4f9d\u636e2"],',
    '  "confidence": "0\u5230100\u7684\u6570\u503c",',
    '  "extra_note": "\u4e0d\u786e\u5b9a\u70b9\u6216\u6b67\u4e49\u8bf4\u660e\uff0c\u6ca1\u6709\u5219\u5199\u65e0"',
    "}",
  ].join("\n");

  if (!extraPrompt) {
    return basePrompt;
  }
  return `${basePrompt}\n\n\u7528\u6237\u8865\u5145\u8981\u6c42\uff1a${extraPrompt}`;
}

function extractJSON(rawText) {
  if (typeof rawText !== "string") {
    return null;
  }

  const fenced = rawText.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) {
    try {
      return JSON.parse(fenced[1].trim());
    } catch {
      // Continue to fallback parsing below.
    }
  }

  const start = rawText.indexOf("{");
  const end = rawText.lastIndexOf("}");
  if (start >= 0 && end > start) {
    const snippet = rawText.slice(start, end + 1);
    try {
      return JSON.parse(snippet);
    } catch {
      return null;
    }
  }

  return null;
}

function renderResult(parsed, rawContent) {
  const fragment = resultTemplate.content.cloneNode(true);

  const characterName = pick(parsed, ["character_name", "name", "character"]) || "\u672a\u77e5\u89d2\u8272";
  const japaneseName =
    pick(parsed, ["japanese_name", "name_ja", "ja_name", "jp_name", "kana_name"]) || "\u672a\u77e5";
  const animeTitle = pick(parsed, ["anime_title", "source", "work"]) || "\u672a\u77e5\u4f5c\u54c1";
  const description =
    pick(parsed, ["description", "intro", "summary"]) ||
    (typeof rawContent === "string" ? rawContent : "\u6682\u65e0\u7b80\u4ecb\u3002");
  const storyRole = pick(parsed, ["story_role", "role_in_story", "role"]);
  const background = pick(parsed, ["background", "backstory", "history"]);
  const backgroundBlock =
    [
      storyRole ? `\u5267\u60c5\u5b9a\u4f4d\uff1a${storyRole}` : "",
      background ? `\u80cc\u666f\u7ecf\u5386\uff1a${background}` : "",
    ]
      .filter(Boolean)
      .join(" ");

  const aliases = pickArray(parsed, ["aliases", "alias", "aka"]);
  const personalityTraits = pickArray(parsed, ["personality_traits", "traits", "personality"]);
  const abilities = pickArray(parsed, ["abilities", "skills", "powers"]);
  const relationships = pickArray(parsed, ["relationships", "relations"]);
  const evidence = pickArray(parsed, ["evidence", "recognition_evidence", "reasoning_points"]);
  const signatures = pickArray(parsed, [
    "signature_lines_or_markers",
    "signature_lines",
    "signature",
    "quotes",
  ]);
  const appearance = pickArray(parsed, ["appearance_points", "appearance", "visual_features"]);

  const gender = pick(parsed, ["gender", "sex"]) || "\u672a\u77e5";
  const ageStage = pick(parsed, ["age_stage", "age_group", "age"]) || "\u672a\u77e5";
  const identity = pick(parsed, ["identity", "occupation", "position"]) || "\u672a\u77e5";
  const affiliation =
    pick(parsed, ["camp_or_affiliation", "affiliation", "faction", "organization"]) || "\u672a\u77e5";

  const extraNote = pick(parsed, ["extra_note", "note", "uncertainty"]) || "\u65e0";
  const confidence = normalizeConfidence(pick(parsed, ["confidence", "score"]));

  fragment.querySelector("#char-name").textContent = characterName;
  fragment.querySelector("#jp-name").textContent = `\u65e5\u6587\u540d\uff1a${japaneseName}`;
  fragment.querySelector("#anime-name").textContent = animeTitle;
  fragment.querySelector("#char-intro").textContent = description;
  fragment.querySelector("#char-background").textContent = backgroundBlock || "\u6682\u65e0\u80cc\u666f\u4fe1\u606f\u3002";
  fragment.querySelector("#char-extra").textContent = `\u8865\u5145\u8bf4\u660e\uff1a${extraNote}`;
  fragment.querySelector("#confidence-text").textContent = `${confidence}%`;
  fragment.querySelector("#confidence-bar").style.width = `${confidence}%`;

  renderTags(fragment.querySelector("#base-tags"), [
    aliases.length ? `\u522b\u540d\uff1a${aliases.join(" / ")}` : "",
    `\u6027\u522b\uff1a${gender}`,
    `\u5e74\u9f84\u6bb5\uff1a${ageStage}`,
    `\u8eab\u4efd\uff1a${identity}`,
    `\u9635\u8425/\u6240\u5c5e\uff1a${affiliation}`,
    appearance.length ? `\u5916\u89c2\u8981\u70b9\uff1a${appearance.slice(0, 2).join(" / ")}` : "",
  ]);
  renderList(fragment.querySelector("#traits-list"), personalityTraits, "\u6682\u65e0\u660e\u786e\u6027\u683c\u7279\u5f81\u3002");
  renderList(fragment.querySelector("#abilities-list"), abilities, "\u6682\u65e0\u660e\u786e\u80fd\u529b/\u6280\u80fd\u4fe1\u606f\u3002");
  renderList(fragment.querySelector("#relations-list"), relationships, "\u6682\u65e0\u660e\u786e\u5173\u7cfb\u4fe1\u606f\u3002");
  renderList(fragment.querySelector("#evidence-list"), evidence, "\u6682\u65e0\u660e\u786e\u8bc6\u522b\u4f9d\u636e\u3002");
  renderList(fragment.querySelector("#signature-list"), signatures, "\u6682\u65e0\u6807\u5fd7\u53f0\u8bcd\u6216\u7279\u5f81\u3002");

  resultContent.className = "";
  resultContent.innerHTML = "";
  resultContent.appendChild(fragment);
}

function showError(message) {
  setStatus("error", "\u8bc6\u522b\u5931\u8d25");
  resultContent.className = "result-empty";
  resultContent.innerHTML = `<p>${escapeHtml(message)}</p>`;
  metaInfo.textContent = "";
}

function showMeta(data) {
  const usage = data?.usage;
  const tokenText = usage
    ? `Token\u7528\u91cf\uff1a\u8f93\u5165 ${usage.prompt_tokens ?? "-"} / \u8f93\u51fa ${usage.completion_tokens ?? "-"} / \u603b\u8ba1 ${usage.total_tokens ?? "-"}`
    : "Token\u7528\u91cf\uff1a-";
  metaInfo.textContent = tokenText;
}

function pick(obj, keys) {
  const value = pickRaw(obj, keys);
  if (value === undefined || value === null) {
    return "";
  }
  if (typeof value === "string") {
    return value.trim();
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return "";
}

function pickArray(obj, keys) {
  const value = pickRaw(obj, keys);
  if (value === undefined || value === null) {
    return [];
  }

  if (Array.isArray(value)) {
    return cleanList(value.map((item) => String(item)));
  }

  if (typeof value === "string") {
    return cleanList(value.split(/[,;|/\n]+/));
  }

  if (typeof value === "object") {
    return cleanList(Object.values(value).map((item) => String(item)));
  }

  return [];
}

function pickRaw(obj, keys) {
  if (!obj || typeof obj !== "object") {
    return undefined;
  }
  for (const key of keys) {
    if (obj[key] !== undefined && obj[key] !== null) {
      return obj[key];
    }
  }
  return undefined;
}

function cleanList(items) {
  const dedup = new Set();
  const result = [];
  for (const item of items) {
    const value = String(item).trim();
    if (!value || dedup.has(value)) {
      continue;
    }
    dedup.add(value);
    result.push(value);
  }
  return result;
}

function renderTags(container, tags) {
  container.innerHTML = "";
  const available = tags.filter((tag) => String(tag).trim());
  const finalTags = available.length ? available : ["\u57fa\u7840\u4fe1\u606f\u4e0d\u8db3"];
  for (const tagText of finalTags) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = tagText;
    container.appendChild(chip);
  }
}

function renderList(container, items, fallbackText) {
  container.innerHTML = "";
  const finalItems = items.length ? items : [fallbackText];
  for (const itemText of finalItems) {
    const li = document.createElement("li");
    li.textContent = itemText;
    container.appendChild(li);
  }
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
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
