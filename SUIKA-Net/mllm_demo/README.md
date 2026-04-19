# Anime Character Retrieval (Qwen + Bailian)

Pure frontend demo: upload one anime character image, call Qwen vision model on Bailian, and render a character profile.

## Fixed Parameters (Hidden from UI)

- Region: China Mainland (Beijing)
- Endpoint: `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
- Model: `qwen3.6-plus`
- API Key: hardcoded in `app.js` and not shown on the page

## Features

- Image upload and preview (PNG/JPG/WEBP)
- Detailed character profile card:
  - basic tags (aliases, identity, affiliation, visual cues)
  - long overview, story role and background
  - personality, abilities, relationships, recognition evidence
  - confidence and uncertainty note
- Error handling for file type/size and request failures

## Run

1. Start a static server in this folder:
   - `python -m http.server 8080`
   - or `npx serve .`
2. Open:
   - `http://localhost:8080`
3. Upload an image and click `Analyze`.

## Notes

- Hardcoding API keys in frontend code is not secure and is only suitable for local testing.
- For production, move the key to a backend proxy.
