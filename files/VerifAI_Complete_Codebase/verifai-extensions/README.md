# VerifAI Extensions

Two new features added to VerifAI:
1. **Video deepfake detection agent** — frame analysis + audio extraction
2. **Browser extension** — right-click anything on the web to check it

---

## 1. Video deepfake detection

### How to integrate

**Step 1 — Copy the video agent into your backend:**
```bash
cp verifai-extensions/video_agent/video_agent.py verifai/backend/agents/video_agent.py
```

**Step 2 — Replace state.py with the updated version (adds video_path + video_result):**
```bash
cp verifai-extensions/video_agent/state_updated.py verifai/backend/models/state.py
```

**Step 3 — Replace pipeline.py with the updated version (adds video agent node):**
```bash
cp verifai-extensions/video_agent/pipeline_updated.py verifai/backend/pipeline.py
```

**Step 4 — Add the video route to routes.py:**
Open `verifai/backend/api/routes.py` and add:
```python
from agents.video_agent import video_agent_node

ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/webm", "video/quicktime",
    "video/x-msvideo", "video/x-matroska"
}

@router.post("/analyse/video")
async def analyse_video(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(400, f"Unsupported video type: {file.content_type}")

    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await file.read())

    try:
        initial_state = {
            "content_type": "video",
            "raw_text": None, "url": None,
            "image_path": None, "audio_path": None,
            "video_path": str(file_path),
            "filename": file.filename,
            "news_result": None, "review_result": None,
            "image_result": None, "audio_result": None, "video_result": None,
            "agents_to_run": ["video"],
            "final_verdict": None, "final_confidence": None,
            "final_reasoning": None, "all_signals": [], "error": None,
        }
        result = await verifai_graph.ainvoke(initial_state)
        return await _save_and_return(result, "video", file.filename, db)
    finally:
        if file_path.exists():
            os.unlink(file_path)
```

**Step 5 — Install ffmpeg (required for frame extraction):**
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
choco install ffmpeg
```

**Step 6 — Add video tab to frontend App.jsx:**
In the tab-row, add `"video"` alongside `"text"`, `"image"`, `"audio"`.
In the file input section, add:
```jsx
{tab === "video" && (
  <div className="upload-zone" onClick={() => fileRef.current?.click()}>
    <div>{file ? file.name : "Drop video file here — MP4, WebM, MOV"}</div>
    <input ref={fileRef} type="file" style={{ display: "none" }}
      accept="video/*" onChange={e => setFile(e.target.files[0])} />
  </div>
)}
```
And add the submit handler:
```jsx
} else if (tab === "video") {
  result = await analyseVideo(file);   // add analyseVideo to api.js
}
```

Add to `api.js`:
```js
export async function analyseVideo(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/analyse/video", form);
  return data;
}
```

### How it works

1. **Frame extraction** — ffmpeg extracts 8 evenly-spaced frames from the video
2. **Temporal consistency** — PIL computes inter-frame pixel differences to detect flickering or GAN artifacts
3. **Audio extraction** — ffmpeg pulls the audio track as WAV
4. **Audio analysis** — librosa spectral analysis (same as audio agent)
5. **LLM synthesis** — Claude combines frame + audio signals into a final verdict

### What it detects
- Face-swap deepfakes (FaceSwap, DeepFaceLab)
- Lip-sync manipulation (wav2lip-style)
- GAN-generated video (temporal flickering, texture instability)
- Spliced audio tracks (AV desync)
- Re-encoded/re-saved manipulations (compression artifact patterns)

---

## 2. Browser Extension

### Installation

**Step 1 — Open Chrome extension manager:**
Go to `chrome://extensions` in your browser.

**Step 2 — Enable Developer Mode:**
Toggle "Developer mode" in the top-right corner.

**Step 3 — Load the extension:**
Click "Load unpacked" → select the `verifai-extensions/browser_extension/` folder.

**Step 4 — Make sure VerifAI backend is running:**
```bash
cd verifai/backend
uvicorn main:app --port 8000
```

### How to use it

**Right-click menu:**
- Right-click any text → "Check selected text with VerifAI"
- Right-click a page → "Check this page with VerifAI"
- Right-click an image → "Check this image with VerifAI"
- Right-click a link → "Check this link with VerifAI"

**Keyboard shortcut:**
- Press `Alt+V` on any page to instantly analyse the current URL

**Extension popup:**
- Click the VerifAI icon in your toolbar
- Paste content directly and choose News or Review type
- See your last result at a glance
- Open the full dashboard

**Floating panel:**
Results appear as a floating panel in the bottom-right corner of any page. You can dismiss it with ✕ or click "Open full report" to go to the dashboard.

### Extension file structure
```
browser_extension/
├── manifest.json          # Extension config (Manifest V3)
├── icons/                 # icon16.png, icon48.png, icon128.png (add your own)
└── src/
    ├── background.js      # Service worker — context menus, API calls
    ├── content.js         # Injected into every page — floating panel
    ├── content.css        # Floating panel styles
    ├── popup.html         # Toolbar popup UI
    └── popup.js           # Popup logic
```

### Adding icons
The extension needs PNG icons at 3 sizes. You can generate them from any image:
```bash
# Using ImageMagick
convert your-logo.png -resize 16x16 browser_extension/icons/icon16.png
convert your-logo.png -resize 48x48 browser_extension/icons/icon48.png
convert your-logo.png -resize 128x128 browser_extension/icons/icon128.png
```
Or use any online PNG resizer.

---

## Verdict scale (both features use the same system)

| Verdict       | Meaning                          |
|---------------|----------------------------------|
| `fake`        | Strong evidence of fabrication   |
| `likely_fake` | Multiple suspicious signals      |
| `uncertain`   | Inconclusive — needs more review |
| `likely_real` | Mostly credible signals          |
| `real`        | Strong authenticity indicators   |
