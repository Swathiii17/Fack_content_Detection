# ─────────────────────────────────────────────────────────────────
# ADD THIS TO backend/api/routes.py
# ─────────────────────────────────────────────────────────────────
# 1. Add to ALLOWED types at top of routes.py:
ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/webm", "video/quicktime",
    "video/x-msvideo", "video/x-matroska"
}

# 2. Add this endpoint:

from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from models.database import get_db
import aiofiles, uuid, os
from pathlib import Path

router = APIRouter(prefix="/api", tags=["analysis"])
UPLOAD_DIR = Path("/tmp/verifai_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


async def analyse_video(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Analyse a video file for deepfake manipulation."""
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        from fastapi import HTTPException
        raise HTTPException(400, f"Unsupported video type: {file.content_type}")

    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    try:
        from models.state import AnalysisState
        from pipeline import verifai_graph

        initial_state: AnalysisState = {
            "content_type": "video",
            "raw_text": None,
            "url": None,
            "image_path": None,
            "audio_path": None,
            "video_path": str(file_path),       # NEW field — add to state.py too
            "filename": file.filename,
            "news_result": None,
            "review_result": None,
            "image_result": None,
            "audio_result": None,
            "video_result": None,               # NEW field
            "agents_to_run": ["video"],
            "final_verdict": None,
            "final_confidence": None,
            "final_reasoning": None,
            "all_signals": [],
            "error": None,
        }

        result = await verifai_graph.ainvoke(initial_state)
        return await _save_and_return(result, "video", file.filename, db)
    finally:
        if file_path.exists():
            os.unlink(file_path)


# Register it:
# router.add_api_route("/analyse/video", analyse_video, methods=["POST"])
