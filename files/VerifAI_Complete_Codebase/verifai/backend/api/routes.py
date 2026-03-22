from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
import aiofiles
import tempfile
import os
import uuid
from pathlib import Path

from pipeline import verifai_graph
from models.database import Analysis, get_db
from models.state import AnalysisState

router = APIRouter(prefix="/api", tags=["analysis"])

UPLOAD_DIR = Path(tempfile.gettempdir()) / "verifai_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/ogg", "audio/mp4", "audio/flac"}


@router.post("/analyse/text")
async def analyse_text(
    text: str = Form(...),
    content_type: str = Form("news"),
    url: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Analyse text content (news article, review, or URL)."""
    if not text and not url:
        raise HTTPException(400, "Provide either text or URL")

    initial_state: AnalysisState = {
        "content_type": content_type,
        "raw_text": text or "",
        "url": url,
        "image_path": None,
        "audio_path": None,
        "filename": None,
        "news_result": None,
        "review_result": None,
        "image_result": None,
        "audio_result": None,
        "agents_to_run": [],
        "final_verdict": None,
        "final_confidence": None,
        "final_reasoning": None,
        "all_signals": [],
        "error": None,
    }

    result = await verifai_graph.ainvoke(initial_state)
    return await _save_and_return(result, content_type, text[:100] if text else url, db)


@router.post("/analyse/image")
async def analyse_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Analyse an image file for forgery or AI generation."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, f"Unsupported image type: {file.content_type}")

    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    try:
        initial_state: AnalysisState = {
            "content_type": "image",
            "raw_text": None,
            "url": None,
            "image_path": str(file_path),
            "audio_path": None,
            "filename": file.filename,
            "news_result": None,
            "review_result": None,
            "image_result": None,
            "audio_result": None,
            "agents_to_run": ["image"],
            "final_verdict": None,
            "final_confidence": None,
            "final_reasoning": None,
            "all_signals": [],
            "error": None,
        }

        result = await verifai_graph.ainvoke(initial_state)
        return await _save_and_return(result, "image", file.filename, db)
    finally:
        if file_path.exists():
            os.unlink(file_path)


@router.post("/analyse/audio")
async def analyse_audio(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Analyse an audio file for deepfake or manipulation."""
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(400, f"Unsupported audio type: {file.content_type}")

    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    try:
        initial_state: AnalysisState = {
            "content_type": "audio",
            "raw_text": None,
            "url": None,
            "image_path": None,
            "audio_path": str(file_path),
            "filename": file.filename,
            "news_result": None,
            "review_result": None,
            "image_result": None,
            "audio_result": None,
            "agents_to_run": ["audio"],
            "final_verdict": None,
            "final_confidence": None,
            "final_reasoning": None,
            "all_signals": [],
            "error": None,
        }

        result = await verifai_graph.ainvoke(initial_state)
        return await _save_and_return(result, "audio", file.filename, db)
    finally:
        if file_path.exists():
            os.unlink(file_path)


@router.get("/analyses")
async def list_analyses(limit: int = 20, db: AsyncSession = Depends(get_db)):
    """Get recent analyses."""
    stmt = select(Analysis).order_by(desc(Analysis.created_at)).limit(limit)
    rows = await db.execute(stmt)
    analyses = rows.scalars().all()
    return [
        {
            "id": a.id,
            "content_type": a.content_type,
            "input_summary": a.input_summary,
            "final_verdict": a.final_verdict,
            "final_confidence": a.final_confidence,
            "created_at": a.created_at.isoformat(),
        }
        for a in analyses
    ]


@router.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """Get full analysis details."""
    row = await db.get(Analysis, analysis_id)
    if not row:
        raise HTTPException(404, "Analysis not found")
    return {
        "id": row.id,
        "content_type": row.content_type,
        "input_summary": row.input_summary,
        "final_verdict": row.final_verdict,
        "final_confidence": row.final_confidence,
        "final_reasoning": row.final_reasoning,
        "all_signals": row.all_signals,
        "agent_results": row.agent_results,
        "created_at": row.created_at.isoformat(),
    }


async def _save_and_return(result: dict, content_type: str, input_summary: str, db: AsyncSession):
    """Save analysis to DB and return response."""
    agent_results = {}
    for key in ["news_result", "review_result", "image_result", "audio_result"]:
        if result.get(key):
            agent_results[key.replace("_result", "")] = result[key]

    analysis = Analysis(
        content_type=content_type,
        input_summary=input_summary[:200] if input_summary else "",
        final_verdict=result.get("final_verdict", "uncertain"),
        final_confidence=result.get("final_confidence", 0.5),
        final_reasoning=result.get("final_reasoning", ""),
        all_signals=result.get("all_signals", []),
        agent_results=agent_results,
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    return {
        "id": analysis.id,
        "content_type": content_type,
        "final_verdict": analysis.final_verdict,
        "final_confidence": analysis.final_confidence,
        "final_reasoning": analysis.final_reasoning,
        "all_signals": analysis.all_signals,
        "agent_results": agent_results,
        "created_at": analysis.created_at.isoformat(),
    }
