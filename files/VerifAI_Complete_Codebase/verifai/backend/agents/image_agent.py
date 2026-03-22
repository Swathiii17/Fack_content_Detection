import io
import json
import math
import base64
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from models.state import AnalysisState
from config import settings

llm = ChatAnthropic(
    model=settings.model,
    api_key=settings.anthropic_api_key,
    max_tokens=2048,
)

SYSTEM_PROMPT = """You are a specialist image forgery detection agent for investigative journalists.

You will be shown an image. Analyse it carefully for signs of manipulation, AI generation, or misrepresentation.

Evaluate:
1. VISUAL CONSISTENCY — lighting direction, shadows, perspective inconsistencies
2. EDGE ARTIFACTS — unnatural edges, halos, blurring around objects (cloning/splicing)
3. NOISE PATTERNS — inconsistent grain/noise across different regions
4. AI GENERATION SIGNS — overly smooth textures, anatomical errors (hands, eyes, teeth), background oddities
5. METADATA CONTEXT — does the image content match claimed context?
6. COMPRESSION ARTIFACTS — double-compression patterns suggest re-saving after editing
7. SEMANTIC COHERENCE — does scene make physical sense?

Respond ONLY with valid JSON:
{
  "verdict": "fake" | "likely_fake" | "uncertain" | "likely_real" | "real",
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation for the journalist",
  "signals": ["signal 1", "signal 2", ...],
  "visual_consistency_score": 0.0-1.0,
  "ai_generation_probability": 0.0-1.0,
  "manipulation_indicators": ["indicator 1", "indicator 2"]
}
"""

def read_exif_data(image_path: str) -> dict:
    """Extract EXIF metadata from image."""
    try:
        import exifread
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, stop_tag="UNDEF", details=False)
        relevant = {}
        for key in ["Image Make", "Image Model", "Image Software", "EXIF DateTimeOriginal",
                    "GPS GPSLatitude", "GPS GPSLongitude", "Image ImageDescription"]:
            if key in tags:
                relevant[key] = str(tags[key])
        return relevant
    except Exception as e:
        return {"error": str(e)}


def compute_ela_score(image_path: str, quality: int = 75) -> float:
    """
    Error Level Analysis — detects re-saved/edited regions.
    Returns a score 0-1 where higher = more suspicious.
    """
    try:
        from PIL import Image, ImageChops
        import numpy as np

        original = Image.open(image_path).convert("RGB")
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")

        diff = ImageChops.difference(original, recompressed)
        diff_array = list(diff.getdata())

        # Calculate mean absolute difference across channels
        total = sum(sum(pixel) / 3 for pixel in diff_array)
        mean_diff = total / len(diff_array)

        # Variance of pixel-level diffs (high variance = uneven compression = manipulation)
        mean_sq = sum((sum(pixel)/3 - mean_diff)**2 for pixel in diff_array) / len(diff_array)
        std_diff = math.sqrt(mean_sq)

        # Normalize: typical range is 0-30 for mean_diff
        score = min(1.0, (mean_diff / 30.0) * 0.5 + (std_diff / 20.0) * 0.5)
        return round(score, 3)
    except Exception:
        return 0.5  # Unknown


def image_to_base64(image_path: str) -> tuple[str, str]:
    """Convert image to base64 for Claude vision."""
    from PIL import Image
    img = Image.open(image_path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    # Resize if too large
    max_size = 1568
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buffer.getvalue()).decode()
    return b64, "image/jpeg"


async def image_agent_node(state: AnalysisState) -> AnalysisState:
    """Detects image manipulation using ELA + EXIF + Claude Vision."""

    if "image" not in state.get("agents_to_run", []):
        return state

    image_path = state.get("image_path")
    if not image_path or not Path(image_path).exists():
        return {**state, "image_result": {
            "verdict": "uncertain",
            "confidence": 0.0,
            "reasoning": "No image file provided or file not found.",
            "signals": [],
            "visual_consistency_score": 0.0,
            "ai_generation_probability": 0.0,
            "manipulation_indicators": [],
        }}

    exif = read_exif_data(image_path)
    ela_score = compute_ela_score(image_path)
    img_b64, media_type = image_to_base64(image_path)

    exif_summary = json.dumps(exif, indent=2) if exif else "No EXIF metadata found (stripped — suspicious)."
    ela_note = (
        f"ELA Score: {ela_score:.2f} — "
        + ("HIGH suspicion of editing" if ela_score > 0.6
           else "MODERATE — inconclusive" if ela_score > 0.35
           else "LOW suspicion of editing")
    )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=[
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": img_b64},
            },
            {
                "type": "text",
                "text": f"""Analyse this image for manipulation or AI generation.

Technical analysis already performed:
EXIF Metadata:
{exif_summary}

Error Level Analysis:
{ela_note}

Now perform your visual analysis and combine with the technical signals above.""",
            },
        ]),
    ])

    try:
        result = json.loads(response.content)
    except Exception:
        result = {
            "verdict": "uncertain",
            "confidence": 0.5,
            "reasoning": response.content,
            "signals": [],
            "visual_consistency_score": 0.5,
            "ai_generation_probability": 0.5,
            "manipulation_indicators": [],
        }

    result["agent"] = "image"
    result["ela_score"] = ela_score
    result["exif"] = exif
    return {**state, "image_result": result}
