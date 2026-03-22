import json
import os
import subprocess
import tempfile
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

SYSTEM_PROMPT = """You are a specialist video deepfake detection agent for investigative journalists.

You receive frame-level image analysis results and audio analysis from a video file.
Combine these signals to determine if the video is authentic, manipulated, or AI-generated.

Evaluate:
1. VISUAL CONSISTENCY ACROSS FRAMES — flickering faces, texture instability, temporal artifacts
2. AUDIO-VISUAL SYNC — lip movement vs audio alignment (deepfakes often desync subtly)
3. FACIAL ANOMALIES — blurring around hairline/ears, unnatural eye blinking rate
4. LIGHTING CONSISTENCY — light direction changing between frames
5. BACKGROUND STABILITY — background warping or inconsistency around subject
6. COMPRESSION PATTERNS — re-encoding artifacts from deepfake pipelines
7. AUDIO DEEPFAKE SIGNALS — voice synthesis indicators from audio track

Respond ONLY with valid JSON:
{
  "verdict": "fake" | "likely_fake" | "uncertain" | "likely_real" | "real",
  "confidence": 0.0-1.0,
  "reasoning": "clear explanation for the journalist",
  "signals": ["signal 1", "signal 2", ...],
  "deepfake_probability": 0.0-1.0,
  "av_sync_score": 0.0-1.0,
  "temporal_consistency_score": 0.0-1.0,
  "facial_anomaly_score": 0.0-1.0
}
"""


def extract_frames(video_path: str, num_frames: int = 8) -> list[str]:
    """
    Extract evenly-spaced frames from video using ffmpeg.
    Returns list of temp file paths.
    """
    frame_paths = []
    try:
        # Get video duration
        probe = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", video_path
        ], capture_output=True, text=True, timeout=30)

        duration = 10.0  # fallback
        try:
            info = json.loads(probe.stdout)
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video":
                    duration = float(stream.get("duration", 10))
                    break
        except Exception:
            pass

        interval = duration / (num_frames + 1)
        tmp_dir = tempfile.mkdtemp()

        for i in range(num_frames):
            timestamp = interval * (i + 1)
            frame_path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
            result = subprocess.run([
                "ffmpeg", "-ss", str(timestamp),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                "-y", frame_path
            ], capture_output=True, timeout=15)

            if result.returncode == 0 and Path(frame_path).exists():
                frame_paths.append(frame_path)

    except FileNotFoundError:
        pass  # ffmpeg not installed — handled gracefully
    except Exception:
        pass

    return frame_paths


def extract_audio_from_video(video_path: str) -> str | None:
    """Extract audio track from video as a WAV file."""
    try:
        tmp_audio = tempfile.mktemp(suffix=".wav")
        result = subprocess.run([
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", tmp_audio
        ], capture_output=True, timeout=30)

        if result.returncode == 0 and Path(tmp_audio).exists():
            return tmp_audio
    except Exception:
        pass
    return None


def analyse_frames_with_vision(frame_paths: list[str]) -> dict:
    """
    Use PIL to compute basic inter-frame consistency metrics.
    Returns heuristic signals without needing Claude Vision on every frame.
    """
    if not frame_paths:
        return {"error": "No frames extracted — ffmpeg may not be installed"}

    try:
        from PIL import Image
        import numpy as np

        arrays = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB").resize((224, 224))
            arrays.append(list(img.getdata()))

        if len(arrays) < 2:
            return {"frames_analysed": len(arrays), "error": "Too few frames"}

        # Compute mean absolute difference between consecutive frames
        diffs = []
        for i in range(len(arrays) - 1):
            frame_diff = sum(
                abs(arrays[i][j][c] - arrays[i+1][j][c])
                for j in range(len(arrays[i]))
                for c in range(3)
            ) / (len(arrays[i]) * 3)
            diffs.append(frame_diff)

        mean_diff = sum(diffs) / len(diffs)
        variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
        std_diff = variance ** 0.5

        # Deepfakes often have unusually low inter-frame variance (too smooth)
        # or unusually high spikes (temporal artifacts)
        temporal_anomaly = std_diff / (mean_diff + 1e-6)

        return {
            "frames_analysed": len(frame_paths),
            "mean_inter_frame_diff": round(mean_diff, 3),
            "std_inter_frame_diff": round(std_diff, 3),
            "temporal_anomaly_ratio": round(temporal_anomaly, 3),
            "interpretation": (
                "High temporal instability — possible deepfake artifacts" if temporal_anomaly > 0.5
                else "Unnaturally smooth — possible GAN generation" if temporal_anomaly < 0.05
                else "Normal temporal consistency"
            )
        }
    except ImportError:
        return {"error": "Pillow not installed"}
    except Exception as e:
        return {"error": str(e)}


def get_audio_features(audio_path: str) -> dict:
    """Reuse librosa analysis from audio agent."""
    try:
        import librosa

        y, sr = librosa.load(audio_path, sr=None, mono=True, duration=60)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        pitch_mean = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y).mean())
        rms = librosa.feature.rms(y=y)[0]
        rms_std = float(rms.std())
        rms_mean = float(rms.mean())

        return {
            "pitch_centroid_hz": round(pitch_mean, 1),
            "zero_crossing_rate": round(zcr, 4),
            "rms_uniformity": round(rms_std / (rms_mean + 1e-6), 3),
            "mfcc_variance": round(float(mfccs.var()), 3),
        }
    except Exception as e:
        return {"error": str(e)}


async def video_agent_node(state: AnalysisState) -> AnalysisState:
    """Detects video deepfakes using frame analysis + audio extraction + LLM."""

    if "video" not in state.get("agents_to_run", []):
        return state

    video_path = state.get("video_path")
    if not video_path or not Path(video_path).exists():
        return {**state, "video_result": {
            "verdict": "uncertain",
            "confidence": 0.0,
            "reasoning": "No video file provided or file not found.",
            "signals": [],
            "deepfake_probability": 0.0,
            "av_sync_score": 0.5,
            "temporal_consistency_score": 0.0,
            "facial_anomaly_score": 0.0,
        }}

    # Step 1: Extract frames
    frame_paths = extract_frames(video_path, num_frames=8)

    # Step 2: Analyse frame consistency
    frame_analysis = analyse_frames_with_vision(frame_paths)

    # Step 3: Extract and analyse audio
    audio_path = extract_audio_from_video(video_path)
    audio_features = get_audio_features(audio_path) if audio_path else {"error": "No audio track"}

    # Step 4: Clean up temp files
    for fp in frame_paths:
        try:
            os.unlink(fp)
        except Exception:
            pass
    if audio_path:
        try:
            os.unlink(audio_path)
        except Exception:
            pass

    # Step 5: LLM synthesis
    user_msg = f"""Video file: {state.get('filename', 'unknown')}

Frame analysis ({frame_analysis.get('frames_analysed', 0)} frames extracted):
{json.dumps(frame_analysis, indent=2)}

Audio track analysis:
{json.dumps(audio_features, indent=2)}

Note: ffmpeg {"is" if frame_paths else "is NOT"} available on this system.
{"Frame-level visual analysis was performed." if frame_paths else "No frames could be extracted — base your analysis on audio signals only and flag the limitation."}

Based on these signals, assess whether this video is authentic or a deepfake."""

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    try:
        result = json.loads(response.content)
    except Exception:
        result = {
            "verdict": "uncertain",
            "confidence": 0.4,
            "reasoning": response.content,
            "signals": [],
            "deepfake_probability": 0.5,
            "av_sync_score": 0.5,
            "temporal_consistency_score": 0.5,
            "facial_anomaly_score": 0.5,
        }

    result["agent"] = "video"
    result["frame_analysis"] = frame_analysis
    result["audio_features"] = audio_features
    return {**state, "video_result": result}
