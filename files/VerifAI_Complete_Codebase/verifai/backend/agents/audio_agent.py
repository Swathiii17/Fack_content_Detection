import json
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

SYSTEM_PROMPT = """You are a specialist audio deepfake detection agent for investigative journalists.

Based on the spectral and acoustic features extracted from an audio file, assess whether the audio is authentic or manipulated/synthesised.

Evaluate:
1. SPECTRAL CONSISTENCY — natural audio has organic spectral variation; TTS/deepfakes often have unnatural uniformity
2. FUNDAMENTAL FREQUENCY — pitch variation patterns in authentic vs synthetic speech
3. SILENCE PATTERNS — deepfakes often have unusual silence distribution
4. FORMANT TRANSITIONS — how vowel sounds transition (deepfakes can sound "smooth" in unnatural ways)
5. BACKGROUND NOISE — authentic recordings have consistent environmental noise; spliced audio has inconsistencies
6. ZERO CROSSING RATE — high ZCR with unusual patterns can indicate synthesis
7. SPECTRAL ROLLOFF — energy distribution across frequencies

Respond ONLY with valid JSON:
{
  "verdict": "fake" | "likely_fake" | "uncertain" | "likely_real" | "real",
  "confidence": 0.0-1.0,
  "reasoning": "explanation for the journalist",
  "signals": ["signal 1", "signal 2", ...],
  "deepfake_probability": 0.0-1.0,
  "voice_consistency_score": 0.0-1.0,
  "spectral_anomaly_score": 0.0-1.0
}
"""

def extract_audio_features(audio_path: str) -> dict:
    """Extract spectral features using librosa."""
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(audio_path, sr=None, mono=True, duration=60)
        duration = librosa.get_duration(y=y, sr=sr)

        # MFCCs — phonetic fingerprint
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfccs.mean(axis=1).tolist()
        mfcc_std = mfccs.std(axis=1).tolist()

        # Spectral features
        spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        spectral_rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
        spectral_bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
        zero_crossing_rate = float(librosa.feature.zero_crossing_rate(y).mean())

        # Fundamental frequency (pitch) via piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > magnitudes.mean()]
        pitch_mean = float(pitch_values.mean()) if len(pitch_values) > 0 else 0
        pitch_std = float(pitch_values.std()) if len(pitch_values) > 0 else 0

        # RMS energy (volume consistency)
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(rms.mean())
        rms_std = float(rms.std())

        # Silence ratio
        silence_threshold = rms_mean * 0.1
        silence_frames = (rms < silence_threshold).sum()
        silence_ratio = float(silence_frames / len(rms))

        return {
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "mfcc_means": [round(v, 3) for v in mfcc_mean[:5]],
            "mfcc_stds": [round(v, 3) for v in mfcc_std[:5]],
            "spectral_centroid_hz": round(spectral_centroid, 1),
            "spectral_rolloff_hz": round(spectral_rolloff, 1),
            "spectral_bandwidth_hz": round(spectral_bandwidth, 1),
            "zero_crossing_rate": round(zero_crossing_rate, 4),
            "pitch_mean_hz": round(pitch_mean, 1),
            "pitch_std_hz": round(pitch_std, 1),
            "rms_energy_mean": round(rms_mean, 4),
            "rms_energy_std": round(rms_std, 4),
            "silence_ratio": round(silence_ratio, 3),
        }
    except ImportError:
        return {"error": "librosa not installed"}
    except Exception as e:
        return {"error": str(e)}


def interpret_features(features: dict) -> list[str]:
    """Generate human-readable interpretations of audio features."""
    signals = []
    if "error" in features:
        return [f"Feature extraction failed: {features['error']}"]

    # Pitch variation — low std often indicates TTS
    if features.get("pitch_std_hz", 100) < 20:
        signals.append("Very low pitch variation — consistent with text-to-speech synthesis")
    elif features.get("pitch_std_hz", 0) > 80:
        signals.append("High pitch variation — consistent with natural speech emotion")

    # ZCR — synthetic speech can have unnaturally high ZCR
    zcr = features.get("zero_crossing_rate", 0)
    if zcr > 0.15:
        signals.append(f"High zero-crossing rate ({zcr:.3f}) — possible synthetic consonants")

    # Silence ratio
    silence = features.get("silence_ratio", 0)
    if silence > 0.4:
        signals.append(f"High silence ratio ({silence:.1%}) — may indicate edited or spliced audio")
    elif silence < 0.05:
        signals.append("Very little silence — atypical for natural conversational speech")

    # RMS consistency
    rms_mean = features.get("rms_energy_mean", 0)
    rms_std = features.get("rms_energy_std", 0)
    if rms_mean > 0 and (rms_std / rms_mean) < 0.2:
        signals.append("Unusually uniform volume — may indicate TTS or heavily processed audio")

    return signals if signals else ["No strong anomaly signals detected in acoustic features"]


async def audio_agent_node(state: AnalysisState) -> AnalysisState:
    """Detects audio deepfakes using spectral analysis + LLM."""

    if "audio" not in state.get("agents_to_run", []):
        return state

    audio_path = state.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        return {**state, "audio_result": {
            "verdict": "uncertain",
            "confidence": 0.0,
            "reasoning": "No audio file provided or file not found.",
            "signals": [],
            "deepfake_probability": 0.0,
            "voice_consistency_score": 0.0,
            "spectral_anomaly_score": 0.0,
        }}

    features = extract_audio_features(audio_path)
    interpretations = interpret_features(features)

    user_msg = f"""Acoustic features extracted from the audio file:

{json.dumps(features, indent=2)}

Pre-interpreted signals:
{chr(10).join(f'- {s}' for s in interpretations)}

Filename: {state.get('filename', 'unknown')}

Based on these spectral and acoustic features, assess whether this audio is authentic or a deepfake/synthetic voice."""

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    try:
        result = json.loads(response.content)
    except Exception:
        result = {
            "verdict": "uncertain",
            "confidence": 0.5,
            "reasoning": response.content,
            "signals": interpretations,
            "deepfake_probability": 0.5,
            "voice_consistency_score": 0.5,
            "spectral_anomaly_score": 0.5,
        }

    result["agent"] = "audio"
    result["features"] = features
    return {**state, "audio_result": result}
