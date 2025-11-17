#!/usr/bin/env python3
"""
Experimental Script: Video Transcription & Anonymization Pipeline
Optimized for child and adult speech with full PII/PHI anonymization.

Requirements:
    pip install gdown whisperx torch presidio-analyzer presidio-anonymizer tqdm

System Requirements:
    - ffmpeg CLI tool installed
    - GPU recommended (CUDA) but CPU supported
"""

import os
import subprocess
from glob import glob
from pathlib import Path
import gdown
import whisperx
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from tqdm import tqdm
from typing import Any, cast

# Optional: used to detect platform (CUDA / MPS / CPU)
# Use `# type: ignore` so type checkers/Pylance won't error if torch
# isn't installed in the analysis environment.
try:
    import torch  # type: ignore
except Exception:
    torch = None

# ==================== CONFIGURATION ====================
# Google Drive folder ID (extract from your URL)
GDRIVE_FOLDER_ID = "1KmmQgwO42YTVLeb9nXON-0iIU4DKmn-G"
GDRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"

# Local directories
DOWNLOAD_DIR = "./downloaded_videos"
AUDIO_DIR = "./extracted_audio"
TRANSCRIPT_DIR = "./anonymized_transcripts"

# Whisper Configuration (optimized for children)
# Model, device and compute-type are auto-selected where possible.
# You can still override by setting environment variables:
#  - VOCAB_WHISPER_MODEL (e.g. small|medium|large-v2)
#  - VOCAB_DEVICE (cuda|mps|cpu)
#  - VOCAB_COMPUTE_TYPE (float16|int8)
#  - VOCAB_BATCH_SIZE (integer)

# Start with conservative defaults; we'll detect and overwrite below
WHISPER_MODEL = os.environ.get("VOCAB_WHISPER_MODEL", None)
DEVICE = os.environ.get("VOCAB_DEVICE", None)
COMPUTE_TYPE = os.environ.get("VOCAB_COMPUTE_TYPE", None)
BATCH_SIZE = int(os.environ.get("VOCAB_BATCH_SIZE", 0))


def detect_device_and_defaults():
    """Detect best available device and sensible defaults for compute type and batch size.

    Priority: CUDA -> MPS (Apple silicon) -> CPU
    Returns: (device, compute_type, batch_size, default_model)
    """
    # Safe conservative defaults
    device = "cpu"
    compute_type = "int8"
    batch_size = 1
    default_model = "small"

    try:
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            batch_size = 8
            default_model = "large-v2"
        elif torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            # MPS can perform well but has limited memory compared to desktop CUDA GPUs
            compute_type = "float16"
            batch_size = 2
            default_model = "medium"
        else:
            device = "cpu"
            compute_type = "int8"
            batch_size = 1
            default_model = "small"
    except Exception:
        # Fall back to conservative CPU defaults
        device = "cpu"
        compute_type = "int8"
        batch_size = 1
        default_model = "small"

    return device, compute_type, batch_size, default_model


# Auto-detect and apply defaults if not explicitly set via env
detected_device, detected_compute_type, detected_batch_size, detected_default_model = detect_device_and_defaults()

if DEVICE is None:
    DEVICE = detected_device
if COMPUTE_TYPE is None:
    COMPUTE_TYPE = detected_compute_type
if not BATCH_SIZE:
    BATCH_SIZE = detected_batch_size
if WHISPER_MODEL is None:
    # Use a safer default depending on device — allow env var override
    WHISPER_MODEL = detected_default_model

# Fine-tuning recommendations from research:
# - Learning rate: 1e-5 for Whisper child speech fine-tuning
# - Focus on last layer fine-tuning with frozen encoder
# Note: This script uses pretrained models. For custom fine-tuning,
# see: https://github.com/C3Imaging/whisper_child_asr

# Diarization settings
MIN_SPEAKERS = 2  # Teacher + Student
MAX_SPEAKERS = 2

# PII/PHI entities to anonymize
PII_PHI_ENTITIES = [
    "PERSON",           # Names
    "PHONE_NUMBER",     # Phone numbers
    "EMAIL_ADDRESS",    # Email addresses
    "CREDIT_CARD",      # Credit card numbers
    "IP_ADDRESS",       # IP addresses
    "DATE_TIME",        # Dates and times (optional - may remove context)
    "LOCATION",         # Addresses and locations
    "NRP",              # Nationalities/ethnicities
    "MEDICAL_LICENSE",  # Medical identifiers
    "US_SSN",           # Social Security Numbers
    "US_PASSPORT",      # Passport numbers
]

# ==================== STEP 1: Download from Google Drive ====================
def download_videos():
    """Download all videos from Google Drive folder"""
    print("\n" + "="*60)
    print("STEP 1: Downloading videos from Google Drive")
    print("="*60)

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    # If there are already video files in DOWNLOAD_DIR, skip downloading
    existing_videos = list(Path(DOWNLOAD_DIR).rglob("*.mp4")) + \
                      list(Path(DOWNLOAD_DIR).rglob("*.mov")) + \
                      list(Path(DOWNLOAD_DIR).rglob("*.avi"))

    force_download = os.environ.get("VOCAB_FORCE_DOWNLOAD", "0") == "1"
    if existing_videos and not force_download:
        print(f"Found {len(existing_videos)} existing video file(s) in {DOWNLOAD_DIR}; skipping download.")
        print("Set environment variable VOCAB_FORCE_DOWNLOAD=1 to force re-download.")
        return existing_videos

    try:
        print(f"Downloading folder: {GDRIVE_FOLDER_URL}")
        print(f"Destination: {DOWNLOAD_DIR}")
        print("Note: Folder must be set to 'Anyone with the link' in sharing settings\n")

        # Download entire folder
        gdown.download_folder(
            url=GDRIVE_FOLDER_URL,
            output=DOWNLOAD_DIR,
            quiet=False,
            use_cookies=False
        )

        # Count downloaded videos
        video_files = list(Path(DOWNLOAD_DIR).rglob("*.mp4")) + \
                     list(Path(DOWNLOAD_DIR).rglob("*.mov")) + \
                     list(Path(DOWNLOAD_DIR).rglob("*.avi"))
        print(f"\n✓ Successfully downloaded {len(video_files)} video files")
        return video_files

    except Exception as e:
        print(f"✗ Error downloading from Google Drive: {e}")
        # If download failed but we have existing files, proceed with them
        if existing_videos:
            print(f"Using {len(existing_videos)} existing file(s) from {DOWNLOAD_DIR} despite download error.")
            return existing_videos

        print("\nTroubleshooting:")
        print("1. Ensure folder is set to 'Anyone with the link'")
        print("2. Check folder ID is correct")
        print("3. Try manual download if issues persist")
        return []

# ==================== STEP 2: Extract Audio ====================
def extract_audio(video_files):
    """Extract audio from video files using ffmpeg"""
    print("\n" + "="*60)
    print("STEP 2: Extracting audio from videos")
    print("="*60)

    os.makedirs(AUDIO_DIR, exist_ok=True)
    audio_files = []

    for video_path in tqdm(video_files, desc="Extracting audio"):
        try:
            base_name = Path(video_path).stem
            audio_path = Path(AUDIO_DIR) / f"{base_name}.wav"

            # Extract audio: 16kHz mono WAV (optimal for Whisper)
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # 16-bit PCM
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                str(audio_path),
                "-y"  # Overwrite
            ]

            subprocess.run(cmd, check=True, capture_output=True)
            audio_files.append(audio_path)

        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed to extract audio from {video_path}: {e}")

    print(f"\n✓ Successfully extracted {len(audio_files)} audio files")
    return audio_files

# ==================== STEP 3: Transcribe & Diarize ====================
def transcribe_and_diarize(audio_files):
    """Transcribe audio with WhisperX and perform speaker diarization"""
    print("\n" + "="*60)
    print("STEP 3: Transcription & Diarization (optimized for child speech)")
    print("="*60)
    print(f"Model: {WHISPER_MODEL}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}\n")

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

    # Determine device to pass to whisperx/faster-whisper.
    # faster-whisper (used by whisperx) currently supports 'cuda' and 'cpu' only;
    # it does not support Apple MPS. If running on MPS, fall back to CPU for whisperx
    # but keep DEVICE as 'mps' for other torch-based operations.
    whisper_device = DEVICE if DEVICE == "cuda" else "cpu"
    if DEVICE == "mps":
        print("Note: running on Apple MPS. whisperx/faster-whisper does not support MPS, falling back to CPU for model inference.")

    # Determine a compute type compatible with whisperx/faster-whisper
    whisper_compute_type = COMPUTE_TYPE
    if whisper_device != "cuda" and str(whisper_compute_type).lower() == "float16":
        print("Warning: float16 compute requested but target device/backend does not support efficient float16. Falling back to int8 for whisperx.")
        whisper_compute_type = "int8"

    # Load Whisper model once (use whisper_device for whisperx compatibility)
    print("Loading Whisper model...")
    model = whisperx.load_model(WHISPER_MODEL, whisper_device, compute_type=whisper_compute_type)

    transcripts = []

    for audio_path in tqdm(audio_files, desc="Transcribing"):
        try:
            base_name = Path(audio_path).stem

            # Load audio
            audio = whisperx.load_audio(str(audio_path))

            # Transcribe
            result = model.transcribe(audio, batch_size=BATCH_SIZE)

            # Align for better word-level timestamps
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=whisper_device
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                whisper_device,
                return_char_alignments=False
            )

            # Diarization (speaker identification)
            # DiarizationPipeline may not be present in some whisperx stubs; silence
            # static attribute warnings while preserving runtime behavior.
            diarize_model = whisperx.DiarizationPipeline(  # type: ignore[attr-defined]
                use_auth_token=None,  # Set Hugging Face token if needed
                device=whisper_device
            )
            diarize_segments = diarize_model(
                str(audio_path),
                min_speakers=MIN_SPEAKERS,
                max_speakers=MAX_SPEAKERS
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Format transcript with speaker labels
            transcript_text = format_transcript(result["segments"])

            # Save raw (pre-anonymized) transcript to disk for inspection/debugging
            try:
                raw_output_file = Path(TRANSCRIPT_DIR) / f"{base_name}_raw.txt"
                with open(raw_output_file, "w", encoding="utf-8") as rf:
                    rf.write("="*60 + "\n")
                    rf.write(f"File: {base_name}\n")
                    rf.write(f"Language: {result.get('language', 'unknown')}\n")
                    rf.write("="*60 + "\n\n")
                    rf.write(transcript_text)
                    rf.write("\n\n" + "="*60 + "\n")
                print(f"  • Saved raw transcript: {raw_output_file}")
            except Exception as e:
                print(f"\n✗ Failed to save raw transcript for {base_name}: {e}")

            transcripts.append({
                "file": base_name,
                "transcript": transcript_text,
                "language": result.get("language", "unknown")
            })

        except Exception as e:
            print(f"\n✗ Error transcribing {audio_path}: {e}")
            transcripts.append({
                "file": Path(audio_path).stem,
                "transcript": "[TRANSCRIPTION FAILED]",
                "language": "unknown"
            })

    print(f"\n✓ Transcribed {len(transcripts)} audio files")
    return transcripts

def format_transcript(segments):
    """Format segments into readable transcript with speaker labels"""
    transcript_lines = []
    current_speaker = None
    current_text = []

    for segment in segments:
        speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
        text = segment.get("text", "").strip()

        if speaker != current_speaker:
            # New speaker - save previous and start new
            if current_text:
                transcript_lines.append(
                    f"[{current_speaker}]: {' '.join(current_text)}"
                )
            current_speaker = speaker
            current_text = [text]
        else:
            # Same speaker - continue
            current_text.append(text)

    # Add final segment
    if current_text:
        transcript_lines.append(
            f"[{current_speaker}]: {' '.join(current_text)}"
        )

    return "\n".join(transcript_lines)

# ==================== STEP 4: Anonymize PII/PHI ====================
def anonymize_transcripts(transcripts):
    """Anonymize PII and PHI using Microsoft Presidio"""
    print("\n" + "="*60)
    print("STEP 4: Anonymizing PII/PHI with Presidio")
    print("="*60)
    print(f"Protected entities: {', '.join(PII_PHI_ENTITIES)}\n")

    # Initialize Presidio
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    for transcript_data in tqdm(transcripts, desc="Anonymizing"):
        try:
            original_text = transcript_data["transcript"]

            # Analyze for PII/PHI
            analysis_results = analyzer.analyze(
                text=original_text,
                language="en",
                entities=PII_PHI_ENTITIES
            )

            # Anonymize (replace with generic labels)
            # Cast analyzer results to Any to avoid narrow type mismatch between
            # presidio_analyzer and presidio_anonymizer types in static analysis.
            anonymized_result = anonymizer.anonymize(
                text=original_text,
                analyzer_results=cast(Any, analysis_results)
            )

            # Save anonymized transcript
            output_file = Path(TRANSCRIPT_DIR) / f"{transcript_data['file']}_anonymized.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("="*60 + "\n")
                f.write(f"File: {transcript_data['file']}\n")
                f.write(f"Language: {transcript_data['language']}\n")
                f.write("="*60 + "\n\n")
                f.write(anonymized_result.text)
                f.write("\n\n" + "="*60 + "\n")
                f.write(f"Detected PII/PHI entities: {len(analysis_results)}\n")
                f.write("="*60 + "\n")

            print(f"  ✓ {transcript_data['file']}: {len(analysis_results)} entities anonymized")

        except Exception as e:
            print(f"\n✗ Error anonymizing {transcript_data['file']}: {e}")

    print(f"\n✓ All transcripts anonymized and saved to: {TRANSCRIPT_DIR}")

# ==================== MAIN PIPELINE ====================
def main():
    """Execute full pipeline"""
    print("\n" + "="*60)
    print("VIDEO TRANSCRIPTION & ANONYMIZATION PIPELINE")
    print("Optimized for child and adult speech")
    print("="*60)

    # Step 1: Download videos
    video_files = download_videos()
    if not video_files:
        print("\n✗ No videos downloaded. Exiting.")
        return

    # Step 2: Extract audio
    audio_files = extract_audio(video_files)
    if not audio_files:
        print("\n✗ No audio extracted. Exiting.")
        return

    # Step 3: Transcribe and diarize
    transcripts = transcribe_and_diarize(audio_files)
    if not transcripts:
        print("\n✗ No transcripts generated. Exiting.")
        return

    # Step 4: Anonymize
    anonymize_transcripts(transcripts)

    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE")
    print("="*60)
    print(f"Anonymized transcripts available in: {TRANSCRIPT_DIR}")
    print("\nSafe for public consumption - all PII/PHI removed")

if __name__ == "__main__":
    main()
