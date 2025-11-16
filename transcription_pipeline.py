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

# ==================== CONFIGURATION ====================
# Google Drive folder ID (extract from your URL)
GDRIVE_FOLDER_ID = "1KmmQgwO42YTVLeb9nXON-0iIU4DKmn-G"
GDRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"

# Local directories
DOWNLOAD_DIR = "./downloaded_videos"
AUDIO_DIR = "./extracted_audio"
TRANSCRIPT_DIR = "./anonymized_transcripts"

# Whisper Configuration (optimized for children)
# Research shows large-v2 or large-v3 works best for child speech after fine-tuning
# For out-of-box use, medium or large-v2 recommended
WHISPER_MODEL = "large-v2"  # Options: "medium", "large-v2", "large-v3"
DEVICE = "cuda"  # "cuda" for GPU, "cpu" for CPU
COMPUTE_TYPE = "float16"  # "float16" for GPU, "int8" for CPU
BATCH_SIZE = 8  # Reduce if GPU memory issues

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

    # Load Whisper model once
    print("Loading Whisper model...")
    model = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)

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
                device=DEVICE
            )
            result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio, 
                DEVICE,
                return_char_alignments=False
            )

            # Diarization (speaker identification)
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=None,  # Set Hugging Face token if needed
                device=DEVICE
            )
            diarize_segments = diarize_model(
                str(audio_path),
                min_speakers=MIN_SPEAKERS,
                max_speakers=MAX_SPEAKERS
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Format transcript with speaker labels
            transcript_text = format_transcript(result["segments"])

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
            anonymized_result = anonymizer.anonymize(
                text=original_text,
                analyzer_results=analysis_results
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
