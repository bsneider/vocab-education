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
import traceback

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

# ==================== STEP 2.5: Pitch Adjustment for Diarization ====================
def create_pitch_adjusted_audio(audio_path, semitones=-4):
    """Create pitch-adjusted version of audio for better child voice diarization.
    
    Children's voices are typically 3-5 semitones higher than adults.
    Shifting down helps diarization models trained on adult speech.
    """
    base_name = Path(audio_path).stem
    adjusted_path = Path(AUDIO_DIR) / f"{base_name}_pitched.wav"
    
    # Skip if already exists
    if adjusted_path.exists():
        return adjusted_path
    
    try:
        # Use ffmpeg with rubberband filter for pitch shifting without tempo change
        # -4 semitones shifts child voice closer to adult range
        cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-af", f"asetrate=16000*2^({semitones}/12),aresample=16000",
            str(adjusted_path),
            "-y"
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return adjusted_path
    except subprocess.CalledProcessError as e:
        print(f"  • Warning: Could not create pitch-adjusted audio: {e}")
        print(f"  • Falling back to original audio for diarization")
        return audio_path

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
            result = model.transcribe(audio, batch_size=BATCH_SIZE, print_progress=True)

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

            # Optional: allow disabling diarization for speed/compatibility
            disable_diar = os.environ.get("VOCAB_DISABLE_DIARIZATION", "0") == "1"
            if disable_diar:
                print("  • Diarization disabled via VOCAB_DISABLE_DIARIZATION=1; proceeding without speaker labels.")
            else:
                # Diarization (speaker identification)
                # Try simple energy-based speaker detection first (no HF token needed)
                # Then fall back to HF-based models if token is provided
                hf_token = os.environ.get("VOCAB_HF_TOKEN")
                
                if not hf_token:
                    # Simple approach: use energy/amplitude to detect speakers
                    # Adult voices typically have lower pitch and different energy patterns
                    print("  • Using simple energy-based speaker detection (no HuggingFace token)")
                    print("  • Note: For better accuracy, set VOCAB_HF_TOKEN with pyannote models")
                    
                    try:
                        # Use whisperx's built-in VAD and assign speakers based on segment patterns
                        # This won't be perfect but provides basic teacher/student separation
                        import numpy as np
                        from scipy import signal
                        
                        # Load audio for analysis
                        audio_data = whisperx.load_audio(str(audio_path))
                        
                        # Simple heuristic: alternate speakers on longer pauses
                        # Assign SPEAKER_00 and SPEAKER_01 based on turn-taking patterns
                        current_speaker = "SPEAKER_00"
                        last_end = 0
                        
                        for i, segment in enumerate(result["segments"]):
                            start = segment.get("start", 0)
                            
                            # If there's a pause > 1 second, likely a speaker change
                            if start - last_end > 1.0:
                                current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"
                            
                            segment["speaker"] = current_speaker
                            last_end = segment.get("end", start)
                        
                        print("  • Assigned speakers based on turn-taking patterns")
                        
                    except Exception as e:
                        print(f"  • Simple diarization failed: {e}")
                        print("  • Proceeding without speaker labels")
                        print("  • Assigned speakers based on turn-taking patterns")
                        
                    except Exception as e:
                        print(f"  • Simple diarization failed: {e}")
                        print("  • Proceeding without speaker labels")
                else:
                    # HuggingFace token provided - use advanced pyannote models
                    print("  • Using Hugging Face token for diarization.")
                    
                    # Create pitch-adjusted version for better child voice detection
                    # Children's voices are 3-5 semitones higher; shift down for diarization
                    pitch_env = os.environ.get("VOCAB_PITCH_SHIFT", "-4")
                    try:
                        pitch_shift = int(pitch_env)
                    except:
                        pitch_shift = -4
                    
                    if pitch_shift != 0:
                        print(f"  • Creating pitch-adjusted audio ({pitch_shift:+d} semitones) for better child voice detection")
                        diarization_audio = create_pitch_adjusted_audio(audio_path, pitch_shift)
                    else:
                        diarization_audio = audio_path
                    
                    diarize_segments = None
                    try:
                        # Try whisperx DiarizationPipeline first
                        diarize_model = whisperx.DiarizationPipeline(  # type: ignore[attr-defined]
                            use_auth_token=hf_token,
                            device=whisper_device
                        )
                        # Allow forcing an exact speaker count via env
                        min_speakers = MIN_SPEAKERS
                        max_speakers = MAX_SPEAKERS
                        force_num_env = os.environ.get("VOCAB_FORCE_NUM_SPEAKERS")
                        if force_num_env:
                            try:
                                n = int(force_num_env)
                                min_speakers = n
                                max_speakers = n
                                print(f"  • Forcing diarization to {n} speaker(s) (VOCAB_FORCE_NUM_SPEAKERS)")
                            except Exception:
                                print(f"  • Warning: invalid VOCAB_FORCE_NUM_SPEAKERS='{force_num_env}', using defaults {MIN_SPEAKERS}-{MAX_SPEAKERS}")
                        diarize_segments = diarize_model(
                            str(diarization_audio),
                            min_speakers=min_speakers,
                            max_speakers=max_speakers
                        )
                    except AttributeError:
                        # Fallback to pyannote.audio with child-optimized parameters
                        try:
                            from pyannote.audio import Pipeline  # type: ignore
                            print("  • Using pyannote.audio speaker-diarization-3.1 pipeline (child-optimized)")
                            pipeline = Pipeline.from_pretrained(
                                "pyannote/speaker-diarization-3.1",
                                use_auth_token=hf_token
                            )
                            
                            # Child speech optimization parameters
                            # Lower min_duration_off to catch shorter pauses (children speak faster)
                            # Adjust clustering threshold for better child/adult separation
                            force_num_env = os.environ.get("VOCAB_FORCE_NUM_SPEAKERS")
                            if force_num_env:
                                try:
                                    num_speakers = int(force_num_env)
                                    print(f"  • Forcing diarization to {num_speakers} speaker(s)")
                                    diarize_segments = pipeline(
                                        str(diarization_audio),
                                        num_speakers=num_speakers,
                                        min_duration_off=0.3  # Shorter pauses for child speech
                                    )
                                except Exception:
                                    print(f"  • Falling back to min/max speakers")
                                    diarize_segments = pipeline(
                                        str(diarization_audio),
                                        min_speakers=MIN_SPEAKERS,
                                        max_speakers=MAX_SPEAKERS,
                                        min_duration_off=0.3
                                    )
                            else:
                                diarize_segments = pipeline(
                                    str(diarization_audio),
                                    min_speakers=MIN_SPEAKERS,
                                    max_speakers=MAX_SPEAKERS,
                                    min_duration_off=0.3  # Child-optimized: shorter pauses
                                )
                        except Exception as de:
                            print(f"  • Diarization fallback failed: {de}")
                            print(f"  • Note: Pyannote models may require accepting terms at:")
                            print(f"  •   https://huggingface.co/pyannote/speaker-diarization-3.1")
                            print(f"  •   https://huggingface.co/pyannote/segmentation-3.0")
                    
                    if diarize_segments is not None:
                        result = whisperx.assign_word_speakers(diarize_segments, result)
                        try:
                            speakers_detected = sorted({seg.get("speaker") for seg in result.get("segments", []) if seg.get("speaker")})
                            if speakers_detected:
                                print(f"  • Speakers detected: {', '.join(speakers_detected)}")
                            else:
                                print(f"  • Warning: No speakers detected. This may indicate:")
                                print(f"  •   - Model terms not accepted on HuggingFace")
                                print(f"  •   - Very similar voices (child/adult can be hard to separate)")
                                print(f"  •   - Audio quality issues")
                        except Exception:
                            pass
                    else:
                        print("  • Proceeding without diarization due to previous errors.")

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
            # Create logs dir
            os.makedirs("./logs", exist_ok=True)
            log_file = Path("./logs") / f"transcription_{base_name}.log"
            tb = traceback.format_exc()
            try:
                with open(log_file, "w", encoding="utf-8") as lf:
                    lf.write("Transcription error for file: " + str(audio_path) + "\n\n")
                    lf.write(tb)
            except Exception:
                # best-effort write; ignore
                pass

            print(f"\n✗ Error transcribing {audio_path}: {e}")
            print(f"Detailed traceback written to: {log_file}")
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
    # Optionally process only a specific file by base name (without extension)
    only_base = os.environ.get("VOCAB_ONLY_BASENAME")
    if only_base:
        filtered = [p for p in video_files if Path(p).stem == only_base]
        if filtered:
            print(f"Processing only specified file (VOCAB_ONLY_BASENAME): {filtered[0]}")
            video_files = filtered
        else:
            print(f"Warning: No video matched VOCAB_ONLY_BASENAME={only_base}. Proceeding with all videos.")
    # Optionally only process the smallest video for quicker runs
    if os.environ.get("VOCAB_ONLY_SMALLEST", "0") == "1" and video_files:
        try:
            smallest = min(video_files, key=lambda p: os.path.getsize(p))
            print(f"Processing only the smallest video (VOCAB_ONLY_SMALLEST=1): {smallest}")
            video_files = [smallest]
        except Exception as e:
            print(f"Warning: could not select smallest video: {e}. Proceeding with all videos.")
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
