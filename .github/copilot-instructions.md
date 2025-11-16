# GitHub Copilot Instructions — vocab-education

Short, focused guidance so Copilot-style AI agents can become productive quickly in this repository.

## Project overview (big picture)
- This repository contains a single purpose script `transcription_pipeline.py`: a pipeline to download recorded videos from Google Drive, extract audio (ffmpeg), transcribe with WhisperX (ASR + alignment + diarization), and anonymize PII/PHI using Microsoft Presidio before saving text outputs.
- Data flow: Google Drive → local `DOWNLOAD_DIR` → `AUDIO_DIR` wav files → transcription & diarization → anonymized transcripts in `TRANSCRIPT_DIR`.
- The script is optimized for children & adult speech; it sets default Whisper model to `large-v2` and uses GPU (`cuda`) + `float16` compute by default for performance.

## Key files to review
- `transcription_pipeline.py` — full pipeline, configuration and helper functions. This is the authoritative source for conventions used by the project.

## Development & execution notes
- Setup (recommended): use a Python virtual environment then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install gdown whisperx torch presidio-analyzer presidio-anonymizer tqdm
# Ensure ffmpeg CLI is installed (macOS example)
brew install ffmpeg
```

- Execute the pipeline with: `python transcription_pipeline.py`.
- The script expects a `GDRIVE_FOLDER_ID` to be set in `transcription_pipeline.py` - open the script and set `GDRIVE_FOLDER_ID` to a Google Drive folder shared with "Anyone with the link" to allow `gdown` folder downloads.

## Observed patterns & conventions
- Configuration is defined at top of `transcription_pipeline.py` with constants (e.g., `WHISPER_MODEL`, `DEVICE`, `COMPUTE_TYPE`, directories). Prefer editing those constants rather than sprinkling overrides in code.
- File naming: audio files are created as `{base_name}.wav` in `AUDIO_DIR`; transcripts are saved as `{file}_anonymized.txt` in `TRANSCRIPT_DIR`.
- Timestamps & speaker attribution: the pipeline uses WhisperX word alignment and `whisperx.assign_word_speakers`. For multi-speaker recordings, the script uses the whisperx `DiarizationPipeline` with `MIN_SPEAKERS` and `MAX_SPEAKERS`.
- PII/PHI Entities: the `PII_PHI_ENTITIES` list defines what Presidio should search for — these are fairly exhaustive and stored in the repository as a single list for maintainability.

## Integration points & external dependencies
- gdown: downloads entire Google Drive folders. The folder must be shareable.
- ffmpeg CLI (external binary) is assumed to be present on PATH: used in `extract_audio()` with a 16kHz, mono WAV output for best results with Whisper.
- whisperx: used for transcription, alignments, and diarization. Some whisperx features may require a Hugging Face token for model downloads or diarization; the script sets `use_auth_token=None` by default but this can be changed if the agent receives rate limits or private model access is needed.
- presidio: Presidio Analyzer & Anonymizer are used. Presidio's NER models or language models may require additional downloads or configuration separately; error messages in the anonymization step may indicate missing dependencies.

## Debugging tips specific to this code
- If `gdown` fails: check `GDRIVE_FOLDER_ID` and that folder's sharing settings. Use `gdown` manually to reproduce the error.
- If ffmpeg fails during extraction: examine the `subprocess` call's `capture_output` message — it contains ffmpeg stderr with helpful diagnostics for codecs or permission issues.
- WhisperX: if model load fails, validate `WHISPER_MODEL` and `compute_type`. For CPU-only macOS, switch `DEVICE` to `cpu` and `COMPUTE_TYPE` to `int8`.
- Diarization errors: whisperx may require a Hugging Face token for private models; set `use_auth_token=<TOKEN>` in the `DiarizationPipeline` call.
- Presidio: Anonymizer requires valid analyzer results; if `analysis_results` is empty, Presidio may not have loaded the NER model for the locale.

## Code patterns & examples for AI agents
- Keep transformations isolated. Example: `format_transcript(segments)` takes a list of segments and returns formatted text — it's a pure function and is the ideal target for unit tests or refactoring.
- Logging: the script uses `print()` and `tqdm` progress bars; prefer `logging` if you need structured logs or tests that assert messages.
- Error handling strategy: the script catches exceptions and emits status for each step but continues processing. Respect this pipeline style when adding features.

## Testability & small changes
- Add unit tests for `format_transcript` (pure), `extract_audio` (mock `subprocess.run`), and anonymization (mock `AnalyzerEngine` and `AnonymizerEngine`).
- When adding tests, keep them focused on logic rather than external services.

## Suggestions for future agents
- Recommend adding a `--dry-run` flag to `transcription_pipeline.py` for quick checks (download list, file count, but don't run long processes).
- Suggest explicitly exposing a CLI by migrating config constants to `argparse` so agents can call the pipeline programmatically.

---

If you want the file merged with an existing `.github/copilot-instructions.md` add it to the workspace and I will re-run the search and update with the merged content — otherwise, does this draft cover the project's needs, or should it include more examples (e.g., unit tests, sample ffmpeg output)?
