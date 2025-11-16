# Activate your venv first
python3.10 -m venv .venv
source .venv/bin/activate

# Visit https://pytorch.org/get-started/locally/ and choose Mac / Pip / MPS,
# then run the command they generate. Example (verify on the website):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gdown whisperx presidio-analyzer presidio-anonymizer tqdm
# whisperx may download model assets the first run
# ensure ffmpeg is installed on mac:
brew install ffmpeg
export PYTORCH_ENABLE_MPS_FALLBACK=1