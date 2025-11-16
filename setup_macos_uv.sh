# Create and activate a virtual environment using UV
uv venv .venv
source .venv/bin/activate

# Install PyTorch with MPS support for Mac (verify the exact command on pytorch.org/get-started/locally)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install additional packages
uv pip install gdown whisperx presidio-analyzer presidio-anonymizer tqdm
brew install ffmpeg
export PYTORCH_ENABLE_MPS_FALLBACK=1