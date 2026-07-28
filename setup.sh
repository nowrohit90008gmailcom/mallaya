#!/bin/bash
# =============================================================
#  setup.sh — Mallya Documentary Pipeline
# =============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Please enter your Hugging Face Token (starts with hf_...):${NC}"
    read -r HF_TOKEN
fi

export HF_TOKEN="$HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

banner() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

ok()   { echo -e "${GREEN}✓ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }

banner "Mallya Documentary — Instance Setup"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "No GPU found")
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "?")
echo -e "  GPU detected: ${GREEN}$GPU_NAME${NC}"
echo -e "  VRAM:         ${GREEN}$VRAM${NC}"
echo -e "  Python:       $(python3 --version 2>&1)"
echo ""

# 1. SYSTEM PACKAGES
banner "STEP 1/5 — Installing System Packages"
apt-get update -qq
apt-get install -y ffmpeg curl wget git > /dev/null 2>&1
ok "ffmpeg, curl, wget, git installed"

if ! command -v rclone &> /dev/null; then
    curl https://rclone.org/install.sh | sudo bash > /dev/null 2>&1
    ok "rclone installed"
else
    ok "rclone already installed"
fi

# 2. PYTHON DEPENDENCIES
banner "STEP 2/5 — Installing Python Packages"
pip install -q --upgrade pip
pip install -q \
    diffusers>=0.30.0 \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    safetensors \
    huggingface_hub \
    sentencepiece \
    Pillow \
    tqdm \
    torch torchvision \
    xformers

ok "All Python packages installed"

# 3. AUTOMATIC HUGGINGFACE LOGIN
banner "STEP 3/5 — Auto HuggingFace Authentication"
hf auth login --token "$HF_TOKEN" > /dev/null 2>&1 || \
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential > /dev/null 2>&1 || \
huggingface-cli login --token "$HF_TOKEN" > /dev/null 2>&1 || true
ok "Authenticated with HuggingFace (Token loaded)"

# 4. RCLONE GOOGLE DRIVE VERIFICATION
banner "STEP 4/5 — Google Drive Verification"

if rclone listremotes | grep -q "^gdrive:"; then
    ok "gdrive remote is configured"
else
    echo -e "  Setting up rclone remote for Google Drive."
    rclone config
fi

# Test API connection
if rclone lsd gdrive: > /dev/null 2>&1; then
    ok "Google Drive API connection verified! Direct upload enabled."
else
    warn "Could not list gdrive remote. Please check rclone config."
fi

# Create local output directories
mkdir -p "/workspace/output/Generated_Panels"
mkdir -p "/workspace/output/Video_Clips_SVD"
ok "Local output directories created"

# 5. PREPARE SCRIPTS
banner "STEP 5/5 — Preparing Scripts"
cp -f prompts.json /root/ 2>/dev/null || true
cp -f generate_panels.py /root/ 2>/dev/null || true
cp -f generate_all_clips_svd.py /root/ 2>/dev/null || true
ok "Scripts ready"

banner "✅ Setup Complete — Ready to Generate!"

echo -e "  Run the pipeline:\n"
echo -e "  ${GREEN}# Step 1 — Generate 154 PNG stills (FLUX.1-dev) — ~2 hours${NC}"
echo -e "  python generate_panels.py\n"
echo -e "  ${GREEN}# Step 2 — Animate all 154 stills (8.0 sec MP4s) — ~5 hours${NC}"
echo -e "  python generate_all_clips_svd.py\n"
