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

# Read Hugging Face token from environment variable HF_TOKEN, or prompt if not set
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
banner "STEP 1/6 — Installing System Packages"
apt-get update -qq
apt-get install -y ffmpeg fuse3 curl wget git > /dev/null 2>&1
ok "ffmpeg, fuse3, curl, wget, git installed"

if ! command -v rclone &> /dev/null; then
    curl https://rclone.org/install.sh | sudo bash > /dev/null 2>&1
    ok "rclone installed"
else
    ok "rclone already installed"
fi

# 2. PYTHON DEPENDENCIES
banner "STEP 2/6 — Installing Python Packages"
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
banner "STEP 3/6 — Auto HuggingFace Authentication"
hf auth login --token "$HF_TOKEN" > /dev/null 2>&1 || \
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential > /dev/null 2>&1 || \
huggingface-cli login --token "$HF_TOKEN" > /dev/null 2>&1 || true
ok "Authenticated with HuggingFace (Token loaded)"

# 4. RCLONE GOOGLE DRIVE SETUP
banner "STEP 4/6 — Google Drive Setup (rclone)"
mkdir -p /root/gdrive

if rclone listremotes | grep -q "^gdrive:"; then
    warn "gdrive remote already configured — skipping interactive setup"
else
    echo -e "  Setting up rclone remote for Google Drive."
    echo -e "  Follow the on-screen prompts (name: ${YELLOW}gdrive${NC}, storage: ${YELLOW}17${NC})."
    echo ""
    rclone config
fi

ok "rclone remote configured"

# 5. MOUNT GOOGLE DRIVE
banner "STEP 5/6 — Mounting Google Drive"
pkill rclone 2>/dev/null || true
sleep 2

rclone mount gdrive: /root/gdrive \
    --vfs-cache-mode writes \
    --allow-non-empty \
    --allow-other \
    --daemon \
    --log-file /root/rclone.log

sleep 3

if ls /root/gdrive > /dev/null 2>&1; then
    ok "Google Drive mounted at /root/gdrive"
else
    echo -e "${RED}✗ Mount failed. Run: rclone mount gdrive: /root/gdrive --vfs-cache-mode writes --daemon${NC}"
    exit 1
fi

mkdir -p "/root/gdrive/Mallya Documentary/Generated Panels"
mkdir -p "/root/gdrive/Mallya Documentary/Video Clips SVD"
ok "Output folders created on Google Drive"

# 6. PREPARE SCRIPTS
banner "STEP 6/6 — Preparing Scripts"
cp -f prompts.json /root/ 2>/dev/null || true
cp -f generate_panels.py /root/ 2>/dev/null || true
cp -f generate_all_clips_svd.py /root/ 2>/dev/null || true
ok "Scripts ready in current folder and /root/"

banner "✅ Setup Complete — Ready to Generate!"

echo -e "  Run the pipeline:\n"
echo -e "  ${GREEN}# Step 1 — Generate 154 PNG stills (FLUX.1-dev) — ~2 hours${NC}"
echo -e "  python generate_panels.py\n"
echo -e "  ${GREEN}# Step 2 — Animate all 154 stills (8.0 sec MP4s) — ~5 hours${NC}"
echo -e "  python generate_all_clips_svd.py\n"
