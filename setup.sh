#!/bin/bash
# =============================================================
#  setup.sh — Mallya Documentary Pipeline
#  Run this ONCE on your RTX 3090 vast.ai / RunPod instance
#
#  Usage:
#    git clone https://github.com/YOUR_USERNAME/mallya-documentary.git
#    cd mallya-documentary
#    chmod +x setup.sh
#    ./setup.sh
#
#  What it does:
#    1. Installs system packages (ffmpeg, fuse3, rclone)
#    2. Installs Python dependencies
#    3. Logs you into HuggingFace (you'll need your HF token)
#    4. Guides you through rclone Google Drive setup
#    5. Mounts Google Drive to /root/gdrive
#    6. Creates output folders on Google Drive
#    7. Prints final run instructions
# =============================================================

set -e   # exit on any error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

banner() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

ok()   { echo -e "${GREEN}✓ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
info() { echo -e "  $1"; }

# ──────────────────────────────────────────────
# 0. DETECT ENVIRONMENT
# ──────────────────────────────────────────────
banner "Mallya Documentary — Instance Setup"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "No GPU found")
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "?")
echo -e "  GPU detected: ${GREEN}$GPU_NAME${NC}"
echo -e "  VRAM:         ${GREEN}$VRAM${NC}"
echo -e "  Python:       $(python3 --version 2>&1)"
echo ""

# ──────────────────────────────────────────────
# 1. SYSTEM PACKAGES
# ──────────────────────────────────────────────
banner "STEP 1/6 — Installing System Packages"

apt-get update -qq
apt-get install -y ffmpeg fuse3 curl wget git > /dev/null 2>&1
ok "ffmpeg, fuse3, curl, wget, git installed"

# Install rclone
if ! command -v rclone &> /dev/null; then
    curl https://rclone.org/install.sh | sudo bash > /dev/null 2>&1
    ok "rclone installed"
else
    ok "rclone already installed ($(rclone --version | head -1))"
fi

# ──────────────────────────────────────────────
# 2. PYTHON DEPENDENCIES
# ──────────────────────────────────────────────
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

# Verify torch + CUDA
python3 -c "
import torch
cuda = torch.cuda.is_available()
device = torch.cuda.get_device_name(0) if cuda else 'CPU only'
print(f'  PyTorch {torch.__version__} | CUDA: {cuda} | Device: {device}')
"

# ──────────────────────────────────────────────
# 3. HUGGINGFACE LOGIN
# ──────────────────────────────────────────────
banner "STEP 3/6 — HuggingFace Login"

echo -e "  You need a free HuggingFace account and a token."
echo -e "  Get your token at: ${CYAN}https://huggingface.co/settings/tokens${NC}"
echo -e "  Then accept model terms at:"
echo -e "    ${CYAN}https://huggingface.co/black-forest-labs/FLUX.1-dev${NC}"
echo -e "    ${CYAN}https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt${NC}"
echo ""

huggingface-cli login

ok "HuggingFace login complete"

# ──────────────────────────────────────────────
# 4. RCLONE GOOGLE DRIVE SETUP
# ──────────────────────────────────────────────
banner "STEP 4/6 — Google Drive Setup (rclone)"

echo -e "  Setting up rclone remote for Google Drive."
echo -e "  When prompted:"
echo -e "    - Name:         ${YELLOW}gdrive${NC}"
echo -e "    - Storage type: ${YELLOW}17${NC}  (Google Drive)"
echo -e "    - Client ID:    ${YELLOW}press Enter${NC} (leave blank)"
echo -e "    - Client Secret:${YELLOW}press Enter${NC} (leave blank)"
echo -e "    - Scope:        ${YELLOW}1${NC}  (full access)"
echo -e "    - Root folder:  ${YELLOW}press Enter${NC} (leave blank)"
echo -e "    - Service acct: ${YELLOW}press Enter${NC} (leave blank)"
echo -e "    - Auto config:  ${YELLOW}n${NC}  (we are on a headless server)"
echo -e "    - Paste the auth token URL into your LOCAL browser, then paste the code here."
echo ""
read -p "Press Enter when ready to configure rclone..." _

# Check if gdrive remote already exists
if rclone listremotes | grep -q "^gdrive:"; then
    warn "gdrive remote already configured — skipping"
else
    rclone config
fi

ok "rclone remote configured"

# ──────────────────────────────────────────────
# 5. MOUNT GOOGLE DRIVE
# ──────────────────────────────────────────────
banner "STEP 5/6 — Mounting Google Drive"

mkdir -p /root/gdrive

# Kill any existing mount
pkill rclone 2>/dev/null || true
sleep 2

rclone mount gdrive: /root/gdrive \
    --vfs-cache-mode writes \
    --allow-non-empty \
    --allow-other \
    --daemon \
    --log-file /root/rclone.log

sleep 3

# Verify mount
if ls /root/gdrive > /dev/null 2>&1; then
    ok "Google Drive mounted at /root/gdrive"
else
    echo -e "${RED}✗ Mount failed. Check /root/rclone.log for details.${NC}"
    echo "  Try running manually: rclone mount gdrive: /root/gdrive --vfs-cache-mode writes --daemon"
    exit 1
fi

# Create output directories
mkdir -p "/root/gdrive/Mallya Documentary/Generated Panels"
mkdir -p "/root/gdrive/Mallya Documentary/Video Clips SVD"
ok "Output folders created on Google Drive"

# ──────────────────────────────────────────────
# 6. VERIFY SCRIPTS ARE PRESENT
# ──────────────────────────────────────────────
banner "STEP 6/6 — Verifying Project Files"

MISSING=0
for f in prompts.json generate_panels.py generate_all_clips_svd.py; do
    if [ -f "/root/mallya-documentary/$f" ]; then
        ok "$f found"
    else
        echo -e "${RED}✗ $f NOT found in /root/mallya-documentary/${NC}"
        MISSING=$((MISSING+1))
    fi
done

if [ $MISSING -gt 0 ]; then
    warn "Some files missing. Make sure git clone completed fully."
fi

# Copy scripts to /root for easy access
cp /root/mallya-documentary/prompts.json /root/
cp /root/mallya-documentary/generate_panels.py /root/
cp /root/mallya-documentary/generate_all_clips_svd.py /root/
ok "Scripts copied to /root/"

# ──────────────────────────────────────────────
# DONE — PRINT RUN INSTRUCTIONS
# ──────────────────────────────────────────────
banner "✅ Setup Complete — Ready to Generate!"

echo -e "  Run the pipeline in order:\n"
echo -e "  ${GREEN}# Step 1 — Generate 154 PNG stills (FLUX.1-dev) — ~2 hours${NC}"
echo -e "  python /root/generate_panels.py\n"
echo -e "  ${GREEN}# Step 2 — Animate all 154 stills with SVD — ~5 hours${NC}"
echo -e "  python /root/generate_all_clips_svd.py\n"
echo -e "  ${CYAN}Monitor progress:${NC}"
echo -e "  watch -n 10 'ls \"/root/gdrive/Mallya Documentary/Generated Panels\" | wc -l'"
echo -e "  watch -n 30 'ls \"/root/gdrive/Mallya Documentary/Video Clips SVD\" | wc -l'"
echo -e "  tail -f /root/generation_log.txt"
echo -e "  tail -f /root/svd_log.txt\n"
echo -e "  ${YELLOW}⚠  If the instance disconnects, just SSH back in and re-run${NC}"
echo -e "  ${YELLOW}   both scripts — they skip already-completed files.${NC}\n"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
