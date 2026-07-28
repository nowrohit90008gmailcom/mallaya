# COMPLETE VIDEO CLIP PIPELINE
## End-to-End: RTX 3090 → Google Drive
### "I Am Not a Chor" — 77 Panels × 2 Variations = 154 AI-Animated MP4 Clips

---

## THE FULL PICTURE

```
STEP 1          STEP 2              STEP 3                  STEP 4
Rent 3090  →   Generate PNGs  →   Animate ALL with SVD  →  Google Drive
(vast.ai)      (FLUX.1-dev)       (SVD-XT, all 77×2)        (rclone mount)
  5 min          ~2 hours            ~5 hours                  auto-sync
```

**Output folder structure in your Google Drive:**
```
Mallya Documentary/
├── Generated Panels/    ← 154 PNG stills   (P01_v1.png … P77_v2.png)
└── Video Clips SVD/     ← 154 MP4 clips    (P01_v1.mp4 … P77_v2.mp4)
                            AI-animated, ~3 sec each
                            Loop ×2 in editor → 6 sec per panel
```

**Total instance time: ~7–8 hours | Total cost on vast.ai: ~$2**

---

## FILES YOU NEED ON THE INSTANCE

Upload these 3 files from `d:\Documentry\` to `/root/` on the instance:

| File | Purpose |
|------|---------|
| `prompts.json` | All 77 panel prompts (machine-readable) |
| `generate_panels.py` | Step 1 — FLUX image generation |
| `generate_all_clips_svd.py` | Step 2 — SVD animation for all panels |

---

## STEP 1 — RENT YOUR RTX 3090 INSTANCE

### vast.ai (Cheapest — ~$0.20–0.35/hr)
1. Sign up at **vast.ai**
2. Click **Search** → set these filters:
   - GPU: **RTX 3090** (24 GB VRAM)
   - RAM: **32 GB+**
   - Disk: **80 GB** (FLUX = 23 GB + SVD = 10 GB + outputs = ~5 GB)
   - Image: **`vastai/pytorch:2.3.0-cuda12.1-devel`**
3. Rent the cheapest available → click **Connect** → copy the SSH command
4. Open PowerShell, paste the SSH command to connect

---

## STEP 2 — ONE-TIME SETUP ON THE INSTANCE

Paste this entire block into the SSH terminal — it installs everything:

```bash
apt-get update -qq && apt-get install -y fuse3 && \
curl https://rclone.org/install.sh | sudo bash && \
pip install -q diffusers transformers accelerate \
    safetensors huggingface_hub sentencepiece Pillow tqdm
echo "All dependencies installed."
```

---

## STEP 3 — CONNECT GOOGLE DRIVE (rclone)

### On your Windows PC (one-time, 3 minutes):
```powershell
# Install rclone from rclone.org/downloads then run:
rclone config
```
- Name → `gdrive`
- Storage type → `17` (Google Drive)
- Client ID → press Enter (blank)
- Client Secret → press Enter (blank)
- Scope → `1` (full access)
- Browser opens → log in with your Google account → Allow
- Copy the full token JSON that appears

### On the instance:
```bash
# Configure rclone — paste your token when prompted
rclone config

# Mount Google Drive
mkdir -p /root/gdrive
rclone mount gdrive: /root/gdrive \
    --vfs-cache-mode writes \
    --allow-non-empty \
    --daemon

# Verify
ls /root/gdrive
# Should show your Google Drive files

# Create output folders
mkdir -p "/root/gdrive/Mallya Documentary/Generated Panels"
mkdir -p "/root/gdrive/Mallya Documentary/Video Clips SVD"
```

---

## STEP 4 — ACCEPT MODEL TERMS ON HUGGINGFACE

Both models are gated and require a free HuggingFace account.

```bash
# Get your token from: huggingface.co/settings/tokens
huggingface-cli login
# Paste token when prompted
```

Then open these two URLs in your browser and click **"Agree and access":**
- `huggingface.co/black-forest-labs/FLUX.1-dev`
- `huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt`

---

## STEP 5 — UPLOAD YOUR FILES

### From Windows PowerShell:
```powershell
$PORT = "REPLACE_WITH_PORT"      # from vast.ai dashboard
$IP   = "REPLACE_WITH_IP"        # from vast.ai dashboard

scp -P $PORT `
    "d:\Documentry\prompts.json" `
    "d:\Documentry\generate_panels.py" `
    "d:\Documentry\generate_all_clips_svd.py" `
    "root@${IP}:/root/"

echo "Files uploaded."
```

> **Alternative:** Use the vast.ai web dashboard file upload button.

---

## STEP 6 — RUN STEP 1: GENERATE PNG STILLS

```bash
# First run downloads FLUX.1-dev (~23 GB) — takes ~10 min
# Then generates 154 PNG images — takes ~2 hours
python /root/generate_panels.py
```

Watch progress in real time:
```bash
# In a second SSH terminal:
watch -n 10 'ls "/root/gdrive/Mallya Documentary/Generated Panels" | wc -l'
# Count climbs from 0 to 154
```

---

## STEP 7 — RUN STEP 2: ANIMATE ALL PANELS WITH SVD

Start immediately after (or while) Step 6 finishes:

```bash
# Downloads SVD-XT model (~10 GB on first run)
# Then animates all 154 images — takes ~5 hours
python /root/generate_all_clips_svd.py
```

Watch progress:
```bash
# In a second terminal:
watch -n 30 'ls "/root/gdrive/Mallya Documentary/Video Clips SVD" | wc -l'
# Count climbs from 0 to 154

# See live log:
tail -f /root/svd_log.txt
```

The script **auto-skips already-completed clips** — safe to restart if disconnected.

---

## STEP 8 — VERIFY IN GOOGLE DRIVE

Open drive.google.com in your browser. You'll see clips appearing in real time:

```
📁 Mallya Documentary
  📁 Generated Panels        154 PNG files ✓
  📁 Video Clips SVD         154 MP4 files ✓  (fills up over ~5 hrs)
```

Each clip: **1024×576, H.264, ~3 seconds, AI-animated**
In your editor: set each clip to **loop ×2** to get the 6-second panel duration.

---

## STEP 9 — DOWNLOAD TO YOUR PC

Once done, sync just the Video Clips SVD folder:

```powershell
# On your Windows PC:
rclone copy "gdrive:Mallya Documentary/Video Clips SVD" `
    "D:\Documentry\VideoClips" --progress
```

Or just open Google Drive in browser, select all MP4s, and download as ZIP.

---

## FULL COST ESTIMATE

| Task | Time | Cost |
|------|------|------|
| PNG generation (FLUX.1-dev) | ~2.5 hrs | ~$0.70 |
| SVD animation (all 154 clips) | ~5.0 hrs | ~$1.40 |
| Disk (80 GB instance) | ~7.5 hrs | ~$0.30 |
| **Total** | **~7.5 hrs** | **~$2.40** |

---

## EDITOR WORKFLOW (After Download)

1. Import all 154 clips from `VideoClips/` into your editor
2. For each panel, pick **v1 or v2** (the better-looking variation)
3. Place in order: P01 → P77 on the timeline
4. **Loop each clip ×2** (right-click → Loop in DaVinci / Premiere)
5. Add voiceover from `voiceover_script_hinglish.md`
6. Add text overlays from `master_narration_and_prompts.md` OVERLAY column
7. Add background music under everything
8. Export: **1080p, H.264, 30fps**

> **Upscale tip:** Run clips through **Topaz Video AI** or **Real-ESRGAN**
> to upscale from 1024×576 → 1920×1080 before dropping into the timeline.

---

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| FLUX CUDA out of memory | In `generate_panels.py` replace `pipe.to("cuda")` with `pipe.enable_sequential_cpu_offload()` |
| SVD CUDA out of memory | In `generate_all_clips_svd.py` set `DECODE_CHUNK_SIZE = 4` |
| Google Drive mount drops | Re-run the `rclone mount` command |
| HuggingFace 403 error | Accept model terms at huggingface.co (both FLUX and SVD) |
| Script stopped mid-way | Just re-run — both scripts skip completed files automatically |
| Clips look too subtle / not enough motion | Increase `MOTION_BUCKET_ID` from 100 → 127 in `generate_all_clips_svd.py` |
| Clips look too chaotic | Decrease `MOTION_BUCKET_ID` from 100 → 60 |

---

## QUICK COMMAND REFERENCE

```bash
# Live GPU usage
watch -n 1 nvidia-smi

# PNG count
ls "/root/gdrive/Mallya Documentary/Generated Panels" | wc -l

# SVD clip count
ls "/root/gdrive/Mallya Documentary/Video Clips SVD" | wc -l

# Live FLUX log
tail -f /root/generation_log.txt

# Live SVD log
tail -f /root/svd_log.txt

# Check any failures
cat /root/failed_panels.txt
cat /root/svd_failed.txt

# Kill and restart rclone mount
pkill rclone
rclone mount gdrive: /root/gdrive --vfs-cache-mode writes --allow-non-empty --daemon
```
