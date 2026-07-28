# 🎬 Mallaya Documentary — AnimateDiff 2D Animated Pipeline

> **"I Am Not a Chor" — The Vijay Mallya Documentary**
> 77 panels × 8-second 2D animated clips in DOKIO documentary style.

---

## Project Structure

| File | Purpose |
|------|---------|
| `prompts.json` | **Master source** — 77 panels with detailed text-to-video prompts, camera, color, narration |
| `generate_animatediff.py` | **Main pipeline** — AnimateDiff-Lightning + epiCRealism → 2D animated 8-sec MP4s |
| `setup.sh` | One-click VPS setup |
| `README.md` | This file |

---

## Quick Start (VPS — vast.ai RTX 3090)

```bash
# 1. Clone
git clone https://github.com/nowrohit90008gmailcom/mallaya.git
cd mallaya

# 2. Setup
export HF_TOKEN=hf_your_token_here
bash setup.sh

# 3. Run
python generate_animatediff.py
```

---

## Model Stack

| Component | Model |
|-----------|-------|
| **Motion Module** | AnimateDiff-Lightning (ByteDance) |
| **Base Model** | epiCRealism (cartoon-friendly) |
| **Style** | 2D animated documentary — DOKIO / Indian graphic novel |
| **Speed** | 4 inference steps — ultra fast |
| **VRAM** | ~10 GB — comfortable on RTX 3090 |
| **Output** | 1920×1080, H.264 CRF 17, 24fps, 8 seconds |

---

## Output

- **77 × 8-second 2D animated MP4 clips**
- Uploaded to Google Drive: `📁 Mallaya Documentary / Video Clips Animated`