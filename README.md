# Mallya Documentary — "I Am Not a Chor"
### AI-Illustrated Documentary Production Pipeline

A complete production pipeline for generating a 20-minute illustrated documentary video using FLUX.1-dev (image generation) and Stable Video Diffusion (AI animation), with automatic sync to Google Drive.

---

## What This Repo Contains

| File | Purpose |
|------|---------|
| `setup.sh` | **One-command instance setup** — run this first on vast.ai |
| `generate_panels.py` | Step 1 — Generates 154 PNG stills (FLUX.1-dev) |
| `generate_all_clips_svd.py` | Step 2 — Animates all 154 stills (SVD-XT) |
| `prompts.json` | All 77 panel prompts in machine-readable format |
| `panel_motions.json` | Ken Burns motion type per panel (for FFmpeg fallback) |
| `make_video_clips.py` | FFmpeg Ken Burns alternative to SVD |
| `generate_hero_clips_svd.py` | SVD on selected hero panels only (lighter option) |
| `style_bible.md` | Visual consistency guide for all AI generation |
| `master_narration_and_prompts.md` | Full script — Hinglish VO + image prompt per panel |
| `voiceover_script_hinglish.md` | Clean VO-only script for recording |
| `bulk_image_prompts.md` | All prompts formatted for manual copy-paste |
| `pipeline_setup_guide.md` | Detailed step-by-step guide |

---

## Quick Start (on vast.ai RTX 3090 instance)

```bash
git clone https://github.com/nowrohit90008gmailcom/mallaya.git
cd mallaya
chmod +x setup.sh
./setup.sh
```

Then run in order:

```bash
# Step 1 — Generate PNG stills (~2 hours)
python /root/generate_panels.py

# Step 2 — Animate all panels with SVD (~5 hours)
python /root/generate_all_clips_svd.py
```

All 154 animated MP4 clips auto-save to your Google Drive.

---

## Output

```
Google Drive/
└── Mallya Documentary/
    ├── Generated Panels/     154 PNG stills  (P01_v1 … P77_v2)
    └── Video Clips SVD/      154 MP4 clips   (P01_v1 … P77_v2)
```

- Each clip: ~3 seconds | 1024×576 | H.264
- Loop ×2 in editor = 6 sec panel duration
- 77 panels × 2 variations = pick the best of each pair

---

## Cost

| Task | Time | Cost (vast.ai RTX 3090) |
|------|------|------------------------|
| PNG generation (FLUX.1-dev) | ~2.5 hrs | ~$0.70 |
| SVD animation (154 clips) | ~5.0 hrs | ~$1.40 |
| Disk (80 GB) | ~7.5 hrs | ~$0.30 |
| **Total** | **~7.5 hrs** | **~$2.40** |

---

## Requirements

- vast.ai or RunPod account (RTX 3090, 80 GB disk)
- HuggingFace account (free) — token needed
  - Accept [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) terms
  - Accept [SVD-XT](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) terms
- Google account (for Drive output)
