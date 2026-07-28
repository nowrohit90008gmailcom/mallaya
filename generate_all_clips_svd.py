"""
generate_all_clips_svd.py
=========================
Mallya Documentary — AI Video Clip Generator (ALL 77 Panels)
Uses Stable Video Diffusion (SVD-XT) to animate every still image

Generates 2 animated MP4 variations per panel = 154 clips total.

RTX 3090 (24GB VRAM) — SVD-XT needs ~16GB, fits comfortably.
Estimated time: ~2 min per clip × 154 clips = ~5 hours

Input:  /root/gdrive/Mallya Documentary/Generated Panels/P01_v1.png ...
Output: /root/gdrive/Mallya Documentary/Video Clips SVD/P01_v1.mp4 ...

Run AFTER generate_panels.py has finished producing all PNGs.
"""

import torch
import os
import json
from pathlib import Path
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from datetime import datetime
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

INPUT_DIR  = "/root/gdrive/Mallya Documentary/Generated Panels"
OUTPUT_DIR = "/root/gdrive/Mallya Documentary/Video Clips SVD"
LOG_FILE   = "/root/svd_log.txt"
FAIL_FILE  = "/root/svd_failed.txt"

PROMPTS_FILE = "/root/prompts.json"   # to get the ordered list of panel IDs

VARIATIONS         = 2
FRAMES             = 25        # SVD-XT generates 25 frames
OUTPUT_FPS         = 8         # 25 frames / 8 fps = ~3.1 sec clip (loop x2 in editor = 6.2 sec)
DECODE_CHUNK_SIZE  = 8         # Reduce to 4 if you get OOM
MOTION_BUCKET_ID   = 100       # 0=very subtle, 127=maximum motion, 100=cinematic
NOISE_AUG_STRENGTH = 0.05      # Higher = more motion/variation between frames

# SVD input resolution (must be 1024x576 for SVD-XT)
SVD_WIDTH  = 1024
SVD_HEIGHT = 576

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_fail(panel_id: str, variation: int, err: str):
    with open(FAIL_FILE, "a", encoding="utf-8") as f:
        f.write(f"{panel_id}_v{variation} | {err}\n")


def already_done(output_dir: Path, panel_id: str, variation: int) -> bool:
    return (output_dir / f"{panel_id}_v{variation}.mp4").exists()


def load_image(path: str) -> Image.Image:
    """Resize 1920x1080 PNG to 1024x576 as required by SVD-XT."""
    img = Image.open(path).convert("RGB")
    img = img.resize((SVD_WIDTH, SVD_HEIGHT), Image.LANCZOS)
    return img


# ─────────────────────────────────────────────
# PIPELINE LOAD
# ─────────────────────────────────────────────

def load_pipeline():
    log("Loading SVD-XT pipeline... (downloads ~10 GB on first run)")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking()
    log("SVD-XT loaded successfully.")
    return pipe


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Validate ──
    if not os.path.exists(PROMPTS_FILE):
        print(f"ERROR: prompts.json not found at {PROMPTS_FILE}")
        return

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        panels = json.load(f)   # list of { id, scene, prompt }

    panel_ids = [p["id"] for p in panels]   # ["P01", "P02", ... "P77"]

    # ── Create output dir ──
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(INPUT_DIR)

    # ── Count remaining ──
    total = len(panel_ids) * VARIATIONS
    done  = sum(
        1 for pid in panel_ids
        for v in range(1, VARIATIONS + 1)
        if already_done(output_dir, pid, v)
    )
    remaining = total - done

    log(f"Total clips: {total} | Already done: {done} | Remaining: {remaining}")
    log(f"Estimated time: ~{remaining * 2 // 60} hrs {(remaining * 2) % 60} min")

    if remaining == 0:
        log("All clips already generated!")
        return

    # ── Load model ──
    pipe = load_pipeline()

    # ── Generate ──
    completed = 0
    failed    = 0

    with tqdm(total=remaining, desc="SVD clip generation", unit="clip") as pbar:
        for panel_id in panel_ids:
            for variation in range(1, VARIATIONS + 1):

                if already_done(output_dir, panel_id, variation):
                    continue

                input_path = input_dir / f"{panel_id}_v{variation}.png"

                if not input_path.exists():
                    log(f"SKIP {panel_id}_v{variation} — PNG not found yet")
                    log_fail(panel_id, variation, "source PNG missing")
                    failed += 1
                    pbar.update(1)
                    continue

                try:
                    image = load_image(str(input_path))

                    # Use different seeds for v1 vs v2 to get real variation
                    seed = (hash(panel_id) + variation * 1337) % (2**32)
                    generator = torch.manual_seed(seed)

                    frames = pipe(
                        image,
                        num_frames=FRAMES,
                        num_inference_steps=25,
                        motion_bucket_id=MOTION_BUCKET_ID,
                        noise_aug_strength=NOISE_AUG_STRENGTH,
                        decode_chunk_size=DECODE_CHUNK_SIZE,
                        generator=generator,
                    ).frames[0]

                    output_path = str(output_dir / f"{panel_id}_v{variation}.mp4")
                    export_to_video(frames, output_path, fps=OUTPUT_FPS)

                    log(f"✓ {panel_id}_v{variation}.mp4 saved")
                    completed += 1

                except torch.cuda.OutOfMemoryError:
                    err = "CUDA OOM — reduce DECODE_CHUNK_SIZE to 4"
                    log(f"✗ OOM: {panel_id}_v{variation} | {err}")
                    log_fail(panel_id, variation, err)
                    failed += 1
                    torch.cuda.empty_cache()

                except Exception as e:
                    err = str(e)[:200]
                    log(f"✗ ERROR: {panel_id}_v{variation} | {err}")
                    log_fail(panel_id, variation, err)
                    failed += 1

                finally:
                    pbar.update(1)
                    torch.cuda.empty_cache()

    # ── Summary ──
    log("=" * 60)
    log(f"DONE. Completed: {completed} | Failed: {failed}")
    log(f"Clips saved to: {OUTPUT_DIR}")
    if failed:
        log(f"Failed list: {FAIL_FILE}")
        log("Re-run the script — it will retry only the failed ones.")
    log("=" * 60)
    log("EDITOR NOTE: Each clip is ~3 sec. Loop it x2 or x3 in your")
    log("timeline to fill the 6-10 sec panel duration.")


if __name__ == "__main__":
    main()
