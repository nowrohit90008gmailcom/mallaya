"""
generate_panels.py
==================
Mallya Documentary — Auto Image Generation Script
RTX 3090 | FLUX.1-dev | Google Drive Output

Generates 2 variations per panel (154 images total)
Saves directly to Google Drive via rclone mount
"""

import json
import os
import time
import torch
import random
from pathlib import Path
from datetime import datetime
from PIL import Image
from diffusers import FluxPipeline
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

GDRIVE_OUTPUT_DIR = "/root/gdrive/Mallya Documentary/Generated Panels"

# Resolve prompts.json location dynamically
SCRIPT_DIR = Path(__file__).parent
if (SCRIPT_DIR / "prompts.json").exists():
    PROMPTS_FILE = str(SCRIPT_DIR / "prompts.json")
else:
    PROMPTS_FILE = "/root/prompts.json"

LOG_FILE    = "/root/generation_log.txt"
FAILED_FILE = "/root/failed_panels.txt"

MODEL_ID          = "black-forest-labs/FLUX.1-dev"
IMAGE_WIDTH       = 1920
IMAGE_HEIGHT      = 1080
INFERENCE_STEPS   = 28          # 28 = good quality/speed balance on 3090
GUIDANCE_SCALE    = 3.5         # FLUX default
VARIATIONS        = 2           # Number of copies per panel
USE_FP8           = False       # Set True if you get OOM errors

NEGATIVE_PROMPT = (
    "photorealistic face, text, watermark, blurry, cartoon, anime, "
    "low quality, deformed hands, extra fingers, logos, nsfw"
)

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

def log(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_failure(panel_id: str, variation: int, error: str):
    with open(FAILED_FILE, "a", encoding="utf-8") as f:
        f.write(f"{panel_id}_v{variation} | {error}\n")


def already_done(output_dir: Path, panel_id: str, variation: int) -> bool:
    path = output_dir / f"{panel_id}_v{variation}.png"
    return path.exists()


def load_pipeline():
    log("Loading FLUX.1-dev pipeline... (first run downloads ~23 GB)")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")

    if USE_FP8:
        pipe.transformer = pipe.transformer.to(torch.float8_e4m3fn)

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    log("Pipeline loaded successfully.")
    return pipe


def generate_image(pipe, prompt: str, seed: int) -> Image.Image:
    generator = torch.Generator("cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )
    return result.images[0]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if not os.path.exists(PROMPTS_FILE):
        print(f"ERROR: prompts.json not found at {PROMPTS_FILE}")
        return

    output_dir = Path(GDRIVE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        panels = json.load(f)

    log(f"Loaded {len(panels)} panels. Generating {VARIATIONS} variations each.")

    already_count = sum(
        1 for p in panels
        for v in range(1, VARIATIONS + 1)
        if already_done(output_dir, p["id"], v)
    )
    log(f"Already completed: {already_count} | Remaining: {len(panels) * VARIATIONS - already_count}")

    pipe = load_pipeline()

    total = len(panels) * VARIATIONS
    completed = 0
    failed = 0

    with tqdm(total=total, desc="Generating panels", unit="img") as pbar:
        for panel in panels:
            panel_id   = panel["id"]
            prompt     = panel["prompt"]
            scene      = panel.get("scene", "")

            for variation in range(1, VARIATIONS + 1):
                if already_done(output_dir, panel_id, variation):
                    pbar.update(1)
                    completed += 1
                    continue

                seed = random.randint(0, 2**32 - 1)

                try:
                    image = generate_image(pipe, prompt, seed)
                    filename = f"{panel_id}_v{variation}.png"
                    save_path = output_dir / filename
                    image.save(str(save_path), format="PNG", optimize=False)

                    log(f"✓ Saved {filename} | seed={seed} | {scene}")
                    completed += 1

                except torch.cuda.OutOfMemoryError:
                    error_msg = "CUDA OOM — try USE_FP8=True"
                    log(f"✗ FAILED {panel_id}_v{variation}: {error_msg}")
                    log_failure(panel_id, variation, error_msg)
                    failed += 1
                    torch.cuda.empty_cache()

                except Exception as e:
                    error_msg = str(e)
                    log(f"✗ FAILED {panel_id}_v{variation}: {error_msg}")
                    log_failure(panel_id, variation, error_msg)
                    failed += 1

                finally:
                    pbar.update(1)
                    torch.cuda.empty_cache()

    log("=" * 50)
    log(f"DONE. Completed: {completed} | Failed: {failed}")
    log(f"Images saved to: {GDRIVE_OUTPUT_DIR}")
    log("=" * 50)


if __name__ == "__main__":
    main()
