"""
generate_panels.py
==================
Mallya Documentary — Auto Image Generation Script
RTX 3090 | FLUX.1-dev | Direct Google Drive Upload (rclone copy)

Generates 2 variations per panel (154 images total)
Uploads directly to Google Drive via rclone API as soon as each image is generated!
"""

import json
import os
import subprocess
import torch
import random
from pathlib import Path
from datetime import datetime
from PIL import Image
from diffusers import FluxPipeline
from tqdm import tqdm

# Fix CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

LOCAL_OUTPUT_DIR  = "/workspace/output/Generated_Panels"
GDRIVE_REMOTE_DIR = "gdrive:Mallya Documentary/Generated Panels"

SCRIPT_DIR = Path(__file__).parent
if (SCRIPT_DIR / "prompts.json").exists():
    PROMPTS_FILE = str(SCRIPT_DIR / "prompts.json")
else:
    PROMPTS_FILE = "/root/prompts.json"

LOG_FILE    = "/workspace/generation_log.txt"
FAILED_FILE = "/workspace/failed_panels.txt"

MODEL_ID          = "black-forest-labs/FLUX.1-dev"
IMAGE_WIDTH       = 1920
IMAGE_HEIGHT      = 1080
INFERENCE_STEPS   = 28
GUIDANCE_SCALE    = 3.5
VARIATIONS        = 2
USE_FP8           = False

NEGATIVE_PROMPT = (
    "photorealistic face, text, watermark, blurry, cartoon, anime, "
    "low quality, deformed hands, extra fingers, logos, nsfw"
)

# ─────────────────────────────────────────────
# HELPERS
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


def upload_to_gdrive(local_path: str):
    """Uploads a generated file directly to Google Drive via rclone API."""
    try:
        cmd = ["rclone", "copy", local_path, GDRIVE_REMOTE_DIR]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        log(f"⚠ Warning: Drive upload failed for {os.path.basename(local_path)}: {e}")
        return False


def already_done(local_dir: Path, panel_id: str, variation: int) -> bool:
    path = local_dir / f"{panel_id}_v{variation}.png"
    return path.exists()


def load_pipeline():
    log("Loading FLUX.1-dev pipeline...")

    # Free any lingering GPU memory before loading
    torch.cuda.empty_cache()

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    # enable_model_cpu_offload() moves model to CUDA automatically per-layer
    # Do NOT call pipe.to("cuda") — that tries to load everything at once
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if USE_FP8:
        pipe.transformer = pipe.transformer.to(torch.float8_e4m3fn)

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

    output_dir = Path(LOCAL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Local output directory: {output_dir}")
    log(f"Google Drive target: {GDRIVE_REMOTE_DIR}")

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

                    # Upload to Google Drive immediately
                    uploaded = upload_to_gdrive(str(save_path))
                    up_str = "☁ Uploaded to Drive" if uploaded else "⚠ Local only"

                    log(f"✓ Saved {filename} | {up_str} | {scene}")
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
    log(f"Images saved locally: {LOCAL_OUTPUT_DIR}")
    log(f"Images uploaded to Google Drive: {GDRIVE_REMOTE_DIR}")
    log("=" * 50)


if __name__ == "__main__":
    main()
