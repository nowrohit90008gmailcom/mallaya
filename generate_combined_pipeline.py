"""
generate_combined_pipeline.py
==============================
Mallya Documentary — Complete Image + 8-Second Video Pipeline
RTX 3090 (24GB VRAM) | FLUX.1-dev + SVD-XT | Direct Google Drive Upload

For EVERY panel:
  1. Generates image PNG (FLUX.1-dev) → Uploads to Google Drive
  2. Animates into 8.0-second MP4 clip (SVD-XT) → Uploads to Google Drive
  3. Moves to next panel!
"""

import json
import os
import subprocess
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from diffusers import FluxPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

LOCAL_PANELS_DIR   = "/workspace/output/Generated_Panels"
LOCAL_CLIPS_DIR    = "/workspace/output/Video_Clips_SVD"

GDRIVE_PANELS_REMOTE = "gdrive:Mallya Documentary/Generated Panels"
GDRIVE_CLIPS_REMOTE  = "gdrive:Mallya Documentary/Video Clips SVD"

SCRIPT_DIR = Path(__file__).parent
if (SCRIPT_DIR / "prompts.json").exists():
    PROMPTS_FILE = str(SCRIPT_DIR / "prompts.json")
else:
    PROMPTS_FILE = "/root/prompts.json"

LOG_FILE   = "/workspace/combined_pipeline.log"
FAIL_FILE  = "/workspace/combined_failed.txt"

FLUX_MODEL_ID     = "black-forest-labs/FLUX.1-dev"
IMAGE_WIDTH       = 1920
IMAGE_HEIGHT      = 1088
INFERENCE_STEPS   = 28
GUIDANCE_SCALE    = 3.5
VARIATIONS        = 1

SVD_FRAMES         = 25
TARGET_DURATION    = 8.0
TARGET_FPS         = 30
DECODE_CHUNK_SIZE  = 8
MOTION_BUCKET_ID   = 100
NOISE_AUG_STRENGTH = 0.05
SVD_WIDTH          = 1024
SVD_HEIGHT         = 576

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


def upload_to_gdrive(local_path: str, remote_dir: str):
    """Uploads a file directly to Google Drive via rclone API."""
    try:
        filename = os.path.basename(local_path)
        remote_target = f"{remote_dir}/{filename}"
        cmd = ["rclone", "copyto", local_path, remote_target]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return True
        else:
            log(f"⚠ Drive upload error for {filename}: {res.stderr.strip()}")
            return False
    except Exception as e:
        log(f"⚠ Drive upload failed for {os.path.basename(local_path)}: {e}")
        return False


def load_flux_pipeline():
    log("Loading FLUX.1-dev pipeline...")
    torch.cuda.empty_cache()
    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def load_svd_pipeline():
    log("Loading SVD-XT video pipeline...")
    torch.cuda.empty_cache()
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking()
    return pipe


def create_8sec_loop(frames: list, target_duration: float = 8.0, target_fps: int = 30) -> list:
    total_needed_frames = int(target_duration * target_fps)
    ping_pong = frames + frames[-2:0:-1]
    repeats = (total_needed_frames // len(ping_pong)) + 2
    return (ping_pong * repeats)[:total_needed_frames]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if not os.path.exists(PROMPTS_FILE):
        print(f"ERROR: prompts.json not found at {PROMPTS_FILE}")
        return

    panels_dir = Path(LOCAL_PANELS_DIR)
    clips_dir  = Path(LOCAL_CLIPS_DIR)
    panels_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        panels = json.load(f)

    log(f"Starting combined pipeline for {len(panels)} panels × {VARIATIONS} variations = {len(panels) * VARIATIONS} total items.")

    # Load FLUX model
    flux_pipe = load_flux_pipeline()
    svd_pipe  = None  # Lazy load SVD on first video step

    for panel in tqdm(panels, desc="Processing documentary panels"):
        panel_id = panel["id"]
        prompt   = panel["prompt"]
        scene    = panel.get("scene", "")

        for variation in range(1, VARIATIONS + 1):
            image_name = f"{panel_id}_v{variation}.png"
            video_name = f"{panel_id}_v{variation}.mp4"

            image_path = panels_dir / image_name
            video_path = clips_dir / video_name

            # ── 1. GENERATE IMAGE IF NOT EXISTS ──
            if not image_path.exists():
                try:
                    seed = random.randint(0, 2**32 - 1)
                    generator = torch.Generator("cuda").manual_seed(seed)
                    image = flux_pipe(
                        prompt=prompt,
                        width=IMAGE_WIDTH,
                        height=IMAGE_HEIGHT,
                        num_inference_steps=INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        generator=generator,
                    ).images[0]

                    image.save(str(image_path), format="PNG", optimize=False)
                    upload_to_gdrive(str(image_path), GDRIVE_PANELS_REMOTE)
                    log(f"✓ Image created: {image_name} | {scene}")

                except Exception as e:
                    log(f"✗ Image FAILED: {image_name} | {e}")
                    log_fail(panel_id, variation, f"Image error: {e}")
                    continue

            # ── 2. GENERATE 8-SEC VIDEO CLIP IF NOT EXISTS ──
            if not video_path.exists():
                try:
                    if svd_pipe is None:
                        svd_pipe = load_svd_pipeline()

                    raw_img = Image.open(str(image_path)).convert("RGB")
                    resized_img = raw_img.resize((SVD_WIDTH, SVD_HEIGHT), Image.LANCZOS)

                    seed = (hash(panel_id) + variation * 1337) % (2**32)
                    generator = torch.manual_seed(seed)

                    raw_frames = svd_pipe(
                        resized_img,
                        num_frames=SVD_FRAMES,
                        num_inference_steps=25,
                        motion_bucket_id=MOTION_BUCKET_ID,
                        noise_aug_strength=NOISE_AUG_STRENGTH,
                        decode_chunk_size=DECODE_CHUNK_SIZE,
                        generator=generator,
                    ).frames[0]

                    frames_8sec = create_8sec_loop(raw_frames, TARGET_DURATION, TARGET_FPS)
                    export_to_video(frames_8sec, str(video_path), fps=TARGET_FPS)

                    upload_to_gdrive(str(video_path), GDRIVE_CLIPS_REMOTE)
                    log(f"✓ Video clip created: {video_name} (8.0 sec MP4)")

                except Exception as e:
                    log(f"✗ Video FAILED: {video_name} | {e}")
                    log_fail(panel_id, variation, f"Video error: {e}")

                finally:
                    torch.cuda.empty_cache()

    log("=" * 60)
    log("ALL PANELS & 8-SECOND VIDEO CLIPS GENERATED & UPLOADED TO DRIVE!")
    log("=" * 60)


if __name__ == "__main__":
    main()
