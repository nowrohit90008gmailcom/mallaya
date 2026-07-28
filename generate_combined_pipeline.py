"""
generate_combined_pipeline.py
==============================
Mallya Documentary — Interleaved Queue Pipeline (1 Image -> 1 Video)
RTX 3090 | FLUX.1-dev + SVD-XT | Direct Google Drive Upload (rclone copyto)

Queue Loop (per panel):
  1. Generate PXX_v1.png (FLUX) -> Upload to Google Drive
  2. Immediately generate PXX_v1.mp4 (8-sec SVD video) -> Upload to Google Drive
  3. Next panel!

Uses enable_sequential_cpu_offload() for both pipelines to keep VRAM < 10 GB.
"""

import json
import os
import gc
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


def create_8sec_loop(frames: list, target_duration: float = 8.0, target_fps: int = 30) -> list:
    total_needed_frames = int(target_duration * target_fps)
    ping_pong = frames + frames[-2:0:-1]
    repeats = (total_needed_frames // len(ping_pong)) + 2
    return (ping_pong * repeats)[:total_needed_frames]


# ─────────────────────────────────────────────
# MAIN PIPELINE
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

    log("=" * 60)
    log(f"INTERLEAVED QUEUE: {len(panels)} panels (1 Image -> 1 Video per step)")
    log("=" * 60)

    # 1. Load FLUX pipeline with sequential CPU offload (VRAM < 8 GB)
    log("Loading FLUX.1-dev pipeline (Sequential CPU Offload)...")
    torch.cuda.empty_cache()
    flux_pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    flux_pipe.enable_sequential_cpu_offload()
    flux_pipe.vae.enable_slicing()
    flux_pipe.vae.enable_tiling()

    # 2. Load SVD pipeline with sequential CPU offload (VRAM < 8 GB)
    log("Loading SVD-XT video pipeline (Sequential CPU Offload)...")
    svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    svd_pipe.enable_sequential_cpu_offload()
    svd_pipe.unet.enable_forward_chunking()

    log("Both pipelines loaded with CPU offloading. Starting interleaved queue...")

    # 3. Interleaved Queue Loop: 1 Image -> 1 Video
    for panel in tqdm(panels, desc="Interleaved Queue (Image -> Video)"):
        panel_id = panel["id"]
        prompt   = panel["prompt"]
        scene    = panel.get("scene", "")

        for variation in range(1, VARIATIONS + 1):
            image_name = f"{panel_id}_v{variation}.png"
            video_name = f"{panel_id}_v{variation}.mp4"

            image_path = panels_dir / image_name
            video_path = clips_dir / video_name

            # Step A: Generate PNG Image
            if not image_path.exists():
                try:
                    torch.cuda.empty_cache()
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
                    uploaded = upload_to_gdrive(str(image_path), GDRIVE_PANELS_REMOTE)
                    up_str = "☁ Uploaded" if uploaded else "⚠ Local only"
                    log(f"✓ [{panel_id}] Image created | {up_str} | {scene}")

                except Exception as e:
                    log(f"✗ [{panel_id}] Image FAILED: {e}")
                    log_fail(panel_id, variation, f"Image error: {e}")
                    torch.cuda.empty_cache()
                    continue

            # Step B: Immediately Generate 8-Second Video Clip
            if not video_path.exists():
                try:
                    torch.cuda.empty_cache()
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

                    uploaded = upload_to_gdrive(str(video_path), GDRIVE_CLIPS_REMOTE)
                    up_str = "☁ Uploaded" if uploaded else "⚠ Local only"
                    log(f"✓ [{panel_id}] Video created | {up_str} (8.0 sec MP4)")

                except Exception as e:
                    log(f"✗ [{panel_id}] Video FAILED: {e}")
                    log_fail(panel_id, variation, f"Video error: {e}")
                finally:
                    torch.cuda.empty_cache()

    log("=" * 60)
    log("ALL 77 PANELS & 8-SECOND VIDEO CLIPS COMPLETED & UPLOADED TO DRIVE!")
    log("=" * 60)


if __name__ == "__main__":
    main()
