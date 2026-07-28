"""
generate_text_to_video.py
=========================
Mallaya Documentary — 2D Animated Documentary Text-to-Video Pipeline
Generates 8-second animated video clips using CogVideoX-5b.
Style: 2D animated illustration, DOKIO / Indian graphic novel documentary style.
Uploads directly to Google Drive via rclone.
"""

import json
import os
import subprocess
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

LOCAL_CLIPS_DIR     = "/workspace/output/Video_Clips_Animated"
GDRIVE_CLIPS_REMOTE = "gdrive:Mallya Documentary/Video Clips Animated"

SCRIPT_DIR = Path(__file__).parent
if (SCRIPT_DIR / "prompts.json").exists():
    PROMPTS_FILE = str(SCRIPT_DIR / "prompts.json")
else:
    PROMPTS_FILE = "/root/prompts.json"

LOG_FILE  = "/workspace/t2v_pipeline.log"
FAIL_FILE = "/workspace/t2v_failed.txt"

# Model: CogVideoX-5b — best quality open-source T2V model for 24 GB VRAM
MODEL_ID          = "THUDM/CogVideoX-5b"
NUM_FRAMES        = 49    # Native CogVideoX frame count (~6 sec at native fps)
INFERENCE_STEPS   = 50    # Higher = better quality & smoother motion
GUIDANCE_SCALE    = 6.0
TARGET_FPS        = 16    # Output FPS — smoother playback

# 2D Animated Documentary Style Prefix (DOKIO style)
# Prepended to every prompt so CogVideoX generates in the correct art style
STYLE_PREFIX = (
    "2D animated documentary illustration, Indian graphic novel style, "
    "bold clean cel-shaded outlines, warm earthy color palette, "
    "detailed hand-drawn illustrated backgrounds, anime-inspired Indian character art, "
    "smooth cinematic animation, professional documentary animation quality, "
    "no photorealism, no 3D render, "
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_fail(panel_id: str, err: str):
    with open(FAIL_FILE, "a", encoding="utf-8") as f:
        f.write(f"{panel_id} | {err}\n")


def upload_to_gdrive(local_path: str, remote_dir: str):
    """Uploads a generated video file directly to Google Drive via rclone API."""
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


def export_high_quality_video(frames: list, output_path: str, fps: int = 24):
    """Encodes frames into 1080p high-bitrate H.264 MP4."""
    import imageio

    hd_frames = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            img = frame.resize((1920, 1080), Image.LANCZOS)
            hd_frames.append(np.array(img))
        else:
            img = Image.fromarray(frame).resize((1920, 1080), Image.LANCZOS)
            hd_frames.append(np.array(img))

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=None,
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "17", "-preset", "slow"]
    )
    for frame in hd_frames:
        writer.append_data(frame)
    writer.close()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    if not os.path.exists(PROMPTS_FILE):
        print(f"ERROR: prompts.json not found at {PROMPTS_FILE}")
        return

    clips_dir = Path(LOCAL_CLIPS_DIR)
    clips_dir.mkdir(parents=True, exist_ok=True)

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        panels = json.load(f)

    log("=" * 60)
    log(f"STARTING PURE TEXT-TO-VIDEO PIPELINE: {len(panels)} Panels")
    log("=" * 60)

    log("Loading CogVideoX-5b pipeline (best quality, 24 GB VRAM, CPU offload)...")
    torch.cuda.empty_cache()

    from diffusers import CogVideoXPipeline

    pipe = CogVideoXPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    log("CogVideoX-5b pipeline ready. Generating 8-second clips from detailed text prompts...")

    for panel in tqdm(panels, desc="CogVideoX-5b Animated T2V"):
        panel_id   = panel["id"]
        # Prepend 2D animated style prefix to every prompt
        base_prompt = panel.get("text_to_video_prompt", panel.get("video_prompt", ""))
        v_prompt    = STYLE_PREFIX + base_prompt
        scene       = panel.get("scene", "")

        video_name = f"{panel_id}.mp4"
        video_path = clips_dir / video_name

        if video_path.exists():
            log(f"SKIP {video_name} — already exists")
            continue

        try:
            torch.cuda.empty_cache()
            log(f"[{panel_id}] Generating animated clip...")

            # Unique seed per panel for variety
            seed = (hash(panel_id) + 999) % (2**32)
            frames = pipe(
                prompt=v_prompt,
                num_videos_per_prompt=1,
                num_inference_steps=INFERENCE_STEPS,
                num_frames=NUM_FRAMES,
                guidance_scale=GUIDANCE_SCALE,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).frames[0]

            export_high_quality_video(frames, str(video_path), fps=TARGET_FPS)

            uploaded = upload_to_gdrive(str(video_path), GDRIVE_CLIPS_REMOTE)
            up_str = "☁ Uploaded" if uploaded else "⚠ Local only"
            log(f"✓ [{panel_id}] Text-to-Video created: {video_name} | {up_str} | {scene}")

        except Exception as e:
            log(f"✗ [{panel_id}] Text-to-Video FAILED: {e}")
            log_fail(panel_id, str(e))
        finally:
            torch.cuda.empty_cache()

    log("=" * 60)
    log("ALL 77 PURE TEXT-TO-VIDEO CLIPS COMPLETED & UPLOADED TO DRIVE!")
    log("=" * 60)


if __name__ == "__main__":
    main()
