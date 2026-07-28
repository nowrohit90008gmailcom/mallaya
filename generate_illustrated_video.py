"""
generate_illustrated_video.py
==============================
Mallaya Documentary — DOKIO-Style 2D Illustrated Video Pipeline
Generates 2D cartoon/illustrated images with FLUX.1-dev,
then applies smooth Ken Burns cinematic motion via FFmpeg
to produce perfectly smooth 8-second MP4 clips.
Uploads directly to Google Drive via rclone.
"""

import json
import os
import subprocess
import torch
import random
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

LOCAL_PANELS_DIR    = "/workspace/output/Illustrated_Panels"
LOCAL_CLIPS_DIR     = "/workspace/output/Video_Clips_Illustrated"
GDRIVE_PANELS_REMOTE = "gdrive:Mallaya Documentary/Illustrated Panels"
GDRIVE_CLIPS_REMOTE  = "gdrive:Mallaya Documentary/Video Clips Illustrated"

SCRIPT_DIR = Path(__file__).parent
PROMPTS_FILE = str(SCRIPT_DIR / "prompts.json") if (SCRIPT_DIR / "prompts.json").exists() else "/root/prompts.json"
LOG_FILE  = "/workspace/illustrated_pipeline.log"
FAIL_FILE = "/workspace/illustrated_failed.txt"

# FLUX.1-dev for 2D illustrated image generation
FLUX_MODEL_ID   = "black-forest-labs/FLUX.1-dev"
IMAGE_WIDTH     = 1920
IMAGE_HEIGHT    = 1088
INFERENCE_STEPS = 30
GUIDANCE_SCALE  = 3.5

# Video output settings
VIDEO_DURATION  = 8       # seconds
VIDEO_FPS       = 30      # smooth 30fps output
VIDEO_WIDTH     = 1920
VIDEO_HEIGHT    = 1080

# 2D Documentary Illustration Style Tag (DOKIO / Dharma animated documentary style)
STYLE_TAG = (
    "2D animated documentary illustration, Indian graphic novel style, "
    "bold clean outlines, cel-shaded flat color fills, warm earthy color palette, "
    "detailed illustrated backgrounds, hand-drawn anime-inspired Indian character art, "
    "Archer animated series quality, professional documentary animation, "
    "no photorealism, no 3D render, no watermarks, no text, 16:9 aspect ratio"
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
    try:
        filename = os.path.basename(local_path)
        remote_target = f"{remote_dir}/{filename}"
        res = subprocess.run(["rclone", "copyto", local_path, remote_target],
                             capture_output=True, text=True)
        return res.returncode == 0
    except Exception as e:
        log(f"⚠ Drive upload failed: {e}")
        return False


def apply_ken_burns(image_path: str, video_path: str, motion_type: str, duration: int = 8, fps: int = 30):
    """
    Applies smooth Ken Burns cinematic motion to a still image using FFmpeg.
    Produces perfectly smooth, professional documentary-quality video.
    """
    total_frames = duration * fps

    # FFmpeg zoompan filter per motion type
    motion_filters = {
        "slow_push_in": (
            f"scale=8000:-1,"
            f"zoompan=z='min(zoom+0.0008,1.4)':d={total_frames}"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}"
        ),
        "slow_pull_out": (
            f"scale=8000:-1,"
            f"zoompan=z='if(lte(zoom,1.0),1.4,max(zoom-0.0008,1.0))':d={total_frames}"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}"
        ),
        "pan_left_to_right": (
            f"scale=8000:-1,"
            f"zoompan=z=1.2:d={total_frames}"
            f":x='iw/2-(iw/zoom/2)+((iw/zoom/4)*on/{total_frames})'"
            f":y='ih/2-(ih/zoom/2)',scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}"
        ),
        "pan_right_to_left": (
            f"scale=8000:-1,"
            f"zoompan=z=1.2:d={total_frames}"
            f":x='iw/2-(iw/zoom/2)-((iw/zoom/4)*on/{total_frames})'"
            f":y='ih/2-(ih/zoom/2)',scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}"
        ),
        "tilt_down": (
            f"scale=8000:-1,"
            f"zoompan=z=1.2:d={total_frames}"
            f":x='iw/2-(iw/zoom/2)'"
            f":y='((ih/zoom/4)*on/{total_frames})',scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}"
        ),
        "tilt_up": (
            f"scale=8000:-1,"
            f"zoompan=z=1.2:d={total_frames}"
            f":x='iw/2-(iw/zoom/2)'"
            f":y='ih/2-(ih/zoom/2)-((ih/zoom/4)*on/{total_frames})',scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}"
        ),
        "static_hold": (
            f"scale=8000:-1,"
            f"zoompan=z='1.05+0.001*sin(2*PI*on/{total_frames})':d={total_frames}"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}"
        ),
    }

    vf = motion_filters.get(motion_type, motion_filters["slow_push_in"])

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-vf", vf,
        "-t", str(duration),
        "-r", str(fps),
        "-c:v", "libx264",
        "-crf", "17",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr[-300:]}")


# ─────────────────────────────────────────────
# MOTION TYPE MAPPING
# ─────────────────────────────────────────────

MOTION_SEQUENCE = [
    "slow_push_in", "slow_pull_out", "pan_left_to_right", "slow_push_in",
    "pan_right_to_left", "tilt_down", "slow_pull_out", "pan_left_to_right",
    "slow_push_in", "slow_pull_out", "slow_push_in", "pan_left_to_right",
    "static_hold", "slow_push_in", "tilt_down", "static_hold",
    "slow_pull_out", "tilt_up", "static_hold", "pan_left_to_right",
    "slow_pull_out", "tilt_up", "slow_push_in", "static_hold",
    "slow_push_in", "pan_left_to_right", "static_hold", "slow_push_in",
    "pan_left_to_right", "slow_pull_out", "slow_pull_out", "slow_push_in",
    "static_hold", "static_hold", "slow_push_in", "pan_left_to_right",
    "static_hold", "slow_push_in", "static_hold", "slow_push_in",
    "static_hold", "pan_left_to_right", "slow_push_in", "static_hold",
    "slow_push_in", "tilt_up", "slow_pull_out", "slow_push_in",
    "slow_push_in", "tilt_up", "static_hold", "slow_pull_out",
    "pan_right_to_left", "static_hold", "pan_left_to_right", "slow_pull_out",
    "slow_push_in", "slow_pull_out", "static_hold", "slow_push_in",
    "slow_push_in", "static_hold", "tilt_up", "slow_push_in",
    "slow_pull_out", "pan_left_to_right", "pan_right_to_left", "static_hold",
    "slow_push_in", "slow_pull_out", "static_hold", "pan_left_to_right",
    "slow_push_in", "pan_left_to_right", "slow_pull_out", "static_hold",
    "static_hold",
]


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
    log(f"STARTING ILLUSTRATED PIPELINE: {len(panels)} Panels")
    log("PHASE 1: FLUX.1-dev → 2D Illustrated Images")
    log("PHASE 2: FFmpeg Ken Burns → Smooth 8-sec Videos")
    log("=" * 60)

    # ── Load FLUX.1-dev ───────────────────────
    log("Loading FLUX.1-dev pipeline...")
    from diffusers import FluxPipeline

    flux_pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    flux_pipe.enable_model_cpu_offload()
    flux_pipe.vae.enable_slicing()
    flux_pipe.vae.enable_tiling()
    log("FLUX.1-dev ready.")

    # ── Main Loop: Image → Ken Burns Video ────
    for idx, panel in enumerate(tqdm(panels, desc="Illustrated Pipeline")):
        panel_id   = panel["id"]
        # Strip photorealistic tags from prompt and inject illustration style
        base_prompt = panel.get("image_prompt", panel.get("prompt", ""))
        # Remove old style tags and inject the 2D illustration style
        illustration_prompt = base_prompt.split("cinematic illustrated documentary panel")[0].strip().rstrip(",")
        full_prompt = f"{illustration_prompt}, {STYLE_TAG}"

        motion_type = MOTION_SEQUENCE[idx] if idx < len(MOTION_SEQUENCE) else "slow_push_in"
        camera_info = panel.get("camera_movement", {}).get("type", motion_type)
        # Use the stored camera type if available
        final_motion = camera_info if camera_info in [
            "slow_push_in","slow_pull_out","pan_left_to_right","pan_right_to_left",
            "tilt_down","tilt_up","static_hold"
        ] else motion_type

        image_name = f"{panel_id}.png"
        video_name = f"{panel_id}.mp4"
        image_path = panels_dir / image_name
        video_path = clips_dir / video_name

        # ── STEP 1: Generate Illustrated Image ──
        if not image_path.exists():
            try:
                torch.cuda.empty_cache()
                seed = random.randint(0, 2**32 - 1)
                generator = torch.Generator("cuda").manual_seed(seed)
                image = flux_pipe(
                    prompt=full_prompt,
                    width=IMAGE_WIDTH,
                    height=IMAGE_HEIGHT,
                    num_inference_steps=INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator,
                ).images[0]
                image.save(str(image_path), quality=98)
                log(f"✓ [{panel_id}] Image generated → {image_name}")
                upload_to_gdrive(str(image_path), GDRIVE_PANELS_REMOTE)
            except Exception as e:
                log(f"✗ [{panel_id}] Image FAILED: {e}")
                log_fail(panel_id, str(e))
                continue
        else:
            log(f"SKIP [{panel_id}] Image — already exists")

        # ── STEP 2: Ken Burns Video via FFmpeg ──
        if not video_path.exists():
            try:
                apply_ken_burns(str(image_path), str(video_path), final_motion, VIDEO_DURATION, VIDEO_FPS)
                uploaded = upload_to_gdrive(str(video_path), GDRIVE_CLIPS_REMOTE)
                up_str = "☁ Uploaded" if uploaded else "⚠ Local only"
                log(f"✓ [{panel_id}] Video → {video_name} [{final_motion}] | {up_str}")
            except Exception as e:
                log(f"✗ [{panel_id}] Video FAILED: {e}")
                log_fail(panel_id, str(e))
        else:
            log(f"SKIP [{panel_id}] Video — already exists")

    log("=" * 60)
    log("ALL 77 ILLUSTRATED CLIPS GENERATED & UPLOADED TO GOOGLE DRIVE!")
    log("=" * 60)


if __name__ == "__main__":
    main()
