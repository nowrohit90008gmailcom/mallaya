"""
generate_all_clips_svd.py
=========================
Mallya Documentary — AI Video Clip Generator (ALL 77 Panels)
Generates EXACTLY 8-SECOND AI-Animated MP4 Clips directly!
Uploads directly to Google Drive via rclone API as soon as each clip is generated!

Uses Stable Video Diffusion (SVD-XT) + automatic smooth ping-pong looping 
to convert 25 AI motion frames into a complete 8.0-second MP4 clip (240 frames @ 30 FPS).
"""

import torch
import os
import json
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from datetime import datetime
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

LOCAL_INPUT_DIR   = "/workspace/output/Generated_Panels"
LOCAL_OUTPUT_DIR  = "/workspace/output/Video_Clips_SVD"
GDRIVE_REMOTE_DIR = "gdrive:Mallya Documentary/Video Clips SVD"

SCRIPT_DIR = Path(__file__).parent
if (SCRIPT_DIR / "prompts.json").exists():
    PROMPTS_FILE = str(SCRIPT_DIR / "prompts.json")
else:
    PROMPTS_FILE = "/root/prompts.json"

LOG_FILE   = "/workspace/svd_log.txt"
FAIL_FILE  = "/workspace/svd_failed.txt"

VARIATIONS         = 2
SVD_FRAMES         = 25
TARGET_DURATION    = 8.0
TARGET_FPS         = 30
DECODE_CHUNK_SIZE  = 8
MOTION_BUCKET_ID   = 100
NOISE_AUG_STRENGTH = 0.05

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


def upload_to_gdrive(local_path: str):
    """Uploads a generated video directly to Google Drive via rclone API."""
    try:
        cmd = ["rclone", "copy", local_path, GDRIVE_REMOTE_DIR]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        log(f"⚠ Warning: Drive upload failed for {os.path.basename(local_path)}: {e}")
        return False


def already_done(output_dir: Path, panel_id: str, variation: int) -> bool:
    return (output_dir / f"{panel_id}_v{variation}.mp4").exists()


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize((SVD_WIDTH, SVD_HEIGHT), Image.LANCZOS)
    return img


def create_8sec_loop(frames: list, target_duration: float = 8.0, target_fps: int = 30) -> list:
    total_needed_frames = int(target_duration * target_fps)
    ping_pong = frames + frames[-2:0:-1]
    repeats = (total_needed_frames // len(ping_pong)) + 2
    full_sequence = (ping_pong * repeats)[:total_needed_frames]
    return full_sequence


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


def main():
    if not os.path.exists(PROMPTS_FILE):
        print(f"ERROR: prompts.json not found at {PROMPTS_FILE}")
        return

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        panels = json.load(f)

    panel_ids = [p["id"] for p in panels]

    output_dir = Path(LOCAL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(LOCAL_INPUT_DIR)

    total = len(panel_ids) * VARIATIONS
    done  = sum(
        1 for pid in panel_ids
        for v in range(1, VARIATIONS + 1)
        if already_done(output_dir, pid, v)
    )
    remaining = total - done

    log(f"Total clips: {total} | Already done: {done} | Remaining: {remaining}")
    log(f"Clip spec: EXACTLY 8.0 SECONDS ({int(TARGET_DURATION * TARGET_FPS)} frames @ {TARGET_FPS} FPS)")
    log(f"Google Drive target: {GDRIVE_REMOTE_DIR}")

    if remaining == 0:
        log("All 8-second clips already generated!")
        return

    pipe = load_pipeline()

    completed = 0
    failed    = 0

    with tqdm(total=remaining, desc="Generating 8-second SVD clips", unit="clip") as pbar:
        for panel_id in panel_ids:
            for variation in range(1, VARIATIONS + 1):

                if already_done(output_dir, panel_id, variation):
                    continue

                input_path = input_dir / f"{panel_id}_v{variation}.png"

                if not input_path.exists():
                    log(f"SKIP {panel_id}_v{variation} — PNG not found in local dir")
                    log_fail(panel_id, variation, "source PNG missing")
                    failed += 1
                    pbar.update(1)
                    continue

                try:
                    image = load_image(str(input_path))
                    seed = (hash(panel_id) + variation * 1337) % (2**32)
                    generator = torch.manual_seed(seed)

                    raw_frames = pipe(
                        image,
                        num_frames=SVD_FRAMES,
                        num_inference_steps=25,
                        motion_bucket_id=MOTION_BUCKET_ID,
                        noise_aug_strength=NOISE_AUG_STRENGTH,
                        decode_chunk_size=DECODE_CHUNK_SIZE,
                        generator=generator,
                    ).frames[0]

                    video_frames_8sec = create_8sec_loop(raw_frames, TARGET_DURATION, TARGET_FPS)

                    output_path = str(output_dir / f"{panel_id}_v{variation}.mp4")
                    export_to_video(video_frames_8sec, output_path, fps=TARGET_FPS)

                    # Upload to Google Drive immediately
                    uploaded = upload_to_gdrive(output_path)
                    up_str = "☁ Uploaded to Drive" if uploaded else "⚠ Local only"

                    log(f"✓ {panel_id}_v{variation}.mp4 saved | {up_str} (8.0 sec @ {TARGET_FPS}fps)")
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

    log("=" * 60)
    log(f"DONE. Completed: {completed} | Failed: {failed}")
    log(f"8-second MP4 clips saved locally: {LOCAL_OUTPUT_DIR}")
    log(f"8-second MP4 clips uploaded to Google Drive: {GDRIVE_REMOTE_DIR}")
    log("=" * 60)


if __name__ == "__main__":
    main()
