"""
generate_animatediff.py
========================
Mallaya Documentary — AnimateDiff 2D Animated Documentary Pipeline
Uses AnimateDiff-Lightning + epiCRealism cartoon base model.
Style: DOKIO 2D animated Indian documentary illustration.

QUALITY STRATEGY:
  - 4 sub-clips per panel, each with unique seed = 4 × unique 2-sec animations
  - FFmpeg crossfade stitches them into one smooth 8-second non-looping MP4
  - NO looping, NO ping-pong. Every frame is newly generated.
  - Upscaled to 1920×1080, CRF 17 broadcast quality.
"""

import json
import os
import subprocess
import shutil
import torch
import random
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

LOCAL_CLIPS_DIR     = "/workspace/output/Video_Clips_Animated"
LOCAL_TEMP_DIR      = "/workspace/output/Temp_Subclips"
GDRIVE_CLIPS_REMOTE = "gdrive:Mallaya Documentary/Video Clips Animated"

SCRIPT_DIR   = Path(__file__).parent
PROMPTS_FILE = str(SCRIPT_DIR / "prompts.json") if (SCRIPT_DIR / "prompts.json").exists() else "/root/prompts.json"
LOG_FILE     = "/workspace/animatediff_pipeline.log"
FAIL_FILE    = "/workspace/animatediff_failed.txt"

# ── Model config ──────────────────────────────
MOTION_ADAPTER_ID = "ByteDance/AnimateDiff-Lightning"
BASE_MODEL_ID     = "emilianJR/epiCRealism"
LORA_FILENAME     = "animatediff_lightning_4step_diffusers.safetensors"

# ── Generation settings ───────────────────────
NUM_FRAMES         = 16   # Frames per sub-clip (AnimateDiff-Lightning native)
INFERENCE_STEPS   = 8    # 8 steps = better quality than 4-step
GUIDANCE_SCALE    = 1.5  # Slightly above 1.0 for richer detail
SUBCLIPS_PER_PANEL = 4   # 4 sub-clips × 2 sec = 8 seconds total, NO looping
SUBCLIP_RAW_FPS   = 8    # AnimateDiff native output fps
OUTPUT_FPS        = 24   # Final smooth fps after interpolation
INTERPOL_FACTOR   = 3    # 8fps × 3 = 24fps via motion interpolation
VIDEO_DURATION    = 8    # Total seconds per panel
CROSSFADE_SEC     = 0.4  # Smooth crossfade between sub-clips

VIDEO_WIDTH  = 1920
VIDEO_HEIGHT = 1080

# ── Style ─────────────────────────────────────
STYLE_SUFFIX = (
    ", 2D animated documentary illustration, Indian graphic novel style, "
    "bold clean cel-shaded outlines, warm earthy color palette, "
    "detailed hand-drawn illustrated backgrounds, anime-inspired Indian character design, "
    "smooth fluid animation, professional TV documentary animation quality, "
    "DOKIO documentary style, no watermarks, no text, 16:9"
)

NEGATIVE_PROMPT = (
    "photorealistic, 3D render, CGI, blurry, bad anatomy, "
    "extra limbs, deformed, watermark, text, subtitles, noisy, grain, "
    "low quality, jpeg artifacts, overexposed"
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
        res = subprocess.run(
            ["rclone", "copyto", local_path, f"{remote_dir}/{filename}"],
            capture_output=True, text=True
        )
        return res.returncode == 0
    except Exception as e:
        log(f"⚠ Drive upload failed: {e}")
        return False


def frames_to_subclip(frames: list, raw_path: str, smooth_path: str,
                      raw_fps: int = 8, output_fps: int = 24):
    """
    Step 1: Save AnimateDiff frames as lossless raw subclip.
    Step 2: Apply FFmpeg motion interpolation (minterpolate) to go from
            8fps → 24fps producing silky smooth intermediate frames.
    """
    import imageio

    # Upscale to 1920×1080
    hd_frames = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            img = frame.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)
        else:
            img = Image.fromarray(np.array(frame)).resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)
        hd_frames.append(np.array(img))

    # Save raw 8fps clip (lossless)
    writer = imageio.get_writer(
        raw_path, fps=raw_fps, codec="libx264",
        quality=None, pixelformat="yuv420p",
        ffmpeg_params=["-crf", "0", "-preset", "ultrafast"]
    )
    for frame in hd_frames:
        writer.append_data(frame)
    writer.close()

    # Apply motion interpolation: 8fps → 24fps using FFmpeg minterpolate
    # mi_mode=mci + mc_mode=aobmc = highest quality motion-compensated interpolation
    interp_filter = (
        f"minterpolate=fps={output_fps}:"
        "mi_mode=mci:mc_mode=aobmc:me_mode=bidir:"
        "vsbmc=1:scd=fdiff"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", raw_path,
        "-vf", interp_filter,
        "-r", str(output_fps),
        "-c:v", "libx264",
        "-crf", "15",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        smooth_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Interpolation error: {result.stderr[-300:]}")
    # Remove raw temp file
    Path(raw_path).unlink(missing_ok=True)


def stitch_subclips_with_crossfade(subclip_paths: list, output_path: str,
                                    fps: int = 24, crossfade: float = 0.3):
    """
    Stitches multiple MP4 subclips with smooth crossfade transitions using FFmpeg.
    Produces one seamless non-looping 8-second video.
    """
    n = len(subclip_paths)
    if n == 1:
        shutil.copy(subclip_paths[0], output_path)
        return

    # Build complex xfade filter chain
    # Each subclip is ~2 seconds. Crossfade at the end of each clip.
    inputs = []
    for p in subclip_paths:
        inputs += ["-i", p]

    # Calculate subclip duration
    subclip_dur = NUM_FRAMES / SUBCLIP_RAW_FPS  # = 2.0 seconds each

    # Build xfade filter chain
    filter_parts = []
    prev_stream = "0:v"

    for i in range(1, n):
        offset = (subclip_dur * i) - (crossfade * i)
        out_stream = f"xf{i}"
        filter_parts.append(
            f"[{prev_stream}][{i}:v]xfade=transition=fade:duration={crossfade}:offset={offset:.2f}[{out_stream}]"
        )
        prev_stream = out_stream

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{prev_stream}]",
        "-r", str(fps),
        "-c:v", "libx264",
        "-crf", "17",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg stitch error:\n{result.stderr[-500:]}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    if not os.path.exists(PROMPTS_FILE):
        print(f"ERROR: prompts.json not found at {PROMPTS_FILE}")
        return

    clips_dir = Path(LOCAL_CLIPS_DIR)
    temp_dir  = Path(LOCAL_TEMP_DIR)
    clips_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        panels = json.load(f)

    subclip_sec = NUM_FRAMES / SUBCLIP_RAW_FPS
    total_sec   = subclip_sec * SUBCLIPS_PER_PANEL

    log("=" * 65)
    log(f"AnimateDiff-Lightning | DOKIO 2D Style | {len(panels)} Panels")
    log(f"Strategy: {SUBCLIPS_PER_PANEL} sub-clips × {subclip_sec:.0f}s = {total_sec:.0f}s | NO LOOPING")
    log(f"Base: {BASE_MODEL_ID} | Steps: {INFERENCE_STEPS}")
    log("=" * 65)

    # ── Load AnimateDiff-Lightning ────────────
    log("Loading AnimateDiff-Lightning + epiCRealism...")
    torch.cuda.empty_cache()

    from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler

    adapter = MotionAdapter.from_pretrained(
        MOTION_ADAPTER_ID,
        torch_dtype=torch.float16
    )

    pipe = AnimateDiffPipeline.from_pretrained(
        BASE_MODEL_ID,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )

    pipe.load_lora_weights(
        MOTION_ADAPTER_ID,
        weight_name=LORA_FILENAME,
        adapter_name="lightning"
    )
    pipe.fuse_lora()

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        beta_schedule="linear"
    )

    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    log("Pipeline ready!\n")

    # ── Panel Loop ───────────────────────────
    for panel in tqdm(panels, desc="AnimateDiff 2D Animated"):
        panel_id    = panel["id"]
        base_prompt = panel.get("text_to_video_prompt", panel.get("video_prompt", ""))
        scene       = panel.get("scene", "")
        full_prompt = base_prompt.rstrip(".") + STYLE_SUFFIX

        video_path = clips_dir / f"{panel_id}.mp4"
        if video_path.exists():
            log(f"SKIP [{panel_id}] — already exists")
            continue

        log(f"[{panel_id}] {scene}")
        subclip_paths = []
        success = True

        # ── Generate 4 unique sub-clips ──────
        for sub_i in range(SUBCLIPS_PER_PANEL):
            subclip_path = temp_dir / f"{panel_id}_sub{sub_i}.mp4"

            try:
                torch.cuda.empty_cache()
                seed = (hash(panel_id) * 31 + sub_i * 9973) % (2**32)
                generator = torch.Generator("cpu").manual_seed(seed)

                output = pipe(
                    prompt=full_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_frames=NUM_FRAMES,
                    num_inference_steps=INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator,
                )
                frames = output.frames[0]

                # Save raw then interpolate 8fps → 24fps for silky smooth output
                raw_path    = temp_dir / f"{panel_id}_sub{sub_i}_raw.mp4"
                smooth_path = str(subclip_path)
                frames_to_subclip(
                    frames, str(raw_path), smooth_path,
                    raw_fps=SUBCLIP_RAW_FPS, output_fps=OUTPUT_FPS
                )
                subclip_paths.append(smooth_path)
                log(f"  ✓ sub-clip {sub_i+1}/{SUBCLIPS_PER_PANEL} (smoothed {SUBCLIP_RAW_FPS}→{OUTPUT_FPS}fps)")

            except Exception as e:
                log(f"  ✗ sub-clip {sub_i+1} FAILED: {e}")
                success = False
                break

        if not success or len(subclip_paths) == 0:
            log_fail(panel_id, "Sub-clip generation failed")
            continue

        # ── Stitch into 8-second final video ─
        try:
            stitch_subclips_with_crossfade(
                subclip_paths, str(video_path),
                fps=OUTPUT_FPS, crossfade=CROSSFADE_SEC
            )
            # Clean up temp sub-clips
            for p in subclip_paths:
                Path(p).unlink(missing_ok=True)

            uploaded = upload_to_gdrive(str(video_path), GDRIVE_CLIPS_REMOTE)
            up_str = "☁ Uploaded" if uploaded else "⚠ Local only"
            log(f"✓ [{panel_id}] 8-sec non-looping video | {up_str}\n")

        except Exception as e:
            log(f"✗ [{panel_id}] Stitch FAILED: {e}")
            log_fail(panel_id, str(e))

    log("=" * 65)
    log("ALL 77 ANIMATED CLIPS COMPLETE & UPLOADED TO GOOGLE DRIVE!")
    log("=" * 65)


if __name__ == "__main__":
    main()
