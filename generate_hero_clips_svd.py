"""
generate_hero_clips_svd.py
==========================
Mallya Documentary — AI-Animated "Hero" Clip Generator
Uses Stable Video Diffusion (SVD-XT) to animate key still images

Run this AFTER generate_panels.py for only the most dramatic panels.
For all other panels, use make_video_clips.py (FFmpeg Ken Burns) instead.

RTX 3090 (24GB VRAM) — SVD-XT needs ~16GB, fits comfortably.

Hero panels recommended for actual AI motion:
  P02 — Airport walkaway silhouette
  P04 — Mallya in leather chair
  P10 — Goa party aerial
  P15 — Airplane falling off chart cliff
  P21 — Employee crowd outside closed building
  P31 — Beachside party fireworks aerial
  P58 — First class cabin, city lights fading
  P69 — Lone figure in empty corridor
  P74 — Cracked mirror fragments
  P76 — Empty airport terminal at dusk
"""

import torch
import json
import os
from pathlib import Path
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

INPUT_DIR  = "/root/gdrive/Mallya Documentary/Generated Panels"
OUTPUT_DIR = "/root/gdrive/Mallya Documentary/Video Clips Hero"

# Only generate AI motion for these panels (the rest use FFmpeg Ken Burns)
HERO_PANELS = [
    "P02", "P04", "P10", "P15", "P21",
    "P31", "P58", "P69", "P74", "P76"
]

VARIATIONS        = 2
FRAMES            = 25          # SVD-XT generates 25 frames
FPS               = 7           # 25 frames / 7 fps ≈ 3.5 sec. Encode at 7fps → smooth.
DECODE_CHUNK_SIZE = 8           # Reduce if OOM
MOTION_BUCKET_ID  = 127         # 0=subtle, 127=high motion, 80=medium
NOISE_AUG_STRENGTH = 0.02       # Subtle = 0.02, More motion = 0.1

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

def load_pipeline():
    print("Loading Stable Video Diffusion pipeline (SVD-XT)...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking()
    print("SVD-XT loaded.")
    return pipe


def load_image(path: str, width: int = 1024, height: int = 576) -> Image.Image:
    """SVD expects 1024x576 (16:9). We resize our 1920x1080 input."""
    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    return img


def already_done(output_dir: Path, panel_id: str, variation: int) -> bool:
    return (output_dir / f"{panel_id}_v{variation}_hero.mp4").exists()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(INPUT_DIR)

    pipe = load_pipeline()

    total = len(HERO_PANELS) * VARIATIONS
    print(f"\nGenerating AI motion for {len(HERO_PANELS)} hero panels × {VARIATIONS} variations = {total} clips")

    with tqdm(total=total, desc="Hero clips", unit="clip") as pbar:
        for panel_id in HERO_PANELS:
            for variation in range(1, VARIATIONS + 1):

                if already_done(output_dir, panel_id, variation):
                    pbar.update(1)
                    continue

                input_path = input_dir / f"{panel_id}_v{variation}.png"

                if not input_path.exists():
                    print(f"\nMissing: {input_path}")
                    pbar.update(1)
                    continue

                try:
                    image = load_image(str(input_path))

                    frames = pipe(
                        image,
                        num_frames=FRAMES,
                        num_inference_steps=25,
                        motion_bucket_id=MOTION_BUCKET_ID,
                        noise_aug_strength=NOISE_AUG_STRENGTH,
                        decode_chunk_size=DECODE_CHUNK_SIZE,
                        generator=torch.manual_seed(variation * 42),
                    ).frames[0]

                    output_path = str(output_dir / f"{panel_id}_v{variation}_hero.mp4")
                    export_to_video(frames, output_path, fps=FPS)
                    print(f"\n✓ Hero clip saved: {panel_id}_v{variation}_hero.mp4")

                except torch.cuda.OutOfMemoryError:
                    print(f"\n✗ OOM on {panel_id}_v{variation}. Try reducing DECODE_CHUNK_SIZE to 4.")
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n✗ Error on {panel_id}_v{variation}: {e}")

                finally:
                    pbar.update(1)
                    torch.cuda.empty_cache()

    print(f"\nDone! Hero clips saved to: {OUTPUT_DIR}")
    print("Note: Hero clips are ~3.5 sec. Loop them 2x in your editor for 7 sec.")


if __name__ == "__main__":
    main()
