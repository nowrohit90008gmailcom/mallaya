"""
make_video_clips.py
===================
Mallya Documentary — Ken Burns Video Clip Generator
Converts FLUX-generated PNG stills into 6-10 sec MP4 clips
Uses FFmpeg zoompan filter — NO GPU needed, runs instantly

Input:  /root/gdrive/Mallya Documentary/Generated Panels/P01_v1.png ...
Output: /root/gdrive/Mallya Documentary/Video Clips/P01_v1.mp4 ...

Run AFTER generate_panels.py has finished.
"""

import subprocess
import json
import os
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

INPUT_DIR  = "/root/gdrive/Mallya Documentary/Generated Panels"
OUTPUT_DIR = "/root/gdrive/Mallya Documentary/Video Clips"
MOTIONS_FILE = "/root/panel_motions.json"   # generated from panel_motions.py

DURATION   = 8          # seconds per clip (change to 6 or 10 if needed)
FPS        = 30
WIDTH      = 1920
HEIGHT     = 1080
VARIATIONS = 2          # must match generate_panels.py

# Video encoding settings
VIDEO_CODEC   = "libx264"
PIXEL_FORMAT  = "yuv420p"
CRF           = 18          # 0=lossless, 23=default, 18=high quality
PRESET        = "slow"      # slow = better compression. Use 'fast' if time is short.

# ─────────────────────────────────────────────
# MOTION PRESETS (FFmpeg zoompan expressions)
# ─────────────────────────────────────────────
# z = zoom level (1.0 = no zoom, 1.2 = 20% in)
# x, y = pan position
# d = total frames = DURATION * FPS

def get_zoompan_filter(motion: str, duration: int, fps: int) -> str:
    """
    Returns an FFmpeg filter_complex string for the given motion type.
    All motions start slightly oversized (1.05-1.2) to allow pan room.
    """
    frames = duration * fps

    motions = {

        "slow_push_in": (
            # Zoom in slowly from 1.0 to 1.08
            f"zoompan=z='min(zoom+0.0003,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={WIDTH}x{HEIGHT}:fps={fps}"
        ),

        "slow_pull_out": (
            # Start zoomed in at 1.08, slowly pull back to 1.0
            f"zoompan=z='if(eq(on,1),1.08,max(zoom-0.0003,1.0))'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={WIDTH}x{HEIGHT}:fps={fps}"
        ),

        "pan_left_to_right": (
            # Zoom slightly (1.08), pan from left edge to right edge
            f"zoompan=z='1.08'"
            f":x='(iw-iw/zoom)*on/{frames}'"
            f":y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={WIDTH}x{HEIGHT}:fps={fps}"
        ),

        "tilt_up": (
            # Zoom 1.08, tilt upward (start at bottom, end at top)
            f"zoompan=z='1.08'"
            f":x='iw/2-(iw/zoom/2)'"
            f":y='(ih-ih/zoom)*(1-on/{frames})'"
            f":d={frames}:s={WIDTH}x{HEIGHT}:fps={fps}"
        ),

        "tilt_down": (
            # Zoom 1.08, tilt downward (start at top, end at bottom)
            f"zoompan=z='1.08'"
            f":x='iw/2-(iw/zoom/2)'"
            f":y='(ih-ih/zoom)*on/{frames}'"
            f":d={frames}:s={WIDTH}x{HEIGHT}:fps={fps}"
        ),

        "static_hold": (
            # No movement — tiny imperceptible breathe zoom
            f"zoompan=z='1.0':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={WIDTH}x{HEIGHT}:fps={fps}"
        ),

        "slow_drift": (
            # Diagonal drift — slow push-in with slight pan
            f"zoompan=z='min(zoom+0.0002,1.06)'"
            f":x='iw/2-(iw/zoom/2)+(iw*0.01*on/{frames})'"
            f":y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={WIDTH}x{HEIGHT}:fps={fps}"
        ),
    }

    return motions.get(motion, motions["static_hold"])


def get_ffmpeg_cmd(
    input_path: str,
    output_path: str,
    zoompan_filter: str,
    duration: int,
    fps: int
) -> list:
    """Build the FFmpeg command list."""
    return [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", input_path,
        "-vf", zoompan_filter,
        "-t", str(duration),
        "-r", str(fps),
        "-c:v", VIDEO_CODEC,
        "-pix_fmt", PIXEL_FORMAT,
        "-crf", str(CRF),
        "-preset", PRESET,
        "-movflags", "+faststart",   # optimize for streaming/web
        output_path
    ]


def already_done(output_dir: Path, panel_id: str, variation: int) -> bool:
    path = output_dir / f"{panel_id}_v{variation}.mp4"
    return path.exists()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Check FFmpeg is installed ──
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: FFmpeg not found. Install with: apt-get install -y ffmpeg")
        return

    # ── Load panel motions ──
    if not os.path.exists(MOTIONS_FILE):
        print(f"ERROR: panel_motions.json not found at {MOTIONS_FILE}")
        print("Run: python build_motions_json.py  first")
        return

    with open(MOTIONS_FILE, "r") as f:
        panel_motions = json.load(f)   # { "P01": "slow_push_in", "P02": "slow_pull_out", ... }

    # ── Create output directory ──
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(INPUT_DIR)

    # ── Count work ──
    total = len(panel_motions) * VARIATIONS
    skipped = sum(
        1 for pid in panel_motions
        for v in range(1, VARIATIONS + 1)
        if already_done(output_dir, pid, v)
    )
    print(f"Total clips to generate: {total}")
    print(f"Already done: {skipped} | Remaining: {total - skipped}")

    failed = []

    with tqdm(total=total - skipped, desc="Rendering clips", unit="clip") as pbar:
        for panel_id, motion_type in panel_motions.items():
            for variation in range(1, VARIATIONS + 1):

                if already_done(output_dir, panel_id, variation):
                    continue

                input_path  = str(input_dir / f"{panel_id}_v{variation}.png")
                output_path = str(output_dir / f"{panel_id}_v{variation}.mp4")

                if not os.path.exists(input_path):
                    print(f"\nWARNING: Missing input image: {input_path}")
                    failed.append(f"{panel_id}_v{variation} (no source PNG)")
                    pbar.update(1)
                    continue

                zoompan = get_zoompan_filter(motion_type, DURATION, FPS)
                cmd = get_ffmpeg_cmd(input_path, output_path, zoompan, DURATION, FPS)

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"\nERROR on {panel_id}_v{variation}:")
                    print(result.stderr[-500:])
                    failed.append(f"{panel_id}_v{variation}")
                else:
                    pass  # success, tqdm shows progress

                pbar.update(1)

    # ── Summary ──
    print("\n" + "=" * 50)
    print(f"DONE. Clips saved to: {OUTPUT_DIR}")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    else:
        print("All clips generated successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
