"""Render "Through Claude's Eyes" to an MP4 with its soundtrack.

1. Opens through-claudes-eyes.html in headless Chrome (Playwright) and pulls
   every film frame as a PNG through the page's `window.__film` hook.
2. Builds the soundtrack (film/soundtrack.py) from the same scene timeline.
3. Muxes both with ffmpeg: 8 fps frames held for 3 video frames each (24 fps
   container, still stop-motion), H.264 + AAC.

    python film/export_video.py                 # 1920x1080
    python film/export_video.py --scale 2       # 2560x1440
    python film/export_video.py --reuse-frames  # skip rendering, redo audio + mux
"""
import argparse
import base64
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

import soundtrack

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "through-claudes-eyes.html"
OUT = Path(__file__).resolve().parent / "out"
FRAMES = OUT / "frames"
VIDEO = ROOT / "through-claudes-eyes.mp4"
CHROME = "/usr/bin/google-chrome"


def render_frames(scale):
    FRAMES.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("f_*.png"):
        old.unlink()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, args=["--disable-gpu"])
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=scale)
        page.goto(HTML.as_uri())
        page.wait_for_function("window.__film && window.__film.ready", timeout=30000)
        info = page.evaluate("window.__film.info()")
        total = info["total"]
        print(f"rendering {total} frames at {info['width']}x{info['height']} ...")
        for n in range(total):
            url = page.evaluate(f"window.__film.frame({n})")
            (FRAMES / f"f_{n:04d}.png").write_bytes(base64.b64decode(url.split(",", 1)[1]))
            if n % 50 == 0:
                print(f"  frame {n + 1}/{total}")
        browser.close()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.5, help="canvas pixel ratio (1.5 = 1920x1080, max 2)")
    ap.add_argument("--reuse-frames", action="store_true")
    args = ap.parse_args()

    if args.reuse_frames and any(FRAMES.glob("f_*.png")):
        info = {"fps": soundtrack.FPS, "scenes": None}
    else:
        info = render_frames(args.scale)

    durs = [s["dur"] for s in info["scenes"]] if info["scenes"] else soundtrack.SCENE_DURS
    if durs != soundtrack.SCENE_DURS:
        print("note: page scene timing differs from soundtrack defaults; using the page's")
    wav, seconds = soundtrack.build(OUT / "soundtrack.wav", durs)
    print(f"soundtrack: {wav} ({seconds:.2f} s)")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-stats",
        "-framerate", str(info["fps"]), "-i", str(FRAMES / "f_%04d.png"),
        "-i", str(wav),
        "-vf", "fps=24,format=yuv420p",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-tune", "animation",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(VIDEO),
    ]
    subprocess.run(cmd, check=True)
    print(f"video: {VIDEO}")


if __name__ == "__main__":
    main()
