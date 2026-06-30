"""
Build an annotated GIF showing the ShapeGrasp pipeline steps.
Usage: conda run -n graspmas python make_gif.py [output_dir] [out_gif]
"""
import sys
import os
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "outputs1/knife1_2d_20260622_202844"
GIF_OUT = sys.argv[2] if len(sys.argv) > 2 else "shapegrasp_pipeline.gif"

TARGET_W = 640
HEADER_H = 56
FOOTER_H = 72
BG = (15, 15, 25)
ACCENT = (80, 200, 120)
WHITE = (240, 240, 240)
DURATION_MS = 2000   # ms per frame (last frame held longer)
LAST_FRAME_MS = 4000


def load_font(size):
    for name in ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "LiberationSans-Bold.ttf"]:
        for base in ["/usr/share/fonts/truetype/dejavu/",
                     "/usr/share/fonts/truetype/liberation/",
                     "/usr/share/fonts/"]:
            path = os.path.join(base, name)
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def resize_keep_aspect(img, target_w):
    w, h = img.size
    new_h = int(h * target_w / w)
    return img.resize((target_w, new_h), Image.LANCZOS)


def make_frame(img_path, title, caption, is_last=False):
    raw = Image.open(img_path).convert("RGB")
    raw = resize_keep_aspect(raw, TARGET_W)
    iw, ih = raw.size

    total_h = HEADER_H + ih + FOOTER_H
    frame = Image.new("RGB", (TARGET_W, total_h), BG)

    # header bar
    hdr = Image.new("RGB", (TARGET_W, HEADER_H), (30, 30, 45))
    d = ImageDraw.Draw(hdr)
    fnt_title = load_font(20)
    d.text((16, 14), title, font=fnt_title, fill=ACCENT)
    frame.paste(hdr, (0, 0))

    # image
    frame.paste(raw, (0, HEADER_H))

    # footer
    ftr_y = HEADER_H + ih
    ftr = Image.new("RGB", (TARGET_W, FOOTER_H), (20, 20, 35))
    d2 = ImageDraw.Draw(ftr)
    fnt_cap = load_font(14)
    lines = textwrap.wrap(caption, width=72)
    for i, line in enumerate(lines[:3]):
        d2.text((12, 8 + i * 20), line, font=fnt_cap, fill=WHITE)
    frame.paste(ftr, (0, ftr_y))

    # green border on last frame (grasp result)
    if is_last:
        d3 = ImageDraw.Draw(frame)
        for t in range(4):
            d3.rectangle([t, t, TARGET_W - 1 - t, total_h - 1 - t], outline=ACCENT)

    return frame


steps = [
    (
        os.path.join("data", "knife_rgb.png"),
        "Step 1 — Input RGB Image",
        "Raw top-down RGB image of the knife object.",
    ),
    (
        os.path.join(OUT_DIR, "knife_masked_rgb.png"),
        "Step 2 — Segmentation Mask Applied",
        "Binary mask isolates the object from the background.",
    ),
    (
        os.path.join(OUT_DIR, "knife_2d_hulls.png"),
        "Step 3 — Convex Decomposition (CoACD)",
        "Object decomposed into convex parts via approximate convex decomposition.",
    ),
    (
        os.path.join(OUT_DIR, "knife_shapes.png"),
        "Step 4 — Graph Node Shapes",
        "Each convex part fitted to a primitive shape (rect/ellipse/triangle) with attributes.",
    ),
    (
        os.path.join(OUT_DIR, "llm_knife_grasp.png"),
        "Step 5 — Grasp Selection (Qwen2-VL-7B)",
        'rect0=handle (0.7) > rect1=blade (0.3)  |  Task: "cut bread"  |  Predicted: rect0 @ angle 171°',
        True,
    ),
]

frames = []
durations = []
for entry in steps:
    path, title, caption = entry[0], entry[1], entry[2]
    is_last = len(entry) == 4
    if not os.path.exists(path):
        print(f"  skipping missing: {path}")
        continue
    print(f"  adding: {title}")
    frames.append(make_frame(path, title, caption, is_last=is_last))
    durations.append(LAST_FRAME_MS if is_last else DURATION_MS)

if not frames:
    print("No frames — check paths.")
    sys.exit(1)

frames[0].save(
    GIF_OUT,
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0,
    optimize=False,
)
print(f"\nSaved: {GIF_OUT}  ({len(frames)} frames)")
