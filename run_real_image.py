"""
ShapeGrasp on a Real Image
===========================
Auto-segments the target object from any photo using GroundingDINO + SAM,
optionally estimates depth with MiDAS, then runs ShapeGrasp.

Usage:
  conda run -n graspmas python run_real_image.py \\
      --image /path/to/photo.jpg \\
      --object "knife" \\
      --task "cut bread" \\
      [--mode 2d|3d] \\
      [--device cuda:0]

Outputs are saved to real_outputs/<object>_<timestamp>/
"""

import os, sys, argparse, shutil, tempfile, datetime
import numpy as np
import cv2
from PIL import Image

SHAPEGRASP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SHAPEGRASP_DIR)
os.chdir(SHAPEGRASP_DIR)

# ── model paths (already downloaded) ─────────────────────────────────────────
GDINO_ID = "/mnt/data/mritunjoyh/models/grounding-dino-tiny"
SAM_ID   = "/mnt/data/mritunjoyh/models/sam-vit-base"
MIDAS_ID = "/mnt/data/mritunjoyh/models/dpt-hybrid-midas"


# ── segmentation helpers ──────────────────────────────────────────────────────
def load_seg_models(device):
    from transformers import pipeline, AutoModelForMaskGeneration, AutoProcessor
    print("Loading GroundingDINO...")
    detector = pipeline(
        model=GDINO_ID,
        task="zero-shot-object-detection",
        device=device,
    )
    print("Loading SAM...")
    seg_model = AutoModelForMaskGeneration.from_pretrained(SAM_ID).to(device)
    seg_proc  = AutoProcessor.from_pretrained(SAM_ID)
    return detector, seg_model, seg_proc


def detect_and_mask(pil_img, obj_text, detector, seg_model, seg_proc, device,
                    det_threshold=0.30):
    """
    Returns (binary_mask, best_box) where binary_mask is HxW uint8 (0/1).
    Raises RuntimeError if object not detected.
    """
    label = obj_text if obj_text.endswith(".") else obj_text + "."
    results = detector(pil_img, candidate_labels=[label], threshold=det_threshold)

    if not results:
        # try lower threshold
        results = detector(pil_img, candidate_labels=[label], threshold=det_threshold * 0.5)
    if not results:
        raise RuntimeError(
            f"GroundingDINO could not detect '{obj_text}' in the image. "
            "Try a different object name or lower --det_threshold."
        )

    # take highest-score detection
    best = max(results, key=lambda r: r["score"])
    box  = [best["box"]["xmin"], best["box"]["ymin"],
            best["box"]["xmax"], best["box"]["ymax"]]
    print(f"  Detected '{obj_text}' (score={best['score']:.2f})  box={[int(v) for v in box]}")

    h_img, w_img = np.array(pil_img).shape[:2]

    inputs  = seg_proc(images=pil_img, input_boxes=[[box]], return_tensors="pt").to(device)
    outputs = seg_model(**inputs)

    # Bypass post_process_masks (its return shape varies by transformers version).
    # Instead: pick best mask logit by IoU score, sigmoid+threshold, bilinear upsample.
    import torch
    import torch.nn.functional as F
    pred = outputs.pred_masks[0]           # (num_masks, H_low, W_low)
    pred = pred.squeeze(0) if pred.ndim == 4 else pred   # drop extra batch dim
    iou  = outputs.iou_scores[0].flatten()
    best_idx = int(iou.argmax().item())
    best_idx = min(best_idx, pred.shape[0] - 1)
    logit = pred[best_idx]                 # (H_low, W_low)
    prob  = torch.sigmoid(logit).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    upsampled = F.interpolate(prob, size=(h_img, w_img), mode="bilinear",
                              align_corners=False).squeeze()
    mask_np = (upsampled.detach().cpu().numpy() > 0.5).astype(np.uint8)

    # fallback: if SAM returns an empty mask, use bounding box rectangle
    if mask_np.sum() == 0:
        print("  WARNING: SAM returned empty mask — falling back to bounding box crop")
        mask_np = np.zeros((h_img, w_img), dtype=np.uint8)
        x1, y1, x2, y2 = [int(v) for v in box]
        mask_np[y1:y2, x1:x2] = 1

    return mask_np, box


def estimate_depth(pil_img, device):
    """Returns a float32 depth array (HxW) using MiDAS."""
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    import torch
    print("Loading MiDAS depth estimator...")
    proc  = DPTImageProcessor.from_pretrained(MIDAS_ID)
    model = DPTForDepthEstimation.from_pretrained(MIDAS_ID).to(device)
    model.eval()
    inputs = proc(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth
    depth = pred.squeeze().cpu().numpy()
    # resize to original image size
    h, w = np.array(pil_img).shape[:2]
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth.astype(np.float32)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",  required=True, help="Path to input image (jpg/png)")
    ap.add_argument("--object", required=True, help="Object name for detection, e.g. 'knife'")
    ap.add_argument("--task",   required=True, help="Task string, e.g. 'cut bread'")
    ap.add_argument("--mode",   default="2d",  choices=["2d", "3d"],
                    help="2d (no depth) or 3d (estimate depth with MiDAS)")
    ap.add_argument("--device", default=os.environ.get("VLM_DEVICE", "cuda:0"))
    ap.add_argument("--det_threshold", type=float, default=0.30,
                    help="GroundingDINO detection confidence threshold")
    ap.add_argument("--threshold", type=float, default=None,
                    help="CoACD decomposition threshold (default: auto)")
    ap.add_argument("--eps",    type=float, default=0.015,
                    help="Contour approximation epsilon")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: image not found: {args.image}")
        sys.exit(1)

    # output dir
    ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    obj_tag = args.object.replace(" ", "_").replace("/", "_")
    out_root = os.path.join("real_outputs", f"{obj_tag}_{ts}")
    os.makedirs(out_root, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ShapeGrasp — Real Image")
    print(f"  Image  : {args.image}")
    print(f"  Object : {args.object}")
    print(f"  Task   : {args.task}")
    print(f"  Mode   : {args.mode}")
    print(f"  Device : {args.device}")
    print(f"  Output : {out_root}/")
    print(f"{'='*60}\n")

    from PIL import ImageOps
    pil_img = ImageOps.exif_transpose(Image.open(args.image).convert("RGB"))
    rgb_np  = np.array(pil_img)
    print(f"  Image size: {pil_img.width}×{pil_img.height} px")

    # ── step 1: detect + mask ─────────────────────────────────────────────────
    print("Step 1 — Detecting and segmenting object...")
    detector, seg_model, seg_proc = load_seg_models(args.device)
    mask, box = detect_and_mask(pil_img, args.object, detector, seg_model,
                                seg_proc, args.device, args.det_threshold)

    # visualise detection
    vis = rgb_np.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    masked_vis = rgb_np.copy()
    masked_vis[mask == 0] = (masked_vis[mask == 0] * 0.3).astype(np.uint8)
    cv2.imwrite(os.path.join(out_root, "detection.png"),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_root, "segmentation.png"),
                cv2.cvtColor(masked_vis, cv2.COLOR_RGB2BGR))
    print(f"  Mask area: {mask.sum()} px  ({100*mask.sum()/mask.size:.1f}% of image)")
    if mask.sum() == 0:
        raise RuntimeError("Object mask is empty — cannot run ShapeGrasp. "
                           "Try lowering --det_threshold or checking the image.")

    # free seg models before loading VLM
    del detector, seg_model, seg_proc
    import torch; torch.cuda.empty_cache()

    # ── step 2: depth (optional) ──────────────────────────────────────────────
    depth = None
    if args.mode == "3d":
        print("\nStep 2 — Estimating depth with MiDAS...")
        depth = estimate_depth(pil_img, args.device)
        del torch; import torch; torch.cuda.empty_cache()

    # ── step 3: save files for ShapeGrasp ────────────────────────────────────
    print("\nStep 3 — Running ShapeGrasp pipeline...")
    tmp_dir = tempfile.mkdtemp(prefix=f"sg_real_{obj_tag}_")
    try:
        cv2.imwrite(os.path.join(tmp_dir, f"{obj_tag}_rgb.png"),
                    cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(tmp_dir, f"{obj_tag}_mask.png"),
                    (mask * 255).astype(np.uint8))
        if depth is not None:
            np.save(os.path.join(tmp_dir, f"{obj_tag}_depth.npy"), depth)

        from demo import run_pipeline
        grasp_point, vis_img, most_likely_idx = run_pipeline(
            obj=obj_tag,
            task_string=args.task,
            data_dir=tmp_dir + "/",
            iter=None,
            output_idx=1,
            mode=args.mode,
            threshold=args.threshold,
            no_object=False,
            model="qwen",
            eps=args.eps,
        )

        # ── step 4: collect outputs ───────────────────────────────────────────
        import glob
        sg_dirs = sorted(glob.glob(os.path.join("outputs1", f"{obj_tag}*")))
        if sg_dirs:
            for fname in os.listdir(sg_dirs[-1]):
                src = os.path.join(sg_dirs[-1], fname)
                if os.path.isfile(src):
                    shutil.copy(src, out_root)
        shutil.copy(os.path.join(tmp_dir, f"{obj_tag}_rgb.png"), out_root)

        print(f"\n{'='*60}")
        print(f"  RESULT")
        print(f"  Predicted node : {most_likely_idx}")
        print(f"  Grasp angle    : {int(grasp_point[0])}°")
        print(f"  Centroid (px)  : {grasp_point[1:3].astype(int)}")
        print(f"  Output dir     : {out_root}/")
        print(f"{'='*60}")

        # ── step 5: build pipeline GIF ────────────────────────────────────────
        _make_summary_gif(out_root, obj_tag, args.object, args.task,
                          grasp_point, most_likely_idx)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _make_summary_gif(out_dir, tag, obj_name, task, grasp_point, best_node):
    import textwrap
    from PIL import ImageDraw, ImageFont
    TARGET_W = 640
    HEADER_H = 44
    FOOTER_H = 52
    BG, ACCENT, WHITE = (15,15,25), (80,200,120), (240,240,240)

    def load_font(sz):
        for n in ["DejaVuSans-Bold.ttf","DejaVuSans.ttf"]:
            p = f"/usr/share/fonts/truetype/dejavu/{n}"
            if os.path.exists(p):
                return ImageFont.truetype(p, sz)
        return ImageFont.load_default()

    def frame(path, title, cap, border=False):
        raw = Image.open(path).convert("RGB")
        w, h = raw.size
        raw = raw.resize((TARGET_W, int(h*TARGET_W/w)), Image.LANCZOS)
        iw, ih = raw.size
        f = Image.new("RGB", (iw, HEADER_H+ih+FOOTER_H), BG)
        hdr = Image.new("RGB", (iw, HEADER_H), (30,30,45))
        ImageDraw.Draw(hdr).text((12,10), title, font=load_font(17), fill=ACCENT)
        f.paste(hdr, (0,0)); f.paste(raw, (0, HEADER_H))
        ftr = Image.new("RGB", (iw, FOOTER_H), (20,20,35))
        d = ImageDraw.Draw(ftr)
        for i, ln in enumerate(textwrap.wrap(cap, 75)[:2]):
            d.text((10, 5+i*20), ln, font=load_font(13), fill=WHITE)
        f.paste(ftr, (0, HEADER_H+ih))
        if border:
            d2 = ImageDraw.Draw(f)
            for t in range(3):
                d2.rectangle([t,t,iw-1-t,HEADER_H+ih+FOOTER_H-1-t], outline=ACCENT)
        return f

    steps = [
        (f"{out_dir}/{tag}_rgb.png",       "Step 1 — Input Image",
         f"Object: '{obj_name}'  |  Task: \"{task}\""),
        (f"{out_dir}/segmentation.png",    "Step 2 — GroundingDINO + SAM Segmentation",
         f"Object auto-detected and masked."),
        (f"{out_dir}/{tag}_2d_hulls.png",  "Step 3 — Convex Decomposition (CoACD)",
         "Object decomposed into convex parts."),
        (f"{out_dir}/{tag}_shapes.png",    "Step 4 — Graph Node Shapes",
         "Primitive shape fitting with geometric attributes."),
        (f"{out_dir}/llm_{tag}_grasp.png", "Step 5 — Grasp Selection (Qwen2-VL-7B)",
         f"Node {best_node}  |  Angle: {int(grasp_point[0])}°  |  Task: \"{task}\""),
    ]

    frames, durs = [], []
    for i, (path, title, cap) in enumerate(steps):
        if os.path.exists(path):
            frames.append(frame(path, title, cap, border=(i==len(steps)-1)))
            durs.append(3500 if i == len(steps)-1 else 1800)

    if frames:
        gif = f"{out_dir}/{tag}_pipeline.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=durs, loop=0, optimize=False)
        print(f"  Pipeline GIF   : {gif}")

    # also save a side-by-side: original | grasp result
    orig_path  = f"{out_dir}/{tag}_rgb.png"
    grasp_path = f"{out_dir}/llm_{tag}_grasp.png"
    if os.path.exists(orig_path) and os.path.exists(grasp_path):
        orig  = Image.open(orig_path).convert("RGB")
        grasp = Image.open(grasp_path).convert("RGB")
        h = min(orig.height, grasp.height, 480)
        orig  = orig.resize((int(orig.width  * h / orig.height),  h), Image.LANCZOS)
        grasp = grasp.resize((int(grasp.width * h / grasp.height), h), Image.LANCZOS)
        side = Image.new("RGB", (orig.width + grasp.width + 8, h), (40, 40, 40))
        side.paste(orig, (0, 0)); side.paste(grasp, (orig.width + 8, 0))
        side.save(f"{out_dir}/{tag}_result.png")
        print(f"  Side-by-side   : {out_dir}/{tag}_result.png")


if __name__ == "__main__":
    main()
