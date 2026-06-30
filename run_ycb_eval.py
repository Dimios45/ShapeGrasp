"""
ShapeGrasp YCB Evaluation
=========================
Runs ShapeGrasp on all 49 YCB objects in ManiSkill PickSingleYCB-v1.
Uses Qwen2-VL-7B-Instruct as the local LLM (no OpenAI key needed).
Uses semantic per-object task prompts (not just directional pick prompts).

Output: runs/YYYYMMDD_HHMMSS_shapegrasp_ycb/
  results.json   — per-object results
  results.csv    — same as CSV
  summary.txt    — human-readable table
  imgs/<obj_id>/ — RGB, mask, shapes, grasp visualisation, pipeline GIF

Usage:
  VLM_DEVICE=cuda:2 TOKENIZERS_PARALLELISM=false \\
    conda run -n graspmas python run_ycb_eval.py [--resume <run_dir>]
"""

import os, sys, json, csv, time, warnings, tempfile, shutil, argparse, textwrap
from datetime import datetime
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("VLM_DEVICE", "cuda:2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── paths ─────────────────────────────────────────────────────────────────────
SHAPEGRASP_DIR = os.path.dirname(os.path.abspath(__file__))
GRASPMAS_DIR   = "/mnt/data/mritunjoyh/GraspMAS"

sys.path.insert(0, SHAPEGRASP_DIR)
sys.path.insert(0, GRASPMAS_DIR)

# run_pipeline uses relative paths (outputs1/, etc.) so we must run from ShapeGrasp dir
os.chdir(SHAPEGRASP_DIR)

import mani_skill.envs
import gymnasium as gym
from mani_skill.utils.registration import register_env
from mani_skill_pick_YCB.pick_single_ycb import PickSingleYCBEnv

from demo import run_pipeline

# ── YCB objects + semantic task prompts ───────────────────────────────────────
TASK_MAP = {
    "002_master_chef_can":      "pour coffee from the can",
    "003_cracker_box":          "open the box to grab crackers",
    "004_sugar_box":            "pour sugar out of the box",
    "005_tomato_soup_can":      "open and pour the tomato soup",
    "006_mustard_bottle":       "squeeze mustard onto food",
    "007_tuna_fish_can":        "open the tuna can",
    "009_gelatin_box":          "open the gelatin box",
    "010_potted_meat_can":      "open and serve the potted meat",
    "011_banana":               "peel the banana",
    "013_apple":                "take a bite of the apple",
    "014_lemon":                "squeeze lemon juice",
    "015_peach":                "eat the peach",
    "017_orange":               "peel the orange",
    "021_bleach_cleanser":      "spray bleach on a surface",
    "024_bowl":                 "fill the bowl with soup",
    "025_mug":                  "drink from the mug",
    "026_sponge":               "wipe a dirty surface",
    "033_spatula":              "flip food in a pan",
    "035_power_drill":          "drill a hole in wood",
    "037_scissors":             "cut a piece of paper",
    "040_large_marker":         "write on a whiteboard",
    "042_adjustable_wrench":    "tighten a bolt",
    "043_phillips_screwdriver": "drive a Phillips head screw",
    "044_flat_screwdriver":     "drive a flathead screw",
    "048_hammer":               "hammer a nail into wood",
    "050_medium_clamp":         "clamp two pieces of wood together",
    "051_large_clamp":          "clamp large objects together",
    "052_extra_large_clamp":    "clamp extra large objects together",
    "053_mini_soccer_ball":     "kick the soccer ball",
    "054_softball":             "throw the softball",
    "055_baseball":             "throw the baseball",
    "056_tennis_ball":          "hit the tennis ball with a racket",
    "058_golf_ball":            "putt the golf ball",
    "061_foam_brick":           "stack the foam brick on others",
    "062_dice":                 "roll the dice on a table",
    "065-f_cups":               "drink from the cup",
    "065-h_cups":               "drink from the cup",
    "065-i_cups":               "drink from the cup",
    "065-j_cups":               "drink from the cup",
    "072-a_toy_airplane":       "fly the toy airplane",
    "072-b_toy_airplane":       "fly the toy airplane",
    "073-a_lego_duplo":         "stack the LEGO brick on others",
    "073-b_lego_duplo":         "stack the LEGO brick on others",
    "073-c_lego_duplo":         "stack the LEGO brick on others",
    "073-d_lego_duplo":         "stack the LEGO brick on others",
    "073-e_lego_duplo":         "stack the LEGO brick on others",
    "073-f_lego_duplo":         "stack the LEGO brick on others",
    "073-g_lego_duplo":         "stack the LEGO brick on others",
    "077_rubiks_cube":          "solve the Rubik's cube",
}

SEED = 42


# ── Pinned env (same trick as GraspMAS) ──────────────────────────────────────
_STATE  = {"obj_id": "002_master_chef_can"}
_ENV_ID = "ShapeGraspPickYCB-v1"

@register_env(_ENV_ID, max_episode_steps=50, asset_download_ids=["ycb"])
class _PinnedEnv(PickSingleYCBEnv):
    def _load_scene(self, options: dict):
        saved = self.all_model_ids
        self.all_model_ids = np.array([_STATE["obj_id"]])
        super()._load_scene(options)
        self.all_model_ids = saved


# ── helpers ───────────────────────────────────────────────────────────────────
def obj_name(obj_id: str) -> str:
    parts = obj_id.split("_", 1)
    return (parts[1] if len(parts) > 1 else obj_id).replace("_", " ")


def extract_obs(env, seed=SEED):
    """Return (rgb_uint8, depth_float32, binary_mask) from a reset env."""
    obs, _ = env.reset(seed=seed)
    sd = obs["sensor_data"]["base_camera"]

    # RGB from Color (RGBA float32 → uint8)
    rgb = (sd["Color"].cpu().squeeze().numpy()[..., :3] * 255).clip(0, 255).astype(np.uint8)

    # Depth from Position Z (negate: positive = distance from camera)
    pos = sd["Position"].cpu().squeeze().numpy()
    depth = (-pos[:, :, 2]).astype(np.float32)

    # Segmentation mask: channel 1 contains link-level actor IDs
    seg = sd["Segmentation"].cpu().squeeze().numpy()
    actor_id = env.unwrapped._objs[0].per_scene_id.item()
    mask = (seg[:, :, 1] == actor_id).astype(np.uint8)

    # Fallback: depth thresholding if seg mask is too small
    if mask.sum() < 200:
        valid = depth[(depth > 0.1) & (depth < 5.0)]
        if len(valid):
            lo, hi = np.percentile(valid, 5), np.percentile(valid, 50)
            mask = ((depth >= lo) & (depth <= hi * 0.97)).astype(np.uint8)

    return rgb, depth, mask


def save_obs_files(tmp_dir: str, obj_id: str, rgb, depth, mask):
    """Write RGB/depth/mask to disk in ShapeGrasp's expected format."""
    tag = obj_id.replace("-", "_")
    cv2.imwrite(os.path.join(tmp_dir, f"{tag}_rgb.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(tmp_dir, f"{tag}_depth.npy"), depth)
    cv2.imwrite(os.path.join(tmp_dir, f"{tag}_mask.png"),
                (mask * 255).astype(np.uint8))
    return tag


# ── GIF builder ───────────────────────────────────────────────────────────────
TARGET_W   = 512
HEADER_H   = 44
FOOTER_H   = 56
BG         = (15, 15, 25)
ACCENT     = (80, 200, 120)
WHITE      = (240, 240, 240)


def _load_font(size):
    for name in ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]:
        for base in ["/usr/share/fonts/truetype/dejavu/",
                     "/usr/share/fonts/truetype/liberation/"]:
            p = os.path.join(base, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _frame(img_path, title, caption, border=False):
    raw = Image.open(img_path).convert("RGB")
    w, h = raw.size
    raw = raw.resize((TARGET_W, int(h * TARGET_W / w)), Image.LANCZOS)
    iw, ih = raw.size
    total_h = HEADER_H + ih + FOOTER_H
    frame = Image.new("RGB", (iw, total_h), BG)

    hdr = Image.new("RGB", (iw, HEADER_H), (30, 30, 45))
    ImageDraw.Draw(hdr).text((12, 10), title, font=_load_font(18), fill=ACCENT)
    frame.paste(hdr, (0, 0))
    frame.paste(raw, (0, HEADER_H))

    ftr = Image.new("RGB", (iw, FOOTER_H), (20, 20, 35))
    lines = textwrap.wrap(caption, 70)
    d = ImageDraw.Draw(ftr)
    for i, ln in enumerate(lines[:2]):
        d.text((10, 6 + i * 20), ln, font=_load_font(13), fill=WHITE)
    frame.paste(ftr, (0, HEADER_H + ih))

    if border:
        d3 = ImageDraw.Draw(frame)
        for t in range(3):
            d3.rectangle([t, t, iw - 1 - t, total_h - 1 - t], outline=ACCENT)
    return frame


def make_gif(obj_id, task, img_dir, tag, grasp_point):
    """All files expected to be in img_dir after the copy step."""
    steps = [
        (os.path.join(img_dir, f"{tag}_rgb.png"),
         "Step 1 — Input (ManiSkill YCB)",
         f"Object: {obj_name(obj_id)}  |  Task: \"{task}\""),
        (os.path.join(img_dir, f"{tag}_masked_rgb.png"),
         "Step 2 — Segmentation Mask",
         "Binary mask from actor segmentation."),
        (os.path.join(img_dir, f"{tag}_2d_hulls.png"),
         "Step 3 — Convex Decomposition (CoACD)",
         "Object decomposed into convex parts."),
        (os.path.join(img_dir, f"{tag}_shapes.png"),
         "Step 4 — Graph Node Shapes",
         "Primitive shape fitting with geometric attributes."),
    ]
    grasp_img = os.path.join(img_dir, f"llm_{tag}_grasp.png")
    if os.path.exists(grasp_img):
        steps.append((grasp_img,
                      "Step 5 — Grasp Selection (Qwen2-VL-7B)",
                      f"Angle: {int(grasp_point[0])}°  Task: \"{task}\""))

    frames, durs = [], []
    for i, entry in enumerate(steps):
        path, title, cap = entry[0], entry[1], entry[2]
        if not os.path.exists(path):
            continue
        is_last = (i == len(steps) - 1)
        frames.append(_frame(path, title, cap, border=is_last))
        durs.append(3500 if is_last else 1800)

    if not frames:
        return None
    gif_path = os.path.join(img_dir, f"{tag}_pipeline.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=durs, loop=0, optimize=False)
    return gif_path


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None, help="Resume a previous run dir")
    ap.add_argument("--objects", nargs="+", default=None,
                    help="Run only these object IDs (space-separated)")
    args = ap.parse_args()

    # output dir
    if args.resume:
        run_dir = args.resume
        resume_path = os.path.join(run_dir, "results.json")
        done_ids = set()
        all_results = []
        if os.path.exists(resume_path):
            with open(resume_path) as f:
                all_results = json.load(f)
            done_ids = {r["obj_id"] for r in all_results if r.get("status") != "error"}
        print(f"Resuming {run_dir}  ({len(done_ids)} objects complete)")
    else:
        run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", f"{run_id}_shapegrasp_ycb")
        all_results = []
        done_ids = set()

    os.makedirs(run_dir, exist_ok=True)

    objects = args.objects if args.objects else list(TASK_MAP.keys())
    objects = [o for o in objects if o not in done_ids]

    print(f"\n{'='*60}")
    print(f"  ShapeGrasp YCB Evaluation")
    print(f"  Objects : {len(objects)}")
    print(f"  GPU     : {os.environ.get('VLM_DEVICE', 'cuda:2')}")
    print(f"  Output  : {run_dir}/")
    print(f"{'='*60}\n")

    def save_results():
        rpath = os.path.join(run_dir, "results.json")
        with open(rpath, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        cpath = os.path.join(run_dir, "results.csv")
        if all_results:
            keys = ["obj_id", "obj_name", "task", "predicted_node",
                    "angle_deg", "centroid_xy", "status", "elapsed_s"]
            with open(cpath, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader(); w.writerows(all_results)

    for idx, obj_id in enumerate(objects):
        task = TASK_MAP.get(obj_id, f"pick up the {obj_name(obj_id)}")
        name = obj_name(obj_id)
        img_dir = os.path.join(run_dir, "imgs", obj_id)
        os.makedirs(img_dir, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"  [{idx+1}/{len(objects)}]  {obj_id}")
        print(f"  Task : \"{task}\"")
        print(f"{'─'*60}")

        t0 = time.time()

        # ── build env for this object ─────────────────────────────────────────
        _STATE["obj_id"] = obj_id
        try:
            env = gym.make(
                _ENV_ID,
                obs_mode="sensor_data",
                control_mode="pd_joint_pos",
                render_mode="rgb_array",
                sensor_configs=dict(shader_pack="default", width=384, height=384),
                sim_backend="cpu",
            )
            rgb, depth, mask = extract_obs(env, seed=SEED)
            env.close()
        except Exception as e:
            print(f"  ENV ERROR: {e}")
            all_results.append(dict(obj_id=obj_id, obj_name=name, task=task,
                                    status="env_error", error=str(e),
                                    elapsed_s=round(time.time()-t0,1)))
            save_results(); continue

        # ── write temp data files ─────────────────────────────────────────────
        tmp_dir = tempfile.mkdtemp(prefix=f"sg_{obj_id}_")
        try:
            tag = save_obs_files(tmp_dir, obj_id, rgb, depth, mask)

            # ── run ShapeGrasp ────────────────────────────────────────────────
            try:
                grasp_point, vis_img, most_likely_idx = run_pipeline(
                    obj=tag,
                    task_string=task,
                    data_dir=tmp_dir + "/",
                    iter=None,
                    output_idx=1,
                    mode="2d",
                    threshold=None,
                    no_object=False,
                    model="qwen",
                    eps=0.015,
                )
                status = "ok"
            except Exception as e:
                print(f"  PIPELINE ERROR: {e}")
                import traceback; traceback.print_exc()
                all_results.append(dict(obj_id=obj_id, obj_name=name, task=task,
                                        status="pipeline_error", error=str(e),
                                        elapsed_s=round(time.time()-t0,1)))
                save_results(); continue

            elapsed = round(time.time() - t0, 1)

            # ── copy outputs to img_dir ───────────────────────────────────────
            # rgb was written to tmp_dir; all other outputs go to outputs1/
            shutil.copy(os.path.join(tmp_dir, f"{tag}_rgb.png"), img_dir)

            import glob
            sg_dirs = sorted(glob.glob(os.path.join(SHAPEGRASP_DIR, "outputs1",
                                                     f"{tag}*")))
            if sg_dirs:
                latest = sg_dirs[-1]
                for fname in os.listdir(latest):
                    src = os.path.join(latest, fname)
                    if os.path.isfile(src):
                        shutil.copy(src, img_dir)

            print(f"  Predicted Node: {most_likely_idx}  |  Angle: {int(grasp_point[0])}°"
                  f"  |  Centroid: {grasp_point[1:3].astype(int)}  |  {elapsed}s")

            # ── build per-object GIF ──────────────────────────────────────────
            gif = make_gif(obj_id, task, img_dir, tag, grasp_point)
            if gif:
                print(f"  GIF → {gif}")

            # ── record result ─────────────────────────────────────────────────
            all_results.append(dict(
                obj_id=obj_id,
                obj_name=name,
                task=task,
                predicted_node=int(most_likely_idx),
                angle_deg=int(grasp_point[0]),
                centroid_xy=grasp_point[1:3].astype(int).tolist(),
                status=status,
                elapsed_s=elapsed,
            ))

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        save_results()

    # ── summary table ─────────────────────────────────────────────────────────
    ok   = [r for r in all_results if r.get("status") == "ok"]
    errs = [r for r in all_results if r.get("status") != "ok"]
    summary = [
        f"ShapeGrasp YCB Evaluation Summary",
        f"Run dir : {run_dir}",
        f"Objects : {len(all_results)} total  |  {len(ok)} completed  |  {len(errs)} errors",
        "",
        f"{'Object':<35} {'Task':<40} {'Node':>5} {'Angle':>6} {'Time':>6}",
        "─" * 95,
    ]
    for r in all_results:
        if r.get("status") == "ok":
            summary.append(
                f"{r['obj_id']:<35} {r['task']:<40} "
                f"{r['predicted_node']:>5} {r['angle_deg']:>5}° {r['elapsed_s']:>5.1f}s"
            )
        else:
            summary.append(
                f"{r['obj_id']:<35} {r['task']:<40} {'ERROR':>5}"
                f"  {r.get('error','')[:30]}"
            )

    summary_text = "\n".join(summary)
    print("\n" + summary_text)
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    print(f"\nDone. Results saved to {run_dir}/")


if __name__ == "__main__":
    main()
