"""
ShapeGrasp YCB — Motion-Planning Video Renderer
================================================
Loads ShapeGrasp predictions from a previous eval run (results.json),
then for each object:
  1. Spawns the ManiSkill PickSingleYCB env
  2. Builds a 6-DoF grasp pose using:
       - GT object position (from sim)
       - ShapeGrasp predicted angle (gripper closing direction)
  3. Executes approach → close gripper → lift via PandaArmMotionPlanningSolver
  4. Records an MP4 video

Output: <eval_run_dir>/videos/<obj_id>.mp4
        <eval_run_dir>/video_results.json

Usage:
  VLM_DEVICE=cuda:2 conda run -n graspmas python render_ycb_videos.py \\
      --run runs/20260622_211951_shapegrasp_ycb [--resume]
"""

import os, sys, json, time, warnings, argparse
from datetime import datetime
import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

SHAPEGRASP_DIR = os.path.dirname(os.path.abspath(__file__))
GRASPMAS_DIR   = "/mnt/data/mritunjoyh/GraspMAS"
sys.path.insert(0, SHAPEGRASP_DIR)
sys.path.insert(0, GRASPMAS_DIR)
os.chdir(SHAPEGRASP_DIR)

import sapien
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.utils.structs import Pose
from mani_skill.utils.registration import register_env
import mani_skill.envs

from mani_skill_pick_YCB.pick_single_ycb import PickSingleYCBEnv

SEED = 42

# ── Pinned env ────────────────────────────────────────────────────────────────
_STATE  = {"obj_id": "002_master_chef_can"}
_ENV_ID = "ShapeGraspRender-v1"

@register_env(_ENV_ID, max_episode_steps=100, asset_download_ids=["ycb"])
class _PinnedEnv(PickSingleYCBEnv):
    def _load_scene(self, options: dict):
        saved = self.all_model_ids
        self.all_model_ids = np.array([_STATE["obj_id"]])
        super()._load_scene(options)
        self.all_model_ids = saved


# ── helpers ───────────────────────────────────────────────────────────────────
def build_grasp_6d(env, angle_deg: float):
    """
    Build a 6-DoF grasp pose:
      - Translation: GT object centroid in world coords
      - Approach:    straight down (-Z)
      - Closing dir: from ShapeGrasp predicted angle (degrees)
    """
    obj_u   = env.unwrapped
    obj_pos = obj_u._objs[0].pose.p[0].cpu().numpy()

    angle_rad   = float(angle_deg - 90) * np.pi / 180.0
    closing_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
    approach    = np.array([0.0, 0.0, -1.0])

    return obj_u.agent.build_grasp_pose(approach, closing_dir, obj_pos)


def is_grasped_lifted(env_u):
    grasped = bool(env_u.agent.is_grasping(env_u.obj).item())
    height  = float(env_u.obj.pose.p[0, 2].item())
    return grasped, height


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="Path to ShapeGrasp eval run dir (contains results.json)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip objects that already have a video")
    ap.add_argument("--objects", nargs="+", default=None,
                    help="Only render these object IDs")
    args = ap.parse_args()

    results_path = os.path.join(args.run, "results.json")
    with open(results_path) as f:
        sg_results = json.load(f)

    ok_results = [r for r in sg_results if r.get("status") == "ok"]
    if args.objects:
        ok_results = [r for r in ok_results if r["obj_id"] in args.objects]

    vid_dir = os.path.join(args.run, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    vid_results_path = os.path.join(args.run, "video_results.json")
    vid_results = []
    if args.resume and os.path.exists(vid_results_path):
        with open(vid_results_path) as f:
            vid_results = json.load(f)
        done_ids = {r["obj_id"] for r in vid_results if r.get("video")}
        ok_results = [r for r in ok_results if r["obj_id"] not in done_ids]
        print(f"Resuming — {len(done_ids)} already done, {len(ok_results)} remaining")

    print(f"\n{'='*60}")
    print(f"  ShapeGrasp YCB — Video Rendering")
    print(f"  Objects : {len(ok_results)}")
    print(f"  Videos  : {vid_dir}/")
    print(f"{'='*60}\n")

    def save_vid_results():
        with open(vid_results_path, "w") as f:
            json.dump(vid_results, f, indent=2, default=str)

    for idx, sg_r in enumerate(ok_results):
        obj_id    = sg_r["obj_id"]
        angle_deg = sg_r["angle_deg"]
        task      = sg_r["task"]
        obj_name  = sg_r["obj_name"]

        obj_vid_dir = os.path.join(vid_dir, obj_id)
        os.makedirs(obj_vid_dir, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"  [{idx+1}/{len(ok_results)}]  {obj_id}")
        print(f"  Task  : \"{task}\"  |  Angle: {angle_deg}°")

        _STATE["obj_id"] = obj_id
        t0 = time.time()

        try:
            base_env = gym.make(
                _ENV_ID,
                obs_mode="rgbd",
                control_mode="pd_joint_pos",
                render_mode="rgb_array",
                reward_mode="dense",
                sensor_configs=dict(shader_pack="default", width=384, height=384),
                human_render_camera_configs=dict(shader_pack="default"),
                viewer_camera_configs=dict(shader_pack="default"),
                sim_backend="cpu",
                enable_shadow=True,
            )
            env = RecordEpisode(
                base_env,
                output_dir=obj_vid_dir,
                save_trajectory=False,
                trajectory_name=obj_id,
                save_video=True,
                source_type="motionplanning",
                source_desc=f"ShapeGrasp — {task}  angle={angle_deg}°",
                video_fps=20,
                save_on_reset=False,
            )

            obs, _ = env.reset(seed=SEED)

            # Build grasp pose from ShapeGrasp angle
            grasp_6d = build_grasp_6d(env, angle_deg)

            planner = PandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )

            # Approach pre-grasp → grasp → lift
            planner.move_to_pose_with_screw(grasp_6d * sapien.Pose([0, 0, -0.05]))
            planner.move_to_pose_with_screw(grasp_6d * sapien.Pose([0.005, 0, 0.015]))
            planner.close_gripper()

            goal_pose = grasp_6d * sapien.Pose([0, 0, -0.4])
            try:
                env.unwrapped.goal_pos = torch.from_numpy(goal_pose.p)
                env.unwrapped.goal_site.set_pose(
                    Pose.create_from_pq(env.unwrapped.goal_pos)
                )
            except AttributeError:
                pass
            planner.move_to_pose_with_screw(goal_pose)

            grasped, height = is_grasped_lifted(env.unwrapped)
            planner.close()
            env.flush_video()
            env.close()

            # find the saved mp4
            mp4s = [f for f in os.listdir(obj_vid_dir) if f.endswith(".mp4")]
            video_path = os.path.join(obj_vid_dir, mp4s[0]) if mp4s else None

            elapsed = round(time.time() - t0, 1)
            status = "grasped" if grasped else "failed"
            print(f"  {status.upper()}  height={height:.3f}m  {elapsed}s"
                  f"  → {video_path}")

            vid_results.append(dict(
                obj_id=obj_id, obj_name=obj_name, task=task,
                angle_deg=angle_deg, grasped=grasped,
                height_m=round(height, 4),
                video=video_path, elapsed_s=elapsed,
                status=status,
            ))

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}")
            try:
                env.flush_video(save=False); env.close()
            except Exception:
                pass
            vid_results.append(dict(
                obj_id=obj_id, task=task, angle_deg=angle_deg,
                grasped=False, status="error", error=str(e),
                elapsed_s=round(time.time() - t0, 1),
            ))

        save_vid_results()

    # ── summary ───────────────────────────────────────────────────────────────
    grasped_n = sum(1 for r in vid_results if r.get("grasped"))
    total_n   = len(vid_results)
    print(f"\n{'='*60}")
    print(f"  Video Rendering Complete")
    print(f"  Grasped : {grasped_n}/{total_n}  ({100*grasped_n/max(total_n,1):.0f}%)")
    print(f"  Videos  : {vid_dir}/")
    print(f"{'='*60}")
    print(f"\n{'Object':<35} {'Status':<10} {'Height':>8}  Video")
    print("─" * 80)
    for r in vid_results:
        st = "✓" if r.get("grasped") else "✗"
        h  = f"{r.get('height_m', 0):.3f}m"
        vp = os.path.basename(r.get("video") or "")
        print(f"  {r['obj_id']:<33} {st}  {h:>8}  {vp}")


if __name__ == "__main__":
    main()
