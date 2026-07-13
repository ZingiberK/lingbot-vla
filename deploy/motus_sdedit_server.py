"""Motus SDEdit refine server (runs in the `motus` conda env).

Wraps the pretrained Motus (Motus_robotwin2 + Qwen3-VL-2B + WAN2.2-5B) as a websocket
policy that REFINES an externally-provided action prior via SDEdit:

  request  = { cam_high, cam_left, cam_right (HWC uint8/float rgb),
               state (14,), task (instruction),
               a_init (n<=16, 14) raw-qpos prior (freq-aligned a_vla), t0 (float) }
  response = { action: (16, 14) raw-qpos refined actions }

The action prior is denoised from partial noise (start_t=t0) toward Motus's action
manifold; t0 small -> stay near a_init (VLA), t0=1 -> pure Motus. See
Motus/WRM_RL_PLAN_MOTUS_REFINE.md. The closed-loop env (RoboTwin, different env) is
driven by RoboTwin/script/rl_rollout_worker.py --refine_motus.

Usage (motus env):
  CUDA_VISIBLE_DEVICES=1 python -m deploy.motus_sdedit_server \
    --motus_ckpt /mnt/data14/liuxiao/pretrained_models/Motus_robotwin2 \
    --wan /mnt/data14/liuxiao/pretrained_models/Wan2.2-TI2V-5B \
    --vlm /mnt/data14/liuxiao/pretrained_models/Qwen3-VL-2B-Instruct \
    --port 8400 --default_t0 0.3 --steps 10
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # lingbot-vla
sys.path.insert(0, str(ROOT))
MOTUS_INFER = os.environ.get(
    "MOTUS_INFER_ROOT", "/mnt/data14/yyg/Motus/inference/robotwin/Motus")
sys.path.insert(0, MOTUS_INFER)

from deploy.websocket_policy_server import WebsocketPolicyServer  # noqa: E402

logger = logging.getLogger("motus_sdedit_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class MotusSDEditPolicy:
    def __init__(self, motus_ckpt, wan, vlm, default_t0=0.3, steps=10, device="cuda",
                 ft_action_ckpt=None):
        import torch
        from deploy_policy import MotusPolicy  # from MOTUS_INFER
        cfg = str(Path(MOTUS_INFER) / "utils" / "robotwin.yml")
        self.policy = MotusPolicy(checkpoint_path=motus_ckpt, config_path=cfg,
                                  wan_path=wan, vlm_path=vlm, device=device,
                                  task_name="motus_sdedit")
        self.policy.save_images = False
        # Tier 1: overlay a BC-finetuned action expert (manifold-gap probe, §10).
        if ft_action_ckpt:
            sd = torch.load(ft_action_ckpt, map_location=device)
            missing, unexpected = self.policy.model.action_expert.load_state_dict(sd, strict=False)
            self.policy.model.eval()
            logger.info("loaded finetuned action_expert from %s (missing=%d unexpected=%d)",
                        ft_action_ckpt, len(missing), len(unexpected))
        self.default_t0 = float(default_t0)
        self.steps = int(steps)
        self.chunk = int(self.policy.model.config.action_chunk_size)
        logger.info("MotusSDEdit ready: chunk=%d default_t0=%.2f steps=%d ft=%s",
                    self.chunk, self.default_t0, self.steps, bool(ft_action_ckpt))

    def infer(self, observation):
        if observation.get("reset"):
            self.policy.obs_cache.clear()
            self.policy.action_cache.clear()
            self.policy.current_state = None
            return dict(action=None)

        cam_high = np.asarray(observation["cam_high"])
        cam_left = np.asarray(observation["cam_left"])
        cam_right = np.asarray(observation["cam_right"])
        state = np.asarray(observation["state"], dtype=np.float32)
        instruction = observation.get("task", "")
        t0 = float(observation.get("t0", self.default_t0))
        a_init = np.asarray(observation["a_init"], dtype=np.float32)  # [n<=16, 14]

        obs_for_motus = {
            "observation": {
                "head_camera": {"rgb": cam_high},
                "left_camera": {"rgb": cam_left},
                "right_camera": {"rgb": cam_right},
            },
            "joint_action": {"vector": state},
        }
        self.policy.action_cache.clear()   # bound memory: we return actions directly
        self.policy.set_instruction(instruction)
        self.policy.update_obs(obs_for_motus)
        refined = self.policy.get_action_sdedit(a_init, start_t=t0, num_inference_steps=self.steps)
        return dict(action=np.asarray(refined, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motus_ckpt", default="/mnt/data14/liuxiao/pretrained_models/Motus_robotwin2")
    ap.add_argument("--wan", default="/mnt/data14/liuxiao/pretrained_models/Wan2.2-TI2V-5B")
    ap.add_argument("--vlm", default="/mnt/data14/liuxiao/pretrained_models/Qwen3-VL-2B-Instruct")
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--default_t0", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--ft_action_ckpt", default="",
                    help="optional finetuned action_expert .pt to overlay (Tier 1 §10)")
    args = ap.parse_args()

    policy = MotusSDEditPolicy(args.motus_ckpt, args.wan, args.vlm,
                               default_t0=args.default_t0, steps=args.steps,
                               ft_action_ckpt=(args.ft_action_ckpt or None))
    logger.info("serving MotusSDEdit on :%d", args.port)
    WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port).serve_forever()


if __name__ == "__main__":
    main()
