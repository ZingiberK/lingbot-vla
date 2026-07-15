"""Motus RL rollout server (Phase 2).

WebSocket server (run in the ``motus`` env) that wraps the Motus deploy policy
and serves Flow-SDE action sampling for RL rollouts. Each ``infer`` call:

  1. composites the 3 raw camera views + state (``MotusPolicy.update_obs``),
  2. samples an action chunk via ``MotusPolicy.get_action_sde`` (eta>0 exploration),
  3. saves the SDE trace (latents + log-prob + first_frame + state) to
     ``trace_path`` for the offline PPO update (``motus_rl/ppo_update.py``),
  4. returns ``{action, trace_path, old_logprob}``.

The RoboTwin driver (``RoboTwin/script/motus_rl_rollout_worker.py``) executes the
returned actions, captures composite future frames for the supervised WM loss,
and records per-episode success.

Env / args:
    MOTUS_INFER_ROOT  path to Motus/inference/robotwin/Motus (deploy_policy + models)
    --motus_ckpt --config --wan --vlm --port
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from deploy.websocket_policy_server import WebsocketPolicyServer

MOTUS_INFER_ROOT = os.environ.get(
    "MOTUS_INFER_ROOT", "/mnt/data14/yyg/Motus/inference/robotwin/Motus"
)
if MOTUS_INFER_ROOT not in sys.path:
    sys.path.insert(0, MOTUS_INFER_ROOT)

from deploy_policy import MotusPolicy  # noqa: E402  (from MOTUS_INFER_ROOT)


class MotusRLServer:
    def __init__(self, checkpoint_path, config_path, wan_path, vlm_path, device="cuda"):
        self.policy = MotusPolicy(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            wan_path=wan_path,
            vlm_path=vlm_path,
            device=device,
        )

    def _build_obs(self, obs):
        return {
            "observation": {
                "head_camera": {"rgb": np.asarray(obs["cam_high"])},
                "left_camera": {"rgb": np.asarray(obs["cam_left"])},
                "right_camera": {"rgb": np.asarray(obs["cam_right"])},
            },
            "joint_action": {"vector": np.asarray(obs["state"], dtype=np.float32)},
        }

    def infer(self, obs: dict) -> dict:
        if obs.get("reset"):
            self.policy.obs_cache.clear()
            self.policy.action_cache.clear()
            self.policy.current_state = None
            if obs.get("instruction"):
                self.policy.set_instruction(obs["instruction"])
            return {"ok": 1}

        self.policy.set_instruction(obs["instruction"])
        self.policy.update_obs(self._build_obs(obs))

        action_init = obs.get("action_init")
        actions, trace = self.policy.get_action_sde(
            num_inference_steps=int(obs.get("num_inference_steps", 10)),
            eta=float(obs.get("eta", 0.5)),
            action_init=np.asarray(action_init) if action_init is not None else None,
            start_t=float(obs.get("start_t", 1.0)),
            seed=int(obs.get("seed", -1)),
        )

        trace_path = obs["trace_path"]
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(trace, trace_path)

        return {
            "action": np.asarray(actions, dtype=np.float32),
            "trace_path": trace_path,
            "old_logprob": float(trace["old_logprob"]),
        }


def main():
    ap = argparse.ArgumentParser(description="Motus RL rollout server (Flow-SDE)")
    ap.add_argument("--motus_ckpt", required=True, help="deploy-format Motus ckpt dir (mp_rank_00_model_states.pt)")
    ap.add_argument("--config", default=os.path.join(MOTUS_INFER_ROOT, "configs", "robotwin.yaml"))
    ap.add_argument("--wan", required=True)
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = MotusRLServer(args.motus_ckpt, args.config, args.wan, args.vlm, device=args.device)
    server = WebsocketPolicyServer(model, port=args.port)
    print(f"Motus RL server listening on :{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
