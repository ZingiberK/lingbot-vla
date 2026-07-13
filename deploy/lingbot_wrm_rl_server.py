"""Flow-GRPO rollout server (extends the WRM policy server).

Same VLA + WRMUnd stack as ``lingbot_wrm_policy.py``, but the residual is drawn
*stochastically* via ``WRMUnd.sample_actions_sde`` and the per-chunk denoise
trace is written to the shared filesystem so an offline GRPO learner can later
recompute log-probs and update the policy. The closed-loop env (RoboTwin, a
different conda env) is driven by ``RoboTwin/script/rl_rollout_worker.py`` which
talks to this server over the existing websocket protocol.

Trace layout (under ``--rl_root/<task>/round<k>/``):
    traces/<task>_seed<seed>_ep<ep>_chunk<chunk>.pt   per-chunk denoise trace
    lang/<task>_seed<seed>_ep<ep>.pt                  per-episode T5 embedding
    server.log                                        this server's log

The worker writes ``episodes/*.json`` (success + chunk list) on its side.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy.lingbot_wrm_policy import (  # noqa: E402
    WRMCorrector, LingbotWRMServer, extract_und_feat, CAM_HIGH_KEY,
)
from deploy.websocket_policy_server import WebsocketPolicyServer  # noqa: E402

logger = logging.getLogger("lingbot_wrm_rl_server")


def _configure_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.handlers = [fh, sh]
    logger.setLevel(logging.INFO)
    logger.propagate = False


class LingbotWRMRLServer(LingbotWRMServer):
    """Stochastic (SDE) rollout server that dumps denoise traces to disk."""

    def __init__(self, *args, rl_root: str, eta: float, rl_steps: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.rl_root = Path(rl_root)
        self.eta = float(eta)
        self.rl_steps = int(rl_steps)
        self._chunks_saved = 0

    def _round_dir(self, task: str, rnd: int) -> Path:
        return self.rl_root / task / f"round{rnd}"

    @torch.no_grad()
    def infer(self, observation):
        if "reset" in observation and observation["reset"]:
            self.reset(robo_name=observation["robo_name"])
            return dict(action=None)

        # ---- RL metadata injected by the worker ----
        rnd = int(observation.pop("rl_round", 0))
        seed = int(observation.pop("rl_seed", 0))
        ep = int(observation.pop("rl_ep", 0))
        chunk_idx = int(observation.pop("rl_chunk", 0))
        force_zero = bool(observation.pop("force_delta_zero", False))
        task_name = observation.get("task_name", "")
        instruction = observation.get("task", "")

        raw_cam_high = np.asarray(observation[CAM_HIGH_KEY]).copy()

        # ---- standard VLA preprocessing (mirrors LingbotWRMServer.infer) ----
        self.resize_image(observation)
        for k, v in list(observation.items()):
            if isinstance(v, np.ndarray):
                observation[k] = torch.from_numpy(v)

        ft = self.vla.feature_transform
        for action_feature in ft.org_features["actions"]:
            if action_feature not in observation:
                observation[action_feature] = torch.zeros(
                    ft.chunk_size, observation[ft.org_features["states"][0]].shape[0]
                )
        observation[ft.org_features["actions"][0] + "_is_pad"] = torch.zeros(
            observation[ft.org_features["actions"][0]].shape[0]
        )

        batch = ft.apply(observation)
        dtype = torch.bfloat16 if self.use_bf16 else torch.float32

        images = batch["images"]
        img_masks = batch["img_masks"]
        if images.ndim == 4:
            images = images.unsqueeze(0)
            img_masks = img_masks.unsqueeze(0)

        a_vla = self.vla.model.sample_actions(
            images.to(dtype=dtype, device="cuda"),
            img_masks.to(device="cuda"),
            batch["lang_tokens"].unsqueeze(0).to(device="cuda"),
            batch["lang_masks"].unsqueeze(0).to(device="cuda"),
            batch["state"].unsqueeze(0).to(dtype=dtype, device="cuda"),
            num_steps=self.num_denoising_step,
        ).squeeze(0).float().cpu()  # [chunk, 75]

        mask = batch["joint_mask"].bool().cpu()
        a_vla_real = a_vla[:, mask]                       # [chunk, 14]
        state_real = batch["state"].float().cpu()[mask]  # [14]
        und_feats, _ = extract_und_feat(self, batch, dtype)

        # ---- residual (+ optional SDE trace) ----
        rdir = self._round_dir(task_name, rnd)
        (rdir / "traces").mkdir(parents=True, exist_ok=True)
        (rdir / "wm_traces").mkdir(parents=True, exist_ok=True)
        (rdir / "lang").mkdir(parents=True, exist_ok=True)
        lang_path = rdir / "lang" / f"{task_name}_seed{seed}_ep{ep}.pt"
        t0 = time.time()
        trace_path = ""
        if force_zero:
            first_frame = self.wrm._prep_frame(raw_cam_high).unsqueeze(0)
            lang = self.wrm._encode_text(instruction).squeeze(0).to("cpu", torch.bfloat16)
            if chunk_idx == 0 or not lang_path.exists():
                torch.save(lang, lang_path)
            delta_real = torch.zeros_like(a_vla_real)
            delta_rms = 0.0
            wm_trace_path = rdir / "wm_traces" / f"{task_name}_seed{seed}_ep{ep}_chunk{chunk_idx}.pt"
            slim = {
                "state": state_real.float().cpu(),
                "a_vla_real": a_vla_real.float().cpu(),
                "und_feats": und_feats.float().cpu(),
                "first_frame": first_frame.squeeze(0).to("cpu", torch.bfloat16),
                "vla_zero": True,
                "meta": {
                    "task": task_name, "round": rnd, "seed": seed, "ep": ep, "chunk": chunk_idx,
                    "lang_path": str(lang_path), "delta_rms": 0.0, "instruction": instruction,
                },
            }
            torch.save(slim, wm_trace_path)
            trace_path = str(wm_trace_path)
            self._chunks_saved += 1
            log_extra = {"mode": "vla_zero", "old_logprob": 0.0}
        else:
            delta_real, trace, lang = self.wrm.correct_sde(
                raw_cam_high, state_real, a_vla_real, instruction, und_feats,
                eta=self.eta, steps=self.rl_steps,
            )
            delta_rms = float(torch.sqrt((delta_real ** 2).mean()).item())
            if chunk_idx == 0 or not lang_path.exists():
                torch.save(lang, lang_path)
            tp = rdir / "traces" / f"{task_name}_seed{seed}_ep{ep}_chunk{chunk_idx}.pt"
            trace["meta"] = {
                "task": task_name, "round": rnd, "seed": seed, "ep": ep, "chunk": chunk_idx,
                "lang_path": str(lang_path), "delta_rms": delta_rms, "instruction": instruction,
            }
            torch.save(trace, tp)
            trace_path = str(tp)
            self._chunks_saved += 1
            log_extra = {"mode": "wrm", "old_logprob": round(float(trace["old_logprob"][0]), 2)}
        infer_ms = (time.time() - t0) * 1000.0

        logger.info(
            "RL_CHUNK %s",
            json.dumps({
                "round": rnd, "task": task_name, "seed": seed, "ep": ep, "chunk": chunk_idx,
                "delta_rms": round(delta_rms, 5), "infer_ms": round(infer_ms, 1),
                "total_saved": self._chunks_saved, **log_extra,
            }, ensure_ascii=False),
        )

        # ---- compose executed action + unapply ----
        a_final = a_vla.clone()
        a_final[:, mask] = a_vla_real + delta_real
        batch["actions"] = a_final.to(dtype=torch.float32)
        if self.use_bf16:
            batch["state"] = batch["state"].to(dtype=torch.float32)
        output = self.vla.feature_transform.unapply(batch)

        action_chunk = {}
        for k in output.keys():
            if k in ft.org_features["actions"]:
                n = self.use_length if self.use_length > 0 else output[k].shape[0]
                action_chunk[k] = output[k][:n, :].float().cpu().numpy()
        action_chunk["_rl_delta_rms"] = delta_rms
        action_chunk["_rl_trace"] = str(trace_path)
        self.global_step += 1
        return action_chunk


def main():
    p = argparse.ArgumentParser(description="WRM Flow-GRPO rollout server")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--use_length", type=int, default=50)
    p.add_argument("--norm_path", type=str, default=None)
    p.add_argument("--num_denoising_step", type=int, default=10)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8200)
    # WRM / WRMUnd
    p.add_argument("--wrm_ckpt", type=str, required=True)
    p.add_argument("--wan_dir", type=str, default="/mnt/data14/liuxiao/pretrained_models/Wan2.2-TI2V-5B")
    p.add_argument("--delta_stats", type=str, required=True)
    p.add_argument("--video_mode", type=str, default="denoise", choices=["skip", "denoise"])
    p.add_argument("--und_hidden_size", type=int, default=512)
    p.add_argument("--vlm_adapter_input_dim", type=int, default=2048)
    p.add_argument("--wan_mode", type=str, default="full", choices=["lora", "frozen", "full"])
    # RL
    p.add_argument("--rl_root", type=str, required=True, help="shared dir for traces/logs")
    p.add_argument("--rl_round", type=int, default=0, help="round id (used only for log/dir banner)")
    p.add_argument("--eta", type=float, default=0.5, help="SDE exploration level (RLinf noise_level=0.5)")
    p.add_argument("--rl_steps", type=int, default=10, help="denoise steps for RL sampling")
    args = p.parse_args()

    _configure_logger(Path(args.rl_root) / "server.log")
    os.environ.setdefault("QWEN25_PATH", str(ROOT / "weights" / "Qwen2.5-VL-3B-Instruct"))

    logger.info("RL_SERVER_START %s", json.dumps({
        "round": args.rl_round, "ckpt": args.wrm_ckpt, "eta": args.eta,
        "rl_steps": args.rl_steps, "video_mode": args.video_mode, "port": args.port,
    }))

    corrector = WRMCorrector(
        wrm_ckpt=args.wrm_ckpt, wan_dir=args.wan_dir, delta_stats=args.delta_stats,
        video_mode=args.video_mode, num_inference_steps=args.rl_steps,
        use_und=True, und_hidden_size=args.und_hidden_size,
        vlm_adapter_input_dim=args.vlm_adapter_input_dim, wan_mode=args.wan_mode,
    )

    server = LingbotWRMRLServer(
        path_to_pi_model=args.model_path,
        robot_norm_path=args.norm_path,
        use_length=args.use_length,
        use_bf16=True,
        num_denoising_step=args.num_denoising_step,
        wrm_corrector=corrector,
        rl_root=args.rl_root,
        eta=args.eta,
        rl_steps=args.rl_steps,
    )
    logger.info(f"RL rollout server ready on {args.host}:{args.port}")
    WebsocketPolicyServer(policy=server, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
