"""WRM-augmented LingBot eval policy server.

Wraps the standard LingBot-VLA policy: at each chunk boundary it samples the
clean action chunk a_VLA (normalized, padded 75-d), then applies a residual
correction Delta from the trained WRM (World Rectification Model) in LingBot's
normalized action space, and executes a_final = a_VLA + Delta.

Injection point mirrors the offline residual data pipeline
(tools/residual_wm/build_residual_dataset.py): right after feature_transform.apply
and sample_actions, before unapply.

Runs in the `lingbotvla` conda env; WRM (+ WAN backbone + umT5 text encoder) is
imported from the Motus repo (added to sys.path).
"""

import os
import sys
import time
import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MOTUS_ROOT = os.environ.get("MOTUS_ROOT", "/mnt/data14/yyg/Motus")
if MOTUS_ROOT not in sys.path:
    sys.path.insert(0, MOTUS_ROOT)

from deploy.lingbot_vla_policy import LingbotVLAServer
from deploy.websocket_policy_server import WebsocketPolicyServer

from models.wrm import WRM, WRMConfig  # from Motus repo
from models.wrm_und import WRMUnd, WRMUndConfig  # trimodal variant (Und expert)
from bak.wan.modules.t5 import T5EncoderModel
from tools.residual_wm.build_residual_dataset import extract_und_feat

def _configure_logger() -> logging.Logger:
    """Dedicated stderr handler; root logger is often pre-configured by transformers."""
    log = logging.getLogger("lingbot_wrm_policy")
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    log.propagate = False
    return log


logger = _configure_logger()

CAM_HIGH_KEY = "observation.images.cam_high"


class WRMCorrector:
    """Loads WRM + umT5 and produces a residual Delta in LingBot-normalized space."""

    def __init__(
        self,
        wrm_ckpt: str,
        wan_dir: str,
        delta_stats: str,
        video_mode: str = "denoise",
        num_inference_steps: int = 50,
        video_height: int = 384,
        video_width: int = 320,
        text_len: int = 512,
        device: str = "cuda",
        use_und: bool = False,
        und_hidden_size: int = 512,
        vlm_adapter_input_dim: int = 2048,
        wan_mode: str = "lora",
    ):
        self.device = device
        self.video_mode = video_mode
        self.num_inference_steps = num_inference_steps
        self.H, self.W = video_height, video_width
        self.text_len = text_len
        self.use_und = use_und

        common = dict(
            wan_checkpoint_path=wan_dir,
            wan_config_path=wan_dir,
            vae_path=str(Path(wan_dir) / "Wan2.2_VAE.pth"),
            num_layers=30,
            action_dim=14,
            action_state_dim=14,
            action_chunk_size=50,
            num_video_frames=8,
            video_height=video_height,
            video_width=video_width,
            batch_size=1,
            wan_finetune_mode=wan_mode,
            delta_stats_path=delta_stats,
        )
        if use_und:
            logger.info(f"building WRMUnd (WAN {wan_mode} + Understanding Expert) ...")
            cfg = WRMUndConfig(
                und_expert_hidden_size=und_hidden_size,
                vlm_adapter_input_dim=vlm_adapter_input_dim,
                **common,
            )
            self.model = WRMUnd(cfg).to(device).eval()
        else:
            logger.info(f"building WRM (WAN {wan_mode}) ...")
            cfg = WRMConfig(**common)
            self.model = WRM(cfg).to(device).eval()

        logger.info(f"loading WRM ckpt {wrm_ckpt}")
        ckpt = torch.load(wrm_ckpt, map_location="cpu")
        trainable = ckpt.get("trainable_state_dict", ckpt)
        missing, unexpected = self.model.load_state_dict(trainable, strict=False)
        logger.info(
            f"WRM ckpt loaded (overlay): {len(trainable)} tensors, "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

        # umT5 text encoder (same as the precomputed umt5_wan embeddings)
        t5_ckpt = str(Path(wan_dir) / "models_t5_umt5-xxl-enc-bf16.pth")
        t5_tok = str(Path(wan_dir) / "google" / "umt5-xxl")
        logger.info(f"loading umT5 encoder {t5_ckpt}")
        self.t5 = T5EncoderModel(
            text_len=text_len, dtype=torch.bfloat16, device=device,
            checkpoint_path=t5_ckpt, tokenizer_path=t5_tok,
        )
        self._t5_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        if text not in self._t5_cache:
            emb = self.t5([text], self.device)[0].float()  # [seq, 4096]
            out = torch.zeros(self.text_len, emb.shape[1], dtype=torch.float32)
            n = min(emb.shape[0], self.text_len)
            out[:n] = emb[:n].cpu()
            self._t5_cache[text] = out
        return self._t5_cache[text]

    def _prep_frame(self, rgb_hwc_uint8: np.ndarray) -> torch.Tensor:
        rgb = cv2.resize(rgb_hwc_uint8, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0  # [C,H,W]

    @torch.no_grad()
    def correct(self, first_frame_rgb, state_real, a_vla_real, text, und_feats=None):
        """All inputs/outputs in LingBot-normalized space.

        Args:
            first_frame_rgb: HWC uint8 raw cam_high image
            state_real:  [14] tensor (normalized state, real dims)
            a_vla_real:  [chunk, 14] tensor (normalized action, real dims)
            text: instruction string
            und_feats: [L_u, vlm_dim] LingBot-VLA Qwen2.5-VL prefix hidden (WRMUnd only)
        Returns:
            delta_real: [chunk, 14] tensor (LingBot-normalized residual)
        """
        first_frame = self._prep_frame(first_frame_rgb).unsqueeze(0)       # [1,C,H,W]
        state = state_real.float().unsqueeze(0)                            # [1,14]
        a_vla = a_vla_real.float().unsqueeze(0)                            # [1,chunk,14]
        t5 = self._encode_text(text).unsqueeze(0)                          # [1,512,4096]

        if self.use_und:
            assert und_feats is not None, "WRMUnd requires und_feats"
            delta = self.model.inference_step(
                first_frame=first_frame,
                state=state,
                a_vla=a_vla,
                language_embeddings=t5,
                und_feats=und_feats.unsqueeze(0).to(self.device),  # [1, L_u, vlm_dim]
                num_inference_steps=self.num_inference_steps,
                video_mode=self.video_mode,
            )
        else:
            delta = self.model.inference_step(
                first_frame=first_frame,
                state=state,
                a_vla=a_vla,
                language_embeddings=t5,
                num_inference_steps=self.num_inference_steps,
                video_mode=self.video_mode,
            )  # [1, chunk, 14], unnormalized residual in lingbot-norm space
        return delta.squeeze(0).cpu()

    @torch.no_grad()
    def correct_sde(self, first_frame_rgb, state_real, a_vla_real, text, und_feats,
                    eta: float, steps: int, generator=None):
        """Flow-GRPO stochastic correction. Returns (delta_real, trace, lang).

        ``trace`` carries everything the offline learner needs to recompute the
        residual log-prob (see ``WRMUnd.action_logprob_from_trace``) EXCEPT the
        language embedding (returned separately as ``lang`` so it can be saved
        once per episode instead of per chunk). All tensors are on CPU.
        """
        assert self.use_und and und_feats is not None, "RL path requires WRMUnd + und_feats"
        first_frame = self._prep_frame(first_frame_rgb).unsqueeze(0)
        state = state_real.float().unsqueeze(0)
        a_vla = a_vla_real.float().unsqueeze(0)
        t5 = self._encode_text(text).unsqueeze(0)
        out = self.model.sample_actions_sde(
            first_frame=first_frame, state=state, a_vla=a_vla,
            language_embeddings=t5, und_feats=und_feats.unsqueeze(0).to(self.device),
            num_inference_steps=steps, video_mode=self.video_mode, eta=eta, generator=generator,
        )
        trace = {
            "state": state_real.float().cpu(),                       # [14]
            "a_vla_real": a_vla_real.float().cpu(),                  # [chunk,14]
            "und_feats": und_feats.float().cpu(),                    # [L_u, vlm_dim]
            "first_frame": first_frame.squeeze(0).to("cpu", torch.bfloat16),  # [C,H,W] in [0,1], WM cond
            "num_denoise_steps": int(steps),
            "action_latents": out["action_latents"].to("cpu", torch.bfloat16),
            "video_latents": out["video_latents"].to("cpu", torch.bfloat16),
            "timesteps": out["timesteps"].float().cpu(),
            "eta": float(eta),
            "old_logprob": out["old_logprob"].float().cpu(),         # [1]
            "video_mode": self.video_mode,
        }
        delta_real = out["delta"].squeeze(0).cpu()                    # [chunk,14]
        lang = t5.squeeze(0).to("cpu", torch.bfloat16)               # [512,4096] saved per-episode
        return delta_real, trace, lang


class LingbotWRMServer(LingbotVLAServer):
    def __init__(self, *args, wrm_corrector: WRMCorrector = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.wrm = wrm_corrector

    @torch.no_grad()
    def infer(self, observation):
        if "reset" in observation and observation["reset"]:
            self.reset(robo_name=observation["robo_name"])
            return dict(action=None)

        # capture raw cam_high + instruction BEFORE resize mutates the obs
        raw_cam_high = np.asarray(observation[CAM_HIGH_KEY]).copy()
        instruction = observation.get("task", "")
        task_name = observation.get("task_name", "")

        self.resize_image(observation)
        for k, v in observation.items():
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
        ).squeeze(0).float().cpu()  # [chunk, 75] normalized + padded

        # ---- WRM residual correction in normalized space ----
        if self.wrm is not None:
            mask = batch["joint_mask"].bool().cpu()
            a_vla_real = a_vla[:, mask]                       # [chunk, 14]
            state_real = batch["state"].float().cpu()[mask]  # [14]
            und_feats = None
            if self.wrm.use_und:
                # Reuse LingBot-VLA's own Qwen2.5-VL prefix (image+instruction) hidden
                # states as the understanding tokens (live extraction, no second VLM).
                und_feats, _ = extract_und_feat(self, batch, dtype)  # [L_u, vlm_dim]
            delta_real = self.wrm.correct(
                raw_cam_high, state_real, a_vla_real, instruction, und_feats=und_feats
            )
            delta_l2 = torch.linalg.vector_norm(delta_real).item()
            avla_l2 = torch.linalg.vector_norm(a_vla_real).item()
            eps = 1e-8
            metric = {
                "step": int(self.global_step),
                "task": str(task_name),
                "delta_l2": delta_l2,
                "avla_l2": avla_l2,
                "rel_delta_to_vla": delta_l2 / (avla_l2 + eps),
                "delta_abs_mean": float(delta_real.abs().mean().item()),
                "delta_abs_max": float(delta_real.abs().max().item()),
                "cos_delta_vla": float(
                    torch.sum(delta_real * a_vla_real).item() / (delta_l2 * avla_l2 + eps)
                ),
            }
            logger.info("WRM_METRIC %s", json.dumps(metric, ensure_ascii=False, sort_keys=True))
            a_final = a_vla.clone()
            a_final[:, mask] = a_vla_real + delta_real
        else:
            a_final = a_vla

        # unapply (denormalize + de-pad) -> real 14-d action
        batch["actions"] = a_final.to(dtype=torch.float32)
        if self.use_bf16:
            batch["state"] = batch["state"].to(dtype=torch.float32)
        output = self.vla.feature_transform.unapply(batch)

        action_chunk = {}
        for k in output.keys():
            if k in ft.org_features["actions"]:
                n = self.use_length if self.use_length > 0 else output[k].shape[0]
                action_chunk[k] = output[k][:n, :].float().cpu().numpy()
        self.global_step += 1
        return action_chunk


def main():
    p = argparse.ArgumentParser(description="WRM-augmented LingBot policy server")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--use_length", type=int, default=50)
    p.add_argument("--norm_path", type=str, default=None)
    p.add_argument("--num_denoising_step", type=int, default=10)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8006)
    # WRM
    p.add_argument("--wrm_ckpt", type=str, required=True)
    p.add_argument("--wan_dir", type=str, default="/mnt/data14/liuxiao/pretrained_models/Wan2.2-TI2V-5B")
    p.add_argument("--delta_stats", type=str, required=True)
    p.add_argument("--video_mode", type=str, default="denoise", choices=["skip", "denoise"])
    p.add_argument("--wrm_steps", type=int, default=50)
    p.add_argument("--disable_wrm", action="store_true", help="baseline: a_VLA only (no correction)")
    # WRMUnd (trimodal) variant
    p.add_argument("--wrm_variant", type=str, default="plain", choices=["plain", "und"],
                   help="plain=WRM (current), und=WRMUnd with Understanding Expert")
    p.add_argument("--und_hidden_size", type=int, default=512)
    p.add_argument("--vlm_adapter_input_dim", type=int, default=2048)
    p.add_argument("--wan_mode", type=str, default="lora", choices=["lora", "frozen", "full"],
                   help="must match training checkpoint (wrm_und_v1 uses full)")
    args = p.parse_args()

    os.environ.setdefault("QWEN25_PATH", str(ROOT / "weights" / "Qwen2.5-VL-3B-Instruct"))

    corrector = None
    if not args.disable_wrm:
        corrector = WRMCorrector(
            wrm_ckpt=args.wrm_ckpt, wan_dir=args.wan_dir, delta_stats=args.delta_stats,
            video_mode=args.video_mode, num_inference_steps=args.wrm_steps,
            use_und=(args.wrm_variant == "und"),
            und_hidden_size=args.und_hidden_size,
            vlm_adapter_input_dim=args.vlm_adapter_input_dim,
            wan_mode=args.wan_mode,
        )

    server = LingbotWRMServer(
        path_to_pi_model=args.model_path,
        robot_norm_path=args.norm_path,
        use_length=args.use_length,
        use_bf16=True,
        num_denoising_step=args.num_denoising_step,
        wrm_corrector=corrector,
    )
    _wrm_tag = "OFF" if args.disable_wrm else f"{args.wrm_variant}/{args.video_mode}"
    logger.info(f"serving WRM policy on {args.host}:{args.port} (wrm={_wrm_tag})")
    WebsocketPolicyServer(policy=server, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
