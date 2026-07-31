"""UniTS checkpoint adapter for the ST-PGN prior branch.

This adapter loads the official UniTS pretraining checkpoint, freezes the
backbone, and exposes intermediate token features as [B, C, L, D].  C-MAPSS
has 21 channels and a 30-step context, while the released checkpoint was
trained with other channel counts and a 96-step context, so the adapter uses
UniTS' shared embedding/backbone weights and creates a C-MAPSS prompt tensor.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _load_units_module(units_repo: str):
    path = Path(units_repo) / "models" / "UniTS.py"
    if not path.exists():
        raise FileNotFoundError(f"UniTS model file not found: {path}")
    spec = importlib.util.spec_from_file_location("stpgn_units_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import UniTS from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_namespace(args):
    if isinstance(args, SimpleNamespace):
        return args
    if hasattr(args, "__dict__"):
        return SimpleNamespace(**vars(args))
    return SimpleNamespace(**args)


class UniTSPriorAdapter(nn.Module):
    """Frozen UniTS feature extractor compatible with C-MAPSS inputs."""

    def __init__(self, units_repo: str, checkpoint: str, channels: int = 21):
        super().__init__()
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"UniTS checkpoint not found: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "student" not in ckpt or "args" not in ckpt:
            raise ValueError("checkpoint must contain student and args")

        units_module = _load_units_module(units_repo)
        args = _as_namespace(ckpt["args"])
        self.d_model = int(args.d_model)
        self.prompt_num = int(args.prompt_num)
        self.patch_len = int(args.patch_len)
        self.stride = int(args.stride)
        self.channels = channels

        # A single synthetic forecasting task is enough to construct the
        # shared UniTS backbone.  The original dataset-specific prompt is
        # replaced below by an expanded C-MAPSS prompt.
        task_config = {
            "dataset": "CMAPSS",
            "task_name": "long_term_forecast",
            "enc_in": channels,
            "seq_len": int(args.seq_len),
            "pred_len": int(args.pred_len),
        }
        configs = [["CMAPSS", task_config]]
        args.enc_in = channels
        model = units_module.Model(args, configs, pretrain=True)

        state = ckpt["student"]
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
        incompatible = model.load_state_dict(state, strict=False)
        self.missing_keys = list(incompatible.missing_keys)
        self.unexpected_keys = list(incompatible.unexpected_keys)
        self.model = model
        self._initialize_cmapss_prompt(state)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def _initialize_cmapss_prompt(self, state: Dict[str, Tensor]):
        prompt_keys = [
            key for key in state
            if key.startswith("prompt_tokens.") and state[key].ndim == 4
        ]
        if not prompt_keys:
            return
        # Different pretraining datasets have different variable counts
        # (e.g. 111 and 321), so reduce each prompt across its own channel
        # dimension before averaging datasets.
        prompt = torch.stack([
            state[key].float().mean(dim=1, keepdim=True)
            for key in prompt_keys
        ]).mean(dim=0)
        prompt = prompt.repeat(1, self.channels, 1, 1)
        target = self.model.prompt_tokens["CMAPSS"]
        target.copy_(prompt.to(dtype=target.dtype, device=target.device))

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Return UniTS hidden tokens with shape [B, C, L, D]."""
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"expected [B,T,{self.channels}], got {tuple(x.shape)}")
        b, _, c = x.shape
        # UniTS expects [B,C,T] before its patch embedding.  We use the same
        # normalization convention as the ST-PGN data loader, so do not
        # normalize a second time here.
        values = x.transpose(1, 2).contiguous()
        remainder = values.shape[-1] % self.patch_len
        if remainder:
            values = F.pad(values, (0, self.patch_len - remainder))
        tokens, n_vars = self.model.patch_embeddings(values)
        token_len = tokens.shape[-2]
        tokens = tokens.reshape(b, n_vars, token_len, self.d_model)
        tokens = tokens + self.model.position_embedding(tokens)

        prompt = self.model.prompt_tokens["CMAPSS"].repeat(b, 1, 1, 1)
        cls = self.model.cls_tokens["CMAPSS"]
        cls = cls.repeat(b, 1, 1, 1)
        full = torch.cat((prompt, tokens, cls), dim=2)
        full = self.model.backbone(
            full,
            prefix_len=self.prompt_num,
            seq_len=token_len,
        )
        return full[:, :, self.prompt_num:self.prompt_num + token_len]
