#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flow Matching ACT (ACT-FM)

Flow convention follows pi0/pi05 (openpi):
  x_t = t * noise + (1 - t) * actions   (t=1 → noise, t=0 → action)
  u_t = noise - actions
  Inference: Euler/Heun from t=1 → 0, dt = -1/num_steps

Time conditioning uses AdaLN-Zero on the decoder layers (DiT-style):
  Each of the 3 LayerNorms in every decoder layer is replaced by:
    AdaLN(x) = (1 + scale_i) * LN(x) + shift_i
  where (scale_i, shift_i) come from a zero-initialised linear projection
  of the time embedding.  Zero-init means the model starts as standard LN
  and the time conditioning is gradually learned — same as DiT AdaLN-Zero.

Solver: second-order Heun (trapezoidal rule) for the ODE integration.
  Heun takes two velocity evaluations per step and averages them, giving
  O(dt²) accuracy vs O(dt) for Euler — enabling good quality at 5 steps.
"""

import math
from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.configuration_act_fm import ACTFMConfig
from lerobot.policies.act.modeling_act import (
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    get_activation_fn,
)
from lerobot.policies.failure_postprocessor import FailurePostprocessor  # noqa: F401
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# ---------------------------------------------------------------------------
# pi05-compatible helpers
# ---------------------------------------------------------------------------


def get_safe_dtype(target_dtype, device_type):
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float, max_period: float, device
) -> Tensor:
    """Exact copy of pi05 `create_sinusoidal_pos_embedding`."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must have shape (batch_size,)")
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).float()


def ot_matching(noise: Tensor, actions: Tensor) -> Tensor:
    """Minibatch Optimal Transport: re-order noise within the batch.

    Solves a linear assignment problem to find the permutation of noise
    samples that minimises total L2 transport cost to the action samples.
    This ensures flow lines within each minibatch do not cross, making the
    learned velocity field single-valued and easier to fit.

    Uses scipy's Hungarian algorithm (O(B³), negligible for B ≤ 256).

    Args:
        noise:   (B, S, A)
        actions: (B, S, A)
    Returns:
        (B, S, A) re-ordered noise (same samples, different assignment).
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError("use_ot_matching requires scipy. Install it with: pip install scipy") from e

    batch_size = noise.shape[0]
    # Flatten to (B, S*A) and compute pairwise L2 cost matrix on CPU.
    n_flat = noise.reshape(batch_size, -1).float().cpu()
    a_flat = actions.reshape(batch_size, -1).float().cpu()
    # cdist: (B, B) where cost[i, j] = ||noise[i] - actions[j]||
    cost = torch.cdist(n_flat, a_flat).numpy()
    # Hungarian algorithm: find permutation perm s.t. noise[perm[j]] → actions[j]
    _, col_ind = linear_sum_assignment(cost)
    return noise[col_ind]


def sample_beta(alpha: float, beta: float, bsize: int, device) -> Tensor:
    """Exact copy of pi05 `sample_beta` (CPU sampling for MPS compatibility)."""
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


# ---------------------------------------------------------------------------
# AdaLN-Zero decoder layer
# ---------------------------------------------------------------------------


class ACTDecoderLayerAdaLN(nn.Module):
    """ACT decoder layer with AdaLN-Zero time conditioning (DiT-style).

    Replaces each of the three LayerNorms with:
        AdaLN(x, t) = (1 + scale_i) * LN(x) + shift_i
    where scale_i and shift_i are linear projections of the time embedding.

    A single linear `adaLN_proj` outputs all 6 (scale, shift) vectors at once.
    Its weights and bias are zero-initialised so that at the start of training
    the layer is equivalent to a standard post-norm transformer block.

    Only post-norm (pre_norm=False) is supported because AdaLN-Zero is
    designed for post-norm architectures.
    """

    def __init__(self, config: ACTFMConfig):
        super().__init__()
        d = config.dim_model

        self.self_attn = nn.MultiheadAttention(d, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(d, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(d, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, d)

        # Standard LayerNorms — AdaLN modulates their output, not their params.
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)

        # AdaLN-Zero: one projection for all 6 (scale, shift) vectors.
        # 3 norms × 2 params × D = 6D outputs.
        # Zero-init → at init, scale=0 shift=0 → equivalent to standard LN.
        self.adaLN_proj = nn.Linear(d, 6 * d, bias=True)
        nn.init.zeros_(self.adaLN_proj.weight)
        nn.init.zeros_(self.adaLN_proj.bias)

    def _ada_ln(self, norm: nn.LayerNorm, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        """Apply adaptive layer norm: (1 + scale) * LN(x) + shift.

        Args:
            norm:  the LayerNorm module.
            x:     (S, B, D) — sequence-first layout.
            scale: (B, D) — from adaLN_proj, unsqueezed for broadcast.
            shift: (B, D) — from adaLN_proj, unsqueezed for broadcast.
        """
        # Unsqueeze to (1, B, D) for broadcasting over sequence dimension.
        s = scale.unsqueeze(0)
        b = shift.unsqueeze(0)
        return (1.0 + s) * norm(x) + b

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        time_emb: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:                 (S, B, D) decoder token sequence.
            encoder_out:       (E, B, D) encoder output.
            time_emb:          (B, D)    time embedding from `embed_time`.
            decoder_pos_embed: (S, 1, D) optional positional bias for queries.
            encoder_pos_embed: (E, 1, D) optional positional bias for keys.
        Returns:
            (S, B, D)
        """
        # Project time_emb → 6 vectors, each (B, D).
        params = self.adaLN_proj(time_emb)  # (B, 6D)
        sh1, sc1, sh2, sc2, sh3, sc3 = params.chunk(6, dim=-1)

        def add_pos(t: Tensor, pos: Tensor | None) -> Tensor:
            return t if pos is None else t + pos

        # --- Self-attention (post-norm) ---
        skip = x
        q = k = add_pos(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        x = self._ada_ln(self.norm1, x, sc1, sh1)
        skip = x

        # --- Cross-attention (post-norm) ---
        x = self.multihead_attn(
            query=add_pos(x, decoder_pos_embed),
            key=add_pos(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        x = self._ada_ln(self.norm2, x, sc2, sh2)
        skip = x

        # --- FFN (post-norm) ---
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        x = self._ada_ln(self.norm3, x, sc3, sh3)

        return x


class ACTDecoderAdaLN(nn.Module):
    """Stack of AdaLN-Zero decoder layers, followed by a final LayerNorm."""

    def __init__(self, config: ACTFMConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayerAdaLN(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        time_emb: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                time_emb,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
        return self.norm(x)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class ACTFlowMatching(nn.Module):
    """Flow Matching ACT with AdaLN-Zero decoder and Heun ODE solver."""

    def __init__(self, config: ACTFMConfig):
        super().__init__()
        self.config = config

        # Vision backbone.
        if config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            backbone_out_channels = backbone_model.fc.in_features
        else:
            backbone_out_channels = None

        # Encoder (observation features — no time conditioning needed here).
        self.encoder = ACTEncoder(config)

        n_1d_tokens = 0
        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
            n_1d_tokens += 1
        if config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(config.env_state_feature.shape[0], config.dim_model)
            n_1d_tokens += 1
        if config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_out_channels, config.dim_model, kernel_size=1
            )

        if n_1d_tokens > 0:
            self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        else:
            self.encoder_1d_feature_pos_embed = None

        if config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # AdaLN-Zero decoder (replaces standard ACTDecoder).
        self.decoder = ACTDecoderAdaLN(config)

        # Project noisy actions x_t → dim_model (pi05: action_in_proj).
        self.action_in_proj = nn.Linear(config.action_feature.shape[0], config.dim_model)

        # Learnable positional embedding for decoder queries.
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Time embedding: sinusoidal(min_period, max_period) → MLP (pi05 convention).
        # Output feeds AdaLN-Zero inside each decoder layer.
        self.time_mlp_in = nn.Linear(config.dim_model, config.dim_model)
        self.time_mlp_out = nn.Linear(config.dim_model, config.dim_model)

        # Velocity prediction head (pi05: action_out_proj).
        self.action_out_proj = nn.Linear(config.dim_model, config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for layer in self.decoder.layers:
            # adaLN_proj is already zero-inited; skip it, init the rest.
            for name, p in layer.named_parameters():
                if "adaLN_proj" not in name and p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------ #
    # Sampling helpers (pi05 convention)                                   #
    # ------------------------------------------------------------------ #

    def sample_noise(self, shape, device) -> Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize: int, device) -> Tensor:
        t = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        return t * self.config.time_sampling_scale + self.config.time_sampling_offset

    # ------------------------------------------------------------------ #
    # Time embedding (pi05 convention)                                     #
    # ------------------------------------------------------------------ #

    def embed_time(self, t: Tensor) -> Tensor:
        """sinusoidal(min_period, max_period) → time_mlp_in → SiLU → time_mlp_out → SiLU.

        Returns (B, dim_model).  Fed into AdaLN-Zero inside each decoder layer.
        """
        time_emb = create_sinusoidal_pos_embedding(
            t.float(),
            self.config.dim_model,
            self.config.min_period,
            self.config.max_period,
            device=t.device,
        )
        time_emb = F.silu(self.time_mlp_in(time_emb))
        time_emb = F.silu(self.time_mlp_out(time_emb))
        return time_emb  # (B, D)

    # ------------------------------------------------------------------ #
    # Encoder                                                              #
    # ------------------------------------------------------------------ #

    def _encode_observations(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        encoder_in_tokens = []
        encoder_in_pos_embed = []
        pos_1d_idx = 0

        if self.config.robot_state_feature:
            token = self.encoder_robot_state_input_proj(batch[OBS_STATE])
            encoder_in_tokens.append(token)
            encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].unsqueeze(0))
            pos_1d_idx += 1

        if self.config.env_state_feature:
            token = self.encoder_env_state_input_proj(batch[OBS_ENV_STATE])
            encoder_in_tokens.append(token)
            encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].unsqueeze(0))
            pos_1d_idx += 1  # noqa: F841

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        return encoder_out, encoder_in_pos_embed

    # ------------------------------------------------------------------ #
    # Decoder — velocity prediction                                        #
    # ------------------------------------------------------------------ #

    def _decode_velocity(
        self,
        x_t: Tensor,
        t: Tensor,
        encoder_out: Tensor,
        encoder_pos_embed: Tensor,
    ) -> Tensor:
        """Predict v_θ(x_t, t, obs) using AdaLN-conditioned decoder.

        Args:
            x_t:               (B, S, A) noisy actions.
            t:                 (B,)      time in [0, 1].
            encoder_out:       (E, B, D) encoder output.
            encoder_pos_embed: (E, 1, D) encoder positional embeddings.
        Returns:
            v_pred: (B, S, A)
        """
        # Time embedding → fed to AdaLN inside each decoder layer.
        time_emb = self.embed_time(t)  # (B, D)

        # Project noisy actions and add learnable positional embedding.
        queries = self.action_in_proj(x_t)  # (B, S, D)
        queries = einops.rearrange(queries, "b s d -> s b d")  # (S, B, D)
        decoder_pos = self.decoder_pos_embed.weight.unsqueeze(1)  # (S, 1, D)
        queries = queries + decoder_pos

        # AdaLN-conditioned cross-attention decoder.
        decoder_out = self.decoder(
            queries,
            encoder_out,
            time_emb,
            encoder_pos_embed=encoder_pos_embed,
            decoder_pos_embed=None,
        )  # (S, B, D)

        decoder_out = decoder_out.transpose(0, 1)  # (B, S, D)
        return self.action_out_proj(decoder_out)  # (B, S, A)

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Flow matching loss (pi05 convention).

        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions
        loss = MSE(v_pred, u_t)  masked over non-padded steps
        """
        actions = batch[ACTION]
        batch_size = actions.shape[0]
        device = actions.device

        encoder_out, encoder_pos_embed = self._encode_observations(batch)

        noise = self.sample_noise(actions.shape, device).to(dtype=actions.dtype)
        if self.config.use_ot_matching:
            noise = ot_matching(noise, actions).to(device=device, dtype=actions.dtype)
        t = self.sample_time(batch_size, device).to(dtype=actions.dtype)

        t_bc = t[:, None, None]
        x_t = t_bc * noise + (1.0 - t_bc) * actions  # pi05 line 733
        u_t = noise - actions  # pi05 line 734

        v_pred = self._decode_velocity(x_t, t, encoder_out, encoder_pos_embed)

        pad_mask = ~batch["action_is_pad"].unsqueeze(-1)
        loss = (F.mse_loss(u_t, v_pred, reduction="none") * pad_mask).mean()
        return loss

    # ------------------------------------------------------------------ #
    # Inference — Heun solver (second-order)                               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate action chunk with Heun ODE solver (2nd-order).

        Heun: predict v at t, Euler-step to t+dt, predict v at t+dt,
              then step with the average velocity.
        Two model evaluations per step vs one for Euler — but O(dt²) accuracy
        allows halving num_inference_steps for the same quality.

        Integration direction: t = 1 → 0,  dt = -1/num_steps  (pi05 convention)
        """
        encoder_out, encoder_pos_embed = self._encode_observations(batch)

        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
            device = batch[OBS_IMAGES][0].device
            dtype = batch[OBS_IMAGES][0].dtype
        else:
            batch_size = batch[OBS_ENV_STATE].shape[0]
            device = batch[OBS_ENV_STATE].device
            dtype = batch[OBS_ENV_STATE].dtype

        action_dim = self.config.action_feature.shape[0]
        num_steps = self.config.num_inference_steps

        # Build time sequence t=1→0 according to the configured schedule.
        if self.config.time_schedule == "quadratic":
            # t_i = (1 - i/N)²  — dense steps near t=0, coarse near t=1.
            # e.g. N=5: [1.00, 0.64, 0.36, 0.16, 0.04, 0.00]
            steps = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
            t_seq = (1.0 - steps) ** 2
        else:
            # "linear": uniform steps [1.0, (N-1)/N, ..., 1/N, 0.0]
            t_seq = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)

        # Start from pure noise at t=1 (pi05 line 823).
        x_t = self.sample_noise((batch_size, self.config.chunk_size, action_dim), device).to(dtype=dtype)

        for step in range(num_steps):
            t_val = t_seq[step].item()
            t_next_val = t_seq[step + 1].item()
            dt = t_next_val - t_val  # negative (moving toward 0)

            t_cur = torch.full((batch_size,), t_val, device=device, dtype=dtype)

            # First evaluation: velocity at current point.
            v1 = self._decode_velocity(x_t, t_cur, encoder_out, encoder_pos_embed)

            # Last step: readout trick or plain Euler.
            if step == num_steps - 1:
                # Direct projection (readout trick): action = x_t - t * v_pred.
                # Derivation: x_t = t*noise + (1-t)*action
                #   -> action = x_t - t*(noise-action) = x_t - t*u_t ≈ x_t - t*v_pred
                # Bypasses the O(dt) Euler truncation error at the endpoint.
                x_t = x_t - t_val * v1 if self.config.use_readout_trick else x_t + dt * v1
                break

            # Euler predictor to next time point (intermediate steps only).
            x_euler = x_t + dt * v1

            # Second evaluation: velocity at Euler-predicted next point.
            t_next = torch.full((batch_size,), t_next_val, device=device, dtype=dtype)
            v2 = self._decode_velocity(x_euler, t_next, encoder_out, encoder_pos_embed)

            # Heun corrector: step with average velocity.
            x_t = x_t + dt * (v1 + v2) / 2.0

        return x_t  # (B, chunk_size, action_dim)


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


class ACTFMPolicy(PreTrainedPolicy):
    """Flow Matching ACT with AdaLN-Zero decoder and Heun ODE solver."""

    config_class = ACTFMConfig
    name = "act_fm"

    def __init__(self, config: ACTFMConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTFlowMatching(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self._failure_postprocessor = None
        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
        if self._failure_postprocessor is not None:
            self._failure_postprocessor.reset()

    def init_failure_postprocessor(self, output_dir=None, failure_handling_json_path=None):
        self._failure_postprocessor = FailurePostprocessor(
            policy=self,
            output_dir=output_dir,
            failure_handling_json_path=failure_handling_json_path,
        )

    def finalize(self):
        if self._failure_postprocessor is not None:
            self._failure_postprocessor.finalize()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        new_actions_chunk = None

        if self.config.temporal_ensemble_coeff is not None:
            new_actions_chunk = self.predict_action_chunk(batch)
            intended_action = self.temporal_ensembler.update(new_actions_chunk)
        else:
            if len(self._action_queue) == 0:
                new_actions_chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
                self._action_queue.extend(new_actions_chunk.transpose(0, 1))
            intended_action = self._action_queue.popleft()

        if self._failure_postprocessor is not None:
            return self._failure_postprocessor.process(
                batch=batch,
                intended_action=intended_action,
                new_actions_chunk=new_actions_chunk,
            )
        return intended_action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return self.model.generate(batch)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        loss = self.model(batch)
        return loss, {"fm_loss": loss.item()}
