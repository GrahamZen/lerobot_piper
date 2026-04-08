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
"""Flow Matching ACT (ACT-FM) Configuration

Flow matching convention follows pi0/pi05 (openpi):
  x_t = t * noise + (1 - t) * actions   (t=1 → noise, t=0 → action)
  u_t = noise - actions                  (target velocity)
  Inference: Euler from t=1 → 0, dt = -1/num_inference_steps

Time sampling follows pi05:
  t ~ Beta(alpha, beta), then t = t * scale + offset

Time embedding follows pi05:
  sinusoidal with (min_period, max_period) → two-layer MLP with SiLU

Time injection into ACT decoder:
  time_emb is broadcast-added to each decoder query token
  (pi05 uses AdaRMS conditioning on the transformer layers; since ACT's
   decoder does not have adaptive norm layers we achieve equivalent global
   time conditioning by adding the embedding directly to the query tokens)
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act_fm")
@dataclass
class ACTFMConfig(PreTrainedConfig):
    """Configuration for Flow Matching ACT (ACT-FM).

    Flow matching parameters use the same defaults as pi05 / openpi.

    Args:
        n_obs_steps: Number of environment steps worth of observations.
        chunk_size: Action prediction horizon (number of steps per chunk).
        n_action_steps: Steps to execute per policy invocation (≤ chunk_size).

        vision_backbone: torchvision ResNet variant for image encoding.
        pretrained_backbone_weights: torchvision weights for the backbone.
        replace_final_stride_with_dilation: Dilated convolution in ResNet layer4.

        pre_norm: Pre-norm transformer variant.
        dim_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        dim_feedforward: Feed-forward expansion dimension.
        feedforward_activation: Activation in FFN ("relu"/"gelu"/"glu").
        n_encoder_layers: Number of transformer encoder layers.
        n_decoder_layers: Number of transformer decoder layers.
        dropout: Dropout applied in transformer layers.

        temporal_ensemble_coeff: Exponential weighting coeff for temporal
            ensembling (None = disabled). When enabled, n_action_steps must be 1.

        num_inference_steps: Euler integration steps at inference (pi05 default: 10).
        time_sampling_beta_alpha: Alpha for Beta time distribution (pi05: 1.5).
        time_sampling_beta_beta: Beta for Beta time distribution (pi05: 1.0).
        time_sampling_scale: Scale applied after Beta sample (pi05: 0.999).
        time_sampling_offset: Offset applied after Beta sample (pi05: 0.001).
        min_period: Min period for sinusoidal time embedding (pi05: 4e-3).
        max_period: Max period for sinusoidal time embedding (pi05: 4.0).
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # Transformer.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    dropout: float = 0.1

    # Inference / ensembling.
    temporal_ensemble_coeff: float | None = None

    # Flow matching — defaults match pi05 / openpi exactly.
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Inference time-step schedule.
    # "linear"    — uniform steps: t = 1, (N-1)/N, ..., 1/N, 0  (default)
    # "quadratic" — denser steps near t=0: t_i = (1 - i/N)²
    #               More evaluations where the velocity field is sharpest.
    time_schedule: str = "linear"

    # Terminal readout trick: replaces the last ODE step with a direct
    # projection  action = x_t - t * v_pred, derived from the flow identity:
    #   x_t = t*noise + (1-t)*action  →  action = x_t - t*(noise-action) = x_t - t*u_t
    # Eliminates the O(dt) truncation error at the endpoint.
    # Most effective combined with time_schedule="quadratic" (smaller last t).
    # Default: True — recommended on real hardware to reduce endpoint jitter.
    use_readout_trick: bool = True

    # Training: minibatch Optimal Transport noise-action coupling.
    # False — standard CFM: noise[i] pairs with action[i] in the same batch.
    # True  — OT-CFM: solve linear assignment within each minibatch to find
    #          the minimum-cost noise↔action pairing (Hungarian algorithm).
    #          Reduces flow-line crossings, simplifying the learned velocity
    #          field. Requires scipy (pip install scipy). Negligible overhead
    #          for batch sizes ≤ 256.
    use_ot_matching: bool = False

    # Optimiser presets.
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError("`n_action_steps` must be 1 when using temporal ensembling.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be >= n_action_steps ({self.n_action_steps})."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not supported yet. Got n_obs_steps={self.n_obs_steps}."
            )
        if self.time_schedule not in ("linear", "quadratic"):
            raise ValueError(f"`time_schedule` must be 'linear' or 'quadratic'. Got {self.time_schedule!r}.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("Provide at least one image or the environment state as input.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
