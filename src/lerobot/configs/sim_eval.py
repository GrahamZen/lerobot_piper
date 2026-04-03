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

import datetime as dt
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

from lerobot import envs, policies  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.scripts.lerobot_record import DatasetRecordConfig

logger = getLogger(__name__)


@dataclass
class SimEvalConfig:
    """Config for lerobot-sim-eval: simulation evaluation with failure recovery and dataset recording.

    Dataset recording (Phase 1) is enabled by providing --dataset.repo_id and --dataset.single_task.
    The dataset root defaults to HF_LEROBOT_HOME/repo_id (same as lerobot-record) unless
    --dataset.root=<path> is set explicitly.

    Failure recovery (Phase 2) is activated automatically when failure_handling.json exists in the
    pretrained model directory with enable_failure_handling=true.
    """

    env: envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    # Optional nested dataset config — same structure as lerobot-record.
    # Provide --dataset.repo_id and --dataset.single_task to enable recording.
    dataset: DatasetRecordConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    rename_map: dict[str, str] = field(default_factory=dict)
    trust_remote_code: bool = False
    # Dataset saving
    save_only_success: bool = False
    # Rerun visualization
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        else:
            logger.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type if self.policy is not None else 'scratch'}"
            else:
                self.job_name = (
                    f"{self.env.type}_{self.policy.type if self.policy is not None else 'scratch'}"
                )
            logger.warning(f"No job name provided, using '{self.job_name}' as job name.")

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/sim_eval") / eval_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]
