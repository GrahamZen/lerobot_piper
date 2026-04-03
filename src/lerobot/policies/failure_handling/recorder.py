"""Metrics recorder: buffers per-step dicts and flushes to failure_metrics.jsonl."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class MetricsRecorder:
    """Buffer per-step metric dicts and write them to ``failure_metrics.jsonl``.

    Attributes:
        global_step: Monotonically increasing step counter (never resets).
        episode:     Current episode index (increments on ``reset_episode()``).
        last_row:    The most recently logged metric dict (with stamps), or None.
    """

    def __init__(self, output_dir: Path | None, flush_every_step: bool = False) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        self.flush_every_step = flush_every_step

        self.global_step: int = 0
        self.episode: int = 0
        self.last_row: dict[str, Any] | None = None

        self._buffer: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Stamp *metrics* with episode/step/timestamp and buffer it.

        Tensors in *metrics* are converted to Python scalars or lists so
        the row is JSON-serialisable.

        Returns:
            The stamped row dict (also stored in ``last_row``).
        """
        row = {
            "episode": self.episode,
            "global_step": self.global_step,
            "timestamp": time.time(),
            "is_recovery_frame": False,
        }
        row.update(_to_serialisable(metrics))

        if self.output_dir is not None:
            self._buffer.append(row)

        self.last_row = row
        self.global_step += 1

        if self.flush_every_step:
            self.flush()

        return row

    def log_recovery_frame(self) -> bool:
        """Duplicate the last logged row with ``is_recovery_frame=True``.

        Called by ``lerobot_record.py`` for every frame where the robot
        replays the recovery action (no new inference is run).

        Returns:
            False if there is no previous row to duplicate.
        """
        source = self._buffer[-1] if self._buffer else self.last_row
        if source is None:
            return False

        row = dict(source)
        row["episode"] = self.episode
        row["global_step"] = self.global_step
        row["timestamp"] = time.time()
        row["is_recovery_frame"] = True

        if self.output_dir is not None:
            self._buffer.append(row)

        self.last_row = row
        self.global_step += 1

        if self.flush_every_step:
            self.flush()

        return True

    def discard_current_episode(self) -> int:
        """Remove buffered rows for the current episode from memory.

        Used when ``lerobot-record`` discards and re-records the current
        episode.  Only works while rows are still in the in-memory buffer
        (i.e. before the next ``flush()``).

        Returns:
            Number of rows discarded.
        """
        before = len(self._buffer)
        self._buffer = [r for r in self._buffer if r.get("episode") != self.episode]
        dropped = before - len(self._buffer)

        if dropped > 0:
            logger.info("Discarded %d buffered metric rows for episode %d", dropped, self.episode)

        if self.last_row is not None and self.last_row.get("episode") == self.episode:
            self.last_row = None

        return dropped

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write buffered rows to ``failure_metrics.jsonl`` (append mode)."""
        if not self._buffer or self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fpath = self.output_dir / "failure_metrics.jsonl"

        with fpath.open("a") as f:
            for row in self._buffer:
                f.write(json.dumps(_to_serialisable(row)) + "\n")

        if len(self._buffer) > 1:
            logger.info("Flushed %d metric rows to %s", len(self._buffer), fpath)
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Flush pending rows and advance the episode counter.

        ``global_step`` is NOT reset — it is a global monotonic counter.
        """
        self.flush()
        self.episode += 1
        self.last_row = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _to_serialisable(d: dict) -> dict:
    """Recursively convert Tensor values to Python scalars / lists."""
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.item() if v.numel() == 1 else v.tolist()
        else:
            out[k] = v
    return out
