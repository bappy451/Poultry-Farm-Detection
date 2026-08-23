"""Retry transient Windows checkpoint-file locks during Ultralytics training."""

from __future__ import annotations

import time

from ultralytics.models.yolo.obb.train import OBBTrainer


class CheckpointSaveRetryMixin:
    """Retry ``save_model`` when Windows briefly locks ``last.pt`` or ``best.pt``."""

    checkpoint_save_attempts = 10
    checkpoint_save_max_delay = 5.0

    def save_model(self):
        delay = 0.5
        for attempt in range(1, self.checkpoint_save_attempts + 1):
            try:
                return super().save_model()
            except PermissionError as exc:
                if attempt == self.checkpoint_save_attempts:
                    raise
                print(
                    f"[warning] Checkpoint file is temporarily locked: {exc}. "
                    f"Retrying save ({attempt}/{self.checkpoint_save_attempts}) in {delay:.1f}s...",
                    flush=True,
                )
                time.sleep(delay)
                delay = min(delay * 2.0, self.checkpoint_save_max_delay)


class RetryingOBBTrainer(CheckpointSaveRetryMixin, OBBTrainer):
    """Standard OBB trainer with resilient checkpoint saves."""


__all__ = ["CheckpointSaveRetryMixin", "RetryingOBBTrainer"]