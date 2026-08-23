from __future__ import annotations

from scripts.checkpoint_retry import CheckpointSaveRetryMixin


class _FailsTwice:
    def __init__(self) -> None:
        self.calls = 0

    def save_model(self):
        self.calls += 1
        if self.calls < 3:
            raise PermissionError("temporary lock")
        return "saved"


class _RetryingTrainer(CheckpointSaveRetryMixin, _FailsTwice):
    checkpoint_save_max_delay = 0.0


def test_checkpoint_save_retries_transient_permission_error(monkeypatch) -> None:
    monkeypatch.setattr("scripts.checkpoint_retry.time.sleep", lambda _: None)
    trainer = _RetryingTrainer()

    assert trainer.save_model() == "saved"
    assert trainer.calls == 3