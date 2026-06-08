from __future__ import annotations

import csv
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass(slots=True)
class RunState:
    status: str
    start_epoch: int
    seed: int
    run_dir: Path
    config: dict[str, Any]


def normalize_training_name(training_name: str) -> str:
    normalized = str(training_name).strip()
    if normalized.lower().startswith("training_name="):
        normalized = normalized.split("=", 1)[1].strip()
    if not normalized:
        raise ValueError("training_name must not be empty.")
    if any(separator in normalized for separator in ("/", "\\")):
        raise ValueError("training_name must be a simple run identifier, not a path.")
    if normalized in {".", ".."}:
        raise ValueError("training_name must not be '.' or '..'.")
    return normalized


class StateManager:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        training_name: str,
        experiment_id: str,
        root_dir: str | Path = "trained_models",
    ) -> None:
        self.config = json.loads(json.dumps(config))
        self.training_name = normalize_training_name(training_name)
        self.experiment_id = str(experiment_id)
        self.root_dir = Path(root_dir)
        self.campaign_dir = self.root_dir / self.training_name
        self.run_dir = self.campaign_dir / self.experiment_id
        self.config_path = self.run_dir / "config_history.yaml"
        self.checkpoint_path = self.run_dir / "model_checkpoint.pt"
        self.metrics_path = self.run_dir / "training_metrics.csv"

    def prepare(self, model, optimizer, scaler=None) -> RunState:
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        if not self.run_dir.exists():
            return self._prepare_new_run()
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Existing run directory '{self.run_dir}' is missing config_history.yaml."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Existing run directory '{self.run_dir}' is missing model_checkpoint.pt."
            )
        return self._prepare_resume(model, optimizer, scaler)

    def append_metrics(
        self,
        *,
        epoch: int,
        loss: float,
        accuracy: float,
        evaluation_matrix: list[list[int]] | tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "epoch": int(epoch),
            "loss": float(loss),
            "accuracy": float(accuracy),
            "evaluation_matrix_json": json.dumps(evaluation_matrix),
        }
        write_header = not self.metrics_path.exists()
        with self.metrics_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["epoch", "loss", "accuracy", "evaluation_matrix_json"],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def save_checkpoint(self, model, optimizer, scaler, *, epoch: int) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        seed = self._resolve_seed(self.config)
        payload = {
            "epoch": int(epoch),
            "seed": int(seed),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": self._extract_scaler_state(scaler),
            "rng_state": self._capture_rng_state(),
        }
        self._atomic_torch_save(payload, self.checkpoint_path)

    def _prepare_new_run(self) -> RunState:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        seed = self._resolve_seed(self.config)
        self._set_global_seed(seed)
        payload = {
            "seed": int(seed),
            "config": self.config,
        }
        self.config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        return RunState(
            status="new",
            start_epoch=0,
            seed=int(seed),
            run_dir=self.run_dir,
            config=self.config,
        )

    def _prepare_resume(self, model, optimizer, scaler) -> RunState:
        stored = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        stored_config = stored.get("config")
        stored_seed = int(stored.get("seed"))
        if stored_config is None:
            raise ValueError(f"Stored config history at '{self.config_path}' is invalid.")
        if self._config_signature(stored_config) != self._config_signature(self.config):
            raise ValueError(
                f"Incoming config for training_name '{self.training_name}' does not match "
                f"stored config_history.yaml."
            )

        self._set_global_seed(stored_seed)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        rng_state = checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            self._restore_rng_state(rng_state)
        return RunState(
            status="resume",
            start_epoch=int(checkpoint["epoch"]) + 1,
            seed=stored_seed,
            run_dir=self.run_dir,
            config=stored_config,
        )

    @staticmethod
    def _config_signature(config: dict[str, Any]) -> str:
        return json.dumps(config, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _resolve_seed(config: dict[str, Any]) -> int:
        runtime = config.get("runtime", {})
        configured = runtime.get("seed")
        if configured is not None:
            return int(configured)
        return random.SystemRandom().randint(0, 2**31 - 1)

    @staticmethod
    def _set_global_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict[str, Any]) -> None:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.random.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    @staticmethod
    def _extract_scaler_state(scaler) -> dict[str, Any] | None:
        if scaler is None:
            return None
        if hasattr(scaler, "is_enabled") and callable(scaler.is_enabled) and not scaler.is_enabled():
            return None
        if not hasattr(scaler, "state_dict"):
            return None
        state = scaler.state_dict()
        return state if state else None

    @staticmethod
    def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f"{path.stem}-",
            suffix=".tmp",
        )
        os.close(handle)
        tmp_path = Path(tmp_name)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


__all__ = ["RunState", "StateManager", "normalize_training_name"]
