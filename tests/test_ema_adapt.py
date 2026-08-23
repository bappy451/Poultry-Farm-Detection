from __future__ import annotations

import numpy as np
import torch


def test_paired_naip_views_are_rgb_and_synchronized(tmp_path):
    from poultry_geo.ema_adapt import PairedNAIPDataset

    np.save(tmp_path / "tile.npy", np.random.default_rng(0).random((4, 512, 512), dtype=np.float32))
    sample = PairedNAIPDataset(tmp_path, crop_size=448)[0]
    assert sample["weak"].shape == (3, 448, 448)
    assert sample["strong"].shape == (3, 448, 448)
    assert sample["weak"].dtype == torch.float32
    assert bool((sample["weak"] >= 0).all() and (sample["weak"] <= 1).all())


def test_ema_trainer_reserves_adaptation_options():
    from poultry_geo.ema_adapt import EMADomainAdaptOBBTrainer

    assert "unlabeled_root" in EMADomainAdaptOBBTrainer._CUSTOM_OPTIONS
    assert "ema_decay" in EMADomainAdaptOBBTrainer._CUSTOM_OPTIONS

def test_stage2_checkpoint_argument_is_explicit(tmp_path):
    from scripts.train_ema_adapt import _checkpoint_pretrained_argument

    checkpoint = tmp_path / "stage1" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert _checkpoint_pretrained_argument(checkpoint) == str(checkpoint.resolve())

def test_publication_runner_rejects_stale_training_artifacts(tmp_path):
    from poultry_geo import __path__  # keep package import behavior exercised
    from scripts.run_ema_benchmark import _assert_fresh_training_dir

    run_dir = tmp_path / "seed0"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text("stale", encoding="utf-8")

    import pytest
    with pytest.raises(RuntimeError, match="already contains artifacts"):
        _assert_fresh_training_dir(run_dir, allow_existing=False)
    _assert_fresh_training_dir(run_dir, allow_existing=True)