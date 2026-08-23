from __future__ import annotations

import torch
import torch.nn as nn
import pytest
import json
from pathlib import Path
import scripts.run_ssl_al_ms_anchored_protocol as protocol

from scripts.ssl_anchored_dino_yolo_obb import (
    AnchoredYOLOOBBStudent,
    anchored_distillation_loss,
    load_artifact,
    relative_anchor_loss,
)
from scripts.run_ssl_al_ms_anchored_protocol import (
    _complete,
    _file_sha256,
    _frozen_checkpoint_map,
    _paired_summary,
    _pretraining_complete,
    _ssl_eligible,
)
from scripts.train_ssl_anchored_dino_yolo_obb import _build_optimizer, _source_group_round_robin
from scripts.create_poultry_deduplicated_trainval import _deduplicate


class TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.ModuleList([nn.Conv2d(3, 4, 1, bias=False)])


def test_relative_anchor_loss_is_zero_then_positive() -> None:
    student = TinyStudent()
    anchors = {name: value.detach().clone() for name, value in student.backbone.named_parameters()}
    assert float(relative_anchor_loss(student, anchors).detach()) == pytest.approx(0.0)
    with torch.no_grad():
        next(student.backbone.parameters()).add_(0.1)
    assert float(relative_anchor_loss(student, anchors).detach()) > 0.0


def test_anchored_loss_combines_feature_and_anchor_terms() -> None:
    student = TinyStudent()
    anchors = {name: value.detach().clone() for name, value in student.backbone.named_parameters()}
    student_tokens = {"p3": torch.randn(2, 4, 8)}
    teacher_tokens = torch.randn(2, 4, 8)
    total, feature, anchor = anchored_distillation_loss(
        student_tokens, teacher_tokens, student, anchors, anchor_weight=0.05, scale_weights={"p3": 1.0}
    )
    assert torch.isfinite(total)
    assert float(total.detach()) == pytest.approx(float((feature + 0.05 * anchor).detach()))


def test_negative_anchor_weight_is_rejected() -> None:
    student = TinyStudent()
    anchors = {name: value.detach().clone() for name, value in student.backbone.named_parameters()}
    with pytest.raises(ValueError, match="anchor_weight"):
        anchored_distillation_loss(
            {"p3": torch.randn(1, 4, 8)}, torch.randn(1, 4, 8), student, anchors,
            anchor_weight=-0.1, scale_weights={"p3": 1.0},
        )


def test_finetune_does_not_reload_original_checkpoint_over_ssl() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "train_ssl_initialized_obb.py").read_text()
    assert "pretrained=True" in source
    assert "pretrained=str(init_model)" not in source
    assert 'parser.add_argument("--resume", action="store_true")' in source


def test_real_ultralytics_backbone_is_explicitly_trainable() -> None:
    root = Path(__file__).resolve().parents[1]
    student = AnchoredYOLOOBBStudent(root / "yolo11l-obb.pt", teacher_dim=8)
    parameters = list(student.backbone.parameters())
    assert parameters
    assert all(parameter.requires_grad for parameter in parameters)


def test_artifact_loader_rejects_frozen_backbone_artifact(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pt"
    torch.save(
        {
            "format": "poultry_geo_anchored_dinov2_yolo_obb/v1",
            "backbone_state_dict": {"model.0.weight": torch.ones(1)},
            "metadata": {"trainable_backbone_parameters": 0, "backbone_relative_anchor_loss": 0.0},
        },
        path,
    )
    with pytest.raises(ValueError, match="without a trainable backbone"):
        load_artifact(path)


def test_partial_pretraining_artifact_is_not_complete(tmp_path: Path) -> None:
    artifact = tmp_path / "anchored_ssl_backbone_best.pt"
    artifact.write_bytes(b"partial")
    marker = tmp_path / "pretraining_complete.json"
    assert not _pretraining_complete(marker, artifact, "expected")

    marker.write_text(
        json.dumps(
            {
                "complete": True,
                "config_sha256": "wrong",
                "epochs_completed": 30,
                "best_artifact": str(artifact.resolve()),
                "best_artifact_sha256": _file_sha256(artifact),
                "final_test_used": False,
                "validation_sampling_contract": "source_group_round_robin_v1",
                "optimizer_contract": "adamw_kernel_matrix_decay_bias_norm_no_decay_v1",
                "ms_ssl_manifest_sha256": "manifest-hash",
            }
        ),
        encoding="utf-8",
    )
    assert not _pretraining_complete(marker, artifact, "expected")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["config_sha256"] = "expected"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    assert _pretraining_complete(marker, artifact, "expected")


def test_all_seed_selection_rule_is_enforced() -> None:
    assert _ssl_eligible(0.01, [0.02, 0.01, 0.005], 0.0, True)
    assert not _ssl_eligible(0.01, [0.02, -0.001, 0.011], 0.0, True)
    assert _ssl_eligible(0.01, [0.02, -0.001, 0.011], 0.0, False)
    assert not _ssl_eligible(0.0, [0.0, 0.0, 0.0], 0.0, True)
    assert not _ssl_eligible(0.0, [0.0, 0.0, 0.0], 0.001, False)


def test_validation_order_interleaves_source_groups(tmp_path: Path) -> None:
    root = tmp_path / "ms"
    files: list[Path] = []
    for group in ("a", "b", "c"):
        directory = root / group
        directory.mkdir(parents=True)
        files.extend(directory / f"{index}.jpg" for index in range(4))
    ordered = _source_group_round_robin(files, root=root, seed=7)
    assert len(ordered) == len(files)
    assert len(set(ordered)) == len(files)
    assert {path.parent.name for path in ordered[:3]} == {"a", "b", "c"}


def test_exact_duplicate_images_require_matching_annotations(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    first, second = image_dir / "a.jpg", image_dir / "b.jpg"
    first.write_bytes(b"same pixels")
    second.write_bytes(b"same pixels")
    (label_dir / "a.txt").write_text("1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    (label_dir / "b.txt").write_text("1 0.10000001 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    retained, records = _deduplicate([first, second], "train")
    assert retained == [first]
    assert records[0]["discarded_exact_duplicates"] == [str(second)]

    (label_dir / "b.txt").write_text("1 0.3 0.1 0.4 0.1 0.4 0.2 0.3 0.2\n")
    with pytest.raises(ValueError, match="materially different annotations"):
        _deduplicate([first, second], "train")


def test_adamw_excludes_bias_and_norm_parameters_from_decay() -> None:
    root = Path(__file__).resolve().parents[1]
    student = AnchoredYOLOOBBStudent(root / "yolo11l-obb.pt", teacher_dim=8)
    optimizer = _build_optimizer(
        student,
        {"backbone_lr": 1e-4, "projector_lr": 1e-3, "weight_decay": 0.05},
    )
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(parameters) == len(list(student.parameters()))
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    for group in optimizer.param_groups:
        if group["decay_contract"] == "kernel_or_matrix":
            assert group["weight_decay"] == pytest.approx(0.05)
            assert all(parameter.ndim >= 2 for parameter in group["params"])
        else:
            assert group["weight_decay"] == pytest.approx(0.0)
            assert all(parameter.ndim < 2 for parameter in group["params"])


def test_supervised_completion_requires_valid_marker_and_exact_request(tmp_path: Path) -> None:
    run = tmp_path / "seed7"
    (run / "weights").mkdir(parents=True)
    (run / "results.csv").write_text("epoch,metrics/mAP50-95(B)\n0,0.5\n")
    best = run / "weights" / "best.pt"
    best.write_bytes(b"checkpoint")
    request = {"config_sha256": "abc", "method": "baseline", "seed": 7, "epochs": 100,
               "data": str(tmp_path / "data.yaml"), "init_model": str(tmp_path / "init.pt")}
    (run / "requested_run.json").write_text(json.dumps(request))
    (run / "supervised_baseline.json").write_text("{")
    assert not _complete(run, "baseline", "abc", request)

    marker = {
        "pipeline": "supervised_yolo_obb_burn_in", "seed": 7, "epochs": 100,
        "data": request["data"], "init_model": request["init_model"],
        "best_checkpoint": str(best.resolve()),
    }
    (run / "supervised_baseline.json").write_text(json.dumps(marker))
    assert _complete(run, "baseline", "abc", request)
    changed = {**request, "epochs": 101}
    assert not _complete(run, "baseline", "abc", changed)


def test_paired_summary_uses_small_sample_student_t_interval() -> None:
    summary = _paired_summary([0.01, 0.02, 0.03])
    assert summary["n_pairs"] == 3
    assert summary["mean_delta"] == pytest.approx(0.02)
    assert summary["ci95_low"] < 0.0 < summary["ci95_high"]


def test_frozen_checkpoint_map_is_nested_by_method_and_seed() -> None:
    rows = [
        {"method": "baseline", "seed": 7, "checkpoint": "base.pt", "checkpoint_sha256": "a"},
        {"method": "anchored_ssl", "seed": 7, "checkpoint": "ssl.pt", "checkpoint_sha256": "b"},
    ]
    frozen = _frozen_checkpoint_map(rows, ["baseline", "anchored_ssl"])
    assert frozen == {
        "baseline": {"7": {"path": "base.pt", "sha256": "a"}},
        "anchored_ssl": {"7": {"path": "ssl.pt", "sha256": "b"}},
    }


def test_validation_summary_executes_and_freezes_all_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "training": {"seeds": [7, 17, 42]},
        "methods": ["baseline", "anchored_ssl"],
        "selection": {"minimum_mean_delta": 0.0, "require_all_seeds": True},
    }
    scores = {
        ("baseline", 7): 0.50, ("baseline", 17): 0.51, ("baseline", 42): 0.52,
        ("anchored_ssl", 7): 0.52, ("anchored_ssl", 17): 0.53, ("anchored_ssl", 42): 0.54,
    }

    def fake_metrics(run_dir: Path) -> dict[str, float | int]:
        method = run_dir.parent.name
        seed = int(run_dir.name.removeprefix("seed"))
        value = scores[(method, seed)]
        return {
            "best_epoch": 9, "metrics/mAP50-95(B)": value,
            "metrics/mAP50(B)": value + 0.1,
            "metrics/precision(B)": 0.8, "metrics/recall(B)": 0.7,
        }

    monkeypatch.setattr(protocol, "_requested_run_payload", lambda *args: {})
    monkeypatch.setattr(protocol, "_complete", lambda *args: True)
    monkeypatch.setattr(protocol, "_best_validation", fake_metrics)
    monkeypatch.setattr(protocol, "_file_sha256", lambda path: f"sha-{path.parent.parent.name}-{path.parent.name}")
    decision = protocol._summarize_validation(config, tmp_path, "config-sha")
    assert decision["winner"] == "anchored_ssl"
    assert decision["all_seed_deltas_meet_requirement"] is True
    assert set(decision["frozen_checkpoints"]) == {"baseline", "anchored_ssl"}
    assert set(decision["frozen_checkpoints"]["baseline"]) == {"7", "17", "42"}
    saved = json.loads((tmp_path / "validation" / "frozen_selection.json").read_text())
    assert saved["frozen_for_explicit_test_phase"] is True
