import torch
import torch.nn as nn
import pytest

from poultry_geo.distill_loss import FeatureDistillationLoss
from poultry_geo.distill_models import export_ultralytics_backbone_state, validate_backbone_compatibility
from poultry_geo.train_distill import DistillConfig, train_one_step


def test_exported_encoder_keys_translate_to_detection_backbone():
    student_state = {
        "_orig_mod.encoder.backbone.0.conv.weight": torch.ones(2, 3, 3, 3),
        "encoder.backbone.9.cv2.conv.weight": torch.ones(4, 2, 1, 1),
        "projectors.p4.proj.weight": torch.ones(3, 2, 1, 1),
    }
    exported = export_ultralytics_backbone_state(student_state)
    assert set(exported) == {"model.0.conv.weight", "model.9.cv2.conv.weight"}


def test_backbone_validation_rejects_partial_or_wrong_shape_loads():
    target = {"model.0.weight": torch.zeros(2, 3), "model.10.weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="missing=1"):
        validate_backbone_compatibility(target, {})
    with pytest.raises(ValueError, match="shape_mismatches=1"):
        validate_backbone_compatibility(target, {"model.0.weight": torch.zeros(3, 2)})
    loaded = validate_backbone_compatibility(target, {"model.0.weight": torch.ones(2, 3)})
    assert torch.equal(loaded["model.0.weight"], torch.ones(2, 3))


def test_multiscale_loss_resizes_teacher_tokens():
    student = {"p3": torch.randn(2, 16, 8), "p4": torch.randn(2, 4, 8), "p5": torch.randn(2, 1, 8)}
    loss = FeatureDistillationLoss()(student, torch.randn(2, 9, 8))
    assert torch.isfinite(loss)


def test_training_uses_raw_student_images_and_normalized_teacher_images():
    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.seen = None
        def forward(self, images):
            self.seen = images.detach().clone()
            return {"p4": self.weight * images.mean(dim=1).reshape(images.shape[0], -1, 1).expand(-1, -1, 2)}

    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None
        def forward_features(self, images):
            self.seen = images.detach().clone()
            return torch.ones(images.shape[0], 16, 2, device=images.device)

    student, teacher = Student(), Teacher()
    images = torch.full((1, 3, 128, 128), 0.5)
    config = DistillConfig(precision="fp32", student_model="yolo11n.yaml", image_size=128)
    train_one_step(student, teacher, FeatureDistillationLoss({"p4": 1.0}),
                   torch.optim.SGD(student.parameters(), lr=0.01), None, images, config)
    assert torch.equal(student.seen, images)
    assert not torch.equal(teacher.seen, images)
