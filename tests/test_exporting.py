from __future__ import annotations

from poultry_geo.exporting import export_ultralytics_model


class _FakeModel:
    def __init__(self) -> None:
        self.last_kwargs = None

    def export(self, **kwargs):
        self.last_kwargs = kwargs
        return "artifact.tflite"


def test_export_ultralytics_model_passes_export_settings(monkeypatch) -> None:
    fake_model = _FakeModel()
    monkeypatch.setattr("poultry_geo.exporting.load_ultralytics_model", lambda model_path, family="auto": fake_model)

    output = export_ultralytics_model(
        "weights/best.pt",
        export_format="tflite",
        family="yolo",
        device="cpu",
        image_size=640,
    )

    assert output == "artifact.tflite"
    assert fake_model.last_kwargs == {"format": "tflite", "device": "cpu", "imgsz": 640}

