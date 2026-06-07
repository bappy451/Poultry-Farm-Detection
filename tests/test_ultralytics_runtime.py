from poultry_geo.ultralytics_runtime import infer_ultralytics_family


def test_infer_ultralytics_family_detects_rtdetr() -> None:
    assert infer_ultralytics_family("rtdetr-l.pt") == "rtdetr"
    assert infer_ultralytics_family("weights/RTDETR-x-best.pt") == "rtdetr"


def test_infer_ultralytics_family_defaults_to_yolo() -> None:
    assert infer_ultralytics_family("yolo11l-seg.pt") == "yolo"
    assert infer_ultralytics_family("weights/best.pt") == "yolo"
