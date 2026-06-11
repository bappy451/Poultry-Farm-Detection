import pytest
import json
import shutil
from pathlib import Path
from downstream_finetune.prepare_data import prepare_full_dataset

@pytest.fixture
def mock_dataset(tmp_path):
    # Create a mock COCO dataset structure
    data_dir = tmp_path / "Data"
    train_dir = data_dir / "poultry_houses_Training" / "train"
    train_dir.mkdir(parents=True)
    
    test_dir = data_dir / "poultry_houses_test" / "test"
    test_dir.mkdir(parents=True)
    
    coco_data = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "img2.jpg", "width": 100, "height": 100},
            {"id": 3, "file_name": "img3.jpg", "width": 100, "height": 100},
            {"id": 4, "file_name": "img4.jpg", "width": 100, "height": 100},
            {"id": 5, "file_name": "img5.jpg", "width": 100, "height": 100},
            {"id": 6, "file_name": "img6.jpg", "width": 100, "height": 100},
            {"id": 7, "file_name": "img7.jpg", "width": 100, "height": 100},
            {"id": 8, "file_name": "img8.jpg", "width": 100, "height": 100},
            {"id": 9, "file_name": "img9.jpg", "width": 100, "height": 100},
            {"id": 10, "file_name": "img10.jpg", "width": 100, "height": 100},
        ],
        "categories": [{"id": 1, "name": "poultry_house"}],
        "annotations": [
            {"image_id": i, "category_id": 1, "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]]}
            for i in range(1, 11)
        ]
    }
    
    with open(train_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco_data, f)
    
    for i in range(1, 11):
        (train_dir / f"img{i}.jpg").touch()
        
    test_coco_data = {
        "images": [{"id": 101, "file_name": "test1.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "poultry_house"}],
        "annotations": [{"image_id": 101, "category_id": 1, "segmentation": [[5, 5, 15, 5, 15, 15, 5, 15]]}]
    }
    with open(test_dir / "_annotations.coco.json", "w") as f:
        json.dump(test_coco_data, f)
    (test_dir / "test1.jpg").touch()
    
    return data_dir

def test_prepare_full_dataset(mock_dataset):
    train_base = mock_dataset / "poultry_houses_Training"
    test_base = mock_dataset / "poultry_houses_test"
    
    prepare_full_dataset(train_base, test_base, val_split=0.2)
    
    # Check if split happened (10 images, 20% split -> 8 train, 2 val)
    train_images = list((train_base / "train" / "images").glob("*.jpg"))
    val_images = list((train_base / "val" / "images").glob("*.jpg"))
    test_images = list((test_base / "test" / "images").glob("*.jpg"))
    
    assert len(train_images) == 8
    assert len(val_images) == 2
    assert len(test_images) == 1
    
    # Check labels
    assert (train_base / "train" / "labels" / (train_images[0].stem + ".txt")).exists()
    assert (train_base / "val" / "labels" / (val_images[0].stem + ".txt")).exists()
    assert (test_base / "test" / "labels" / (test_images[0].stem + ".txt")).exists()
    
    # Check data.yaml
    data_yaml = train_base / "data.yaml"
    assert data_yaml.exists()
    content = data_yaml.read_text()
    assert "train:" in content
    assert "val:" in content
    assert "test:" in content
    assert "nc: 1" in content
