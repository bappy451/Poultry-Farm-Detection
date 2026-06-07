from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from poultry_geo.naip_data import NAIPDataset, get_dataloaders


def _write_rgb(path: Path, *, size: tuple[int, int] = (32, 32), value: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _write_grouped_rgb_tiles(root: Path, groups: int = 4, tiles_per_group: int = 2) -> None:
    for group_index in range(groups):
        for tile_index in range(tiles_per_group):
            _write_rgb(
                root / f"source_{group_index}" / f"tile_{tile_index}.jpg",
                value=20 + group_index * 20 + tile_index,
            )


def test_dataset_discovers_nested_images_in_deterministic_order(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "z_source" / "b.jpg")
    _write_rgb(tmp_path / "a_source" / "c.png")
    _write_rgb(tmp_path / "a_source" / "a.jpeg")

    dataset = NAIPDataset(tmp_path, crop_size=16, augment=False)

    assert [path.relative_to(tmp_path).as_posix() for path in dataset.files] == [
        "a_source/a.jpeg",
        "a_source/c.png",
        "z_source/b.jpg",
    ]
    sample = dataset[0]
    assert set(sample) == {"image", "path"}
    assert sample["path"].endswith("a_source\\a.jpeg") or sample["path"].endswith("a_source/a.jpeg")
    assert sample["image"].shape == (3, 16, 16)
    assert sample["image"].device.type == "cpu"


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ],
)
def test_dataset_supports_requested_floating_precision(
    tmp_path: Path,
    precision: str,
    expected_dtype: torch.dtype,
) -> None:
    _write_rgb(tmp_path / "source" / "tile.jpg", value=128)

    image = NAIPDataset(tmp_path, crop_size=16, precision=precision, augment=False)[0]["image"]

    assert image.dtype == expected_dtype
    assert image.device.type == "cpu"
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0


def test_dataset_loads_four_band_and_n_band_npy_in_chw_or_hwc_layout(tmp_path: Path) -> None:
    four_band = np.stack(
        [np.full((24, 20), value, dtype=np.uint16) for value in (0, 1000, 2000, 4000)]
    )
    five_band = np.moveaxis(
        np.stack([np.full((24, 20), value / 10, dtype=np.float32) for value in range(5)]),
        0,
        -1,
    )
    np.save(tmp_path / "four_band.npy", four_band)
    np.save(tmp_path / "five_band.npy", five_band)

    four = NAIPDataset(
        tmp_path,
        files=[tmp_path / "four_band.npy"],
        crop_size=16,
        channels=4,
        augment=False,
    )[0]["image"]
    five = NAIPDataset(
        tmp_path,
        files=[tmp_path / "five_band.npy"],
        crop_size=16,
        channels=5,
        augment=False,
    )[0]["image"]

    assert four.shape == (4, 16, 16)
    assert five.shape == (5, 16, 16)
    assert four.dtype == torch.float32
    assert five.dtype == torch.float32
    assert float(four.min()) == pytest.approx(0.0)
    assert float(four.max()) == pytest.approx(4000 / np.iinfo(np.uint16).max)
    assert torch.allclose(five[:, 0, 0], torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]))


def test_dataset_loads_multiband_tiff(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "source" / "cir.tif"
    path.parent.mkdir()
    data = np.stack([np.full((20, 24), value, dtype=np.uint16) for value in (0, 1, 2, 3)])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=24,
        height=20,
        count=4,
        dtype=data.dtype,
        transform=from_origin(0, 20, 1, 1),
    ) as dst:
        dst.write(data)

    image = NAIPDataset(tmp_path, crop_size=16, channels=4, augment=False)[0]["image"]

    assert image.shape == (4, 16, 16)
    assert image.dtype == torch.float32


def test_spatial_augmentations_are_synchronized_across_all_bands(tmp_path: Path) -> None:
    base = np.arange(24 * 24, dtype=np.float32).reshape(24, 24) / (24 * 24 - 1)
    np.save(tmp_path / "identical_bands.npy", np.stack([base, base, base, base]))

    torch.manual_seed(7)
    image = NAIPDataset(
        tmp_path,
        crop_size=16,
        channels=4,
        augment=True,
    )[0]["image"]

    assert image.shape == (4, 16, 16)
    assert torch.equal(image[0], image[1])
    assert torch.equal(image[1], image[2])
    assert torch.equal(image[2], image[3])


def test_rgb_augmentation_changes_pixels_while_validation_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "source" / "gradient.png"
    path.parent.mkdir()
    x = np.arange(32, dtype=np.uint8)
    gradient = np.stack(np.meshgrid(x, x), axis=-1)
    rgb = np.concatenate([gradient, gradient[..., :1]], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path)

    validation = NAIPDataset(tmp_path, crop_size=20, augment=False)
    first = validation[0]["image"]
    second = validation[0]["image"]

    torch.manual_seed(11)
    augmented = NAIPDataset(tmp_path, crop_size=20, augment=True)[0]["image"]

    assert torch.equal(first, second)
    assert not torch.equal(first, augmented)


def test_get_dataloaders_returns_batched_cpu_tensors_and_grouped_split(tmp_path: Path) -> None:
    _write_grouped_rgb_tiles(tmp_path, groups=5, tiles_per_group=2)

    train_loader, val_loader = get_dataloaders(
        tmp_path,
        batch_size=2,
        val_fraction=0.4,
        seed=27,
        crop_size=16,
        num_workers=0,
    )

    batch = next(iter(train_loader))
    assert batch["image"].shape == (2, 3, 16, 16)
    assert batch["image"].device.type == "cpu"
    assert batch["image"].dtype == torch.float32
    train_groups = {Path(path).parent for path in train_loader.dataset.files}
    val_groups = {Path(path).parent for path in val_loader.dataset.files}
    assert train_groups
    assert val_groups
    assert train_groups.isdisjoint(val_groups)

    repeated_train, repeated_val = get_dataloaders(
        tmp_path,
        batch_size=2,
        val_fraction=0.4,
        seed=27,
        crop_size=16,
        num_workers=0,
    )
    assert repeated_train.dataset.files == train_loader.dataset.files
    assert repeated_val.dataset.files == val_loader.dataset.files


def test_get_dataloaders_supports_relative_root_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "tiles"
    _write_grouped_rgb_tiles(root, groups=3, tiles_per_group=2)
    monkeypatch.chdir(tmp_path)

    train_loader, val_loader = get_dataloaders(
        Path("tiles"),
        batch_size=2,
        crop_size=16,
        num_workers=0,
    )

    assert next(iter(train_loader))["image"].shape == (2, 3, 16, 16)
    assert len(val_loader.dataset) > 0


def test_dataloader_throughput_options_are_conditional_on_worker_count(tmp_path: Path) -> None:
    _write_grouped_rgb_tiles(tmp_path, groups=3, tiles_per_group=2)

    single_process, _ = get_dataloaders(
        tmp_path,
        batch_size=2,
        crop_size=16,
        num_workers=0,
        prefetch_factor=3,
    )
    worker_loader, _ = get_dataloaders(
        tmp_path,
        batch_size=2,
        crop_size=16,
        num_workers=1,
        prefetch_factor=3,
    )

    assert single_process.pin_memory is True
    assert single_process.num_workers == 0
    assert single_process.persistent_workers is False
    assert single_process.prefetch_factor is None
    assert worker_loader.pin_memory is True
    assert worker_loader.num_workers == 1
    assert worker_loader.persistent_workers is True
    assert worker_loader.prefetch_factor == 3


def test_get_dataloaders_uses_distributed_samplers_when_initialized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from torch.utils.data.distributed import DistributedSampler

    _write_grouped_rgb_tiles(tmp_path, groups=3, tiles_per_group=2)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    train_loader, val_loader = get_dataloaders(
        tmp_path,
        batch_size=2,
        crop_size=16,
        num_workers=0,
    )

    assert isinstance(train_loader.sampler, DistributedSampler)
    assert isinstance(val_loader.sampler, DistributedSampler)
    assert train_loader.sampler.shuffle is True
    assert val_loader.sampler.shuffle is False


@pytest.mark.parametrize("precision", ["float64", "nf4", "invalid"])
def test_invalid_input_precision_is_rejected(tmp_path: Path, precision: str) -> None:
    _write_rgb(tmp_path / "source" / "tile.jpg")

    with pytest.raises(ValueError, match="fp32, fp16, or bf16"):
        NAIPDataset(tmp_path, precision=precision)


def test_dataset_reports_clear_input_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        NAIPDataset(tmp_path / "missing")

    (tmp_path / "unsupported.bmp").write_bytes(b"not used")
    with pytest.raises(ValueError, match="No supported imagery"):
        NAIPDataset(tmp_path)


def test_dataset_rejects_channel_mismatch_ambiguous_npy_and_small_images(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "rgb" / "tile.jpg", size=(12, 12))
    np.save(tmp_path / "ambiguous.npy", np.zeros((4, 4, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="Expected 4 channels"):
        NAIPDataset(
            tmp_path,
            files=[tmp_path / "rgb" / "tile.jpg"],
            crop_size=8,
            channels=4,
        )[0]

    with pytest.raises(ValueError, match="Ambiguous NPY layout"):
        NAIPDataset(
            tmp_path,
            files=[tmp_path / "ambiguous.npy"],
            crop_size=4,
            channels=4,
        )[0]

    with pytest.raises(ValueError, match="crop_size"):
        NAIPDataset(
            tmp_path,
            files=[tmp_path / "rgb" / "tile.jpg"],
            crop_size=16,
        )[0]


def test_dataset_rejects_inconsistent_channels_when_not_explicit(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "rgb" / "tile.jpg")
    np.save(tmp_path / "cir.npy", np.zeros((4, 32, 32), dtype=np.float32))
    dataset = NAIPDataset(tmp_path, crop_size=16, augment=False)

    assert dataset[0]["image"].shape[0] in {3, 4}
    with pytest.raises(ValueError, match="Inconsistent channel count"):
        dataset[1]


def test_real_nested_jpg_smoke_when_assets_are_available() -> None:
    root = Path(__file__).resolve().parents[1] / "tmp" / "test_jpg"
    if not root.exists():
        pytest.skip("Real pre-sliced JPG assets are not available.")

    dataset = NAIPDataset(root, crop_size=224, channels=3, augment=False)
    sample = dataset[0]

    assert sample["image"].shape == (3, 224, 224)
    assert sample["image"].dtype == torch.float32
