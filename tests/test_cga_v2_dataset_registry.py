from pathlib import Path

import yaml
from PIL import Image

from dataset import TestSetLoader, TrainSetLoader, read_split_items, resolve_item_paths
from tools.official.check_cga_v2_dataset_preflight import run_preflight


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=value).save(path)


def _make_registry(path: Path) -> None:
    data = {
        "datasets": {
            "TOY": {
                "root_name": "TOY",
                "image_dir": "images",
                "mask_dir": "masks",
                "list_dir": "img_idx",
                "splits": {
                    "train": "train_TOY.txt",
                    "test": "test_TOY.txt",
                    "hcval": "hcval_TOY.txt",
                },
                "expected_counts": {"train": None, "test": None, "hcval": None},
                "item_format": {
                    "list_has_extension": False,
                    "image_suffix": ".png",
                    "mask_suffix": ".png",
                    "id_prefix": "",
                    "id_suffix": "",
                    "strip_extension_from_list_item": True,
                },
                "mask_policy": {
                    "type": "binary",
                    "allowed_values": [0, 255],
                    "canonical_required": False,
                },
                "claim_policy": {
                    "role": "test_only",
                    "external_validation_claim_allowed": False,
                    "benchmark_validation_claim_allowed": False,
                },
            }
        }
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_dataset_registry_loader_and_preflight(tmp_path):
    dataset_dir = tmp_path / "datasets"
    root = dataset_dir / "TOY"
    registry = tmp_path / "datasets.yaml"
    _make_registry(registry)
    (root / "img_idx").mkdir(parents=True)
    (root / "img_idx" / "train_TOY.txt").write_text("a\n", encoding="utf-8")
    (root / "img_idx" / "test_TOY.txt").write_text("b.png\n", encoding="utf-8")
    (root / "img_idx" / "hcval_TOY.txt").write_text("", encoding="utf-8")
    for item in ["a", "b"]:
        _write_png(root / "images" / f"{item}.png", 10)
        _write_png(root / "masks" / f"{item}.png", 0)
    _write_png(root / "masks" / "a.png", 255)
    _write_png(root / "masks" / "b.png", 255)

    assert read_split_items(dataset_dir, "TOY", "test", registry) == ["b"]
    image_path, mask_path = resolve_item_paths(dataset_dir, "TOY", "b", registry)
    assert image_path.exists()
    assert mask_path.exists()

    train = TrainSetLoader(dataset_dir, "TOY", patch_size=16, registry_path=registry)
    image, mask = train[0]
    assert image.shape == (1, 16, 16)
    assert mask.shape == (1, 16, 16)

    test = TestSetLoader(dataset_dir, "TOY", "TOY", registry_path=registry)
    image, mask, size, item_id = test[0]
    assert image.shape == (1, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert size.tolist() == [32, 32]
    assert item_id == "b"

    summary = run_preflight(dataset_dir, "TOY", registry)
    assert summary["gate_pass"] is True
    assert summary["hcval_count"] == 0
