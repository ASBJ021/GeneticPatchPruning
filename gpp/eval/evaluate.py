import os
import sys
import types
import unittest
import tempfile
from unittest import mock

import torch
from torch import nn
from torch.utils.data import Dataset

# Stub out `clip` before importing the trainer so model.py doesn't try to load real CLIP weights.
class _DummyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1)

class _DummyClipModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _DummyVisual()

    def float(self):
        return self

def _clip_load(model_id: str, device: str):
    return _DummyClipModel(), None

dummy_clip = types.ModuleType("clip")
dummy_clip.load = _clip_load
sys.modules["clip"] = dummy_clip  # ensure the stub is used during import

import trainer  # noqa: E402  (import after stubbing clip)


class AccuracyAtThresholdTest(unittest.TestCase):
    def test_perfect_accuracy_when_logits_match_targets(self) -> None:
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        acc = trainer.accuracy_at_threshold(logits, targets)

        self.assertAlmostEqual(acc, 1.0)

    def test_threshold_controls_predicted_bits(self) -> None:
        logits = torch.tensor([[0.8, 0.0, 0.0]])
        targets = torch.tensor([[1.0, 0.0, 1.0]])

        loose_acc = trainer.accuracy_at_threshold(logits, targets, thresh=0.4)
        strict_acc = trainer.accuracy_at_threshold(logits, targets, thresh=0.75)

        self.assertGreater(loose_acc, strict_acc)


class DummyPatchDataset(Dataset):
    def __init__(self, length: int = 8, num_classes: int = 5, img_size: int = 32) -> None:
        self.length = length
        self.num_classes = num_classes
        self.img_size = img_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        image = torch.randn(3, self.img_size, self.img_size)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[idx % self.num_classes] = 1.0
        return image, target


class DummyModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrainerCheckpointTest(unittest.TestCase):
    def test_checkpoint_artifacts_are_created(self) -> None:
        dataset = DummyPatchDataset()

        with tempfile.TemporaryDirectory(dir=".") as tmpdir, \
             mock.patch.object(trainer, "load_completed_indices", return_value=len(dataset)) as load_completed_mock, \
             mock.patch.object(trainer, "load_data_normal", return_value=(None, None)) as load_data_mock, \
             mock.patch.object(trainer, "PatchIndexDataset", return_value=dataset) as patch_dataset_mock, \
             mock.patch.object(trainer, "split_dataset", return_value=(dataset, dataset, dataset)) as split_dataset_mock, \
             mock.patch.object(trainer, "PatchPredictionModel", DummyModel), \
             mock.patch.dict(trainer.cfg, {"num_workers": 0}, clear=False):

            argv = [
                "trainer.py",
                "--epochs", "1",
                "--batch_size", "2",
                "--save_dir", tmpdir,
                "--data_dir", "dummy.jsonl",
                "--device", "cpu",
            ]
            with mock.patch.object(sys, "argv", argv):
                trainer.main()

            epoch_ckpt = os.path.join(tmpdir, "checkpoint_epoch_001.pt")
            best_ckpt = os.path.join(tmpdir, "checkpoint_best.pt")
            self.assertTrue(os.path.exists(epoch_ckpt), "Epoch checkpoint not written")
            self.assertTrue(os.path.exists(best_ckpt), "Best checkpoint not written")

            load_completed_mock.assert_called_once_with("dummy.jsonl")
            load_data_mock.assert_called_once_with(
                trainer.cfg["dataset_name"],
                len(dataset),
                SPLIT=trainer.cfg["split"],
            )
            patch_dataset_mock.assert_called_once_with(
                ds=None,
                jsonl_path="dummy.jsonl",
                img_size=trainer.cfg["img_size"],
            )
            split_dataset_mock.assert_called_once_with(
                dataset,
                ratios=(0.7, 0.15, 0.15),
                seed=trainer.cfg["seed"],
            )

            epoch_payload = torch.load(epoch_ckpt, map_location="cpu")
            best_payload = torch.load(best_ckpt, map_location="cpu")

            for payload in (epoch_payload, best_payload):
                self.assertIn("epoch", payload)
                self.assertIn("model_state_dict", payload)
                self.assertIn("optimizer_state_dict", payload)
                self.assertIn("train_loss", payload)
                self.assertIn("val_loss", payload)

            self.assertEqual(epoch_payload["epoch"], 1)
            self.assertEqual(best_payload["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
