import unittest

import torch
from torch.utils.data import DataLoader

from training.data import OHLCDatasetMmap
from training.model import CryptoLlama, CryptoLlamaModel
from training.utils import evaluation_metrics, evaluation_metrics_single, get_lr


class TestModelAndUtils(unittest.TestCase):
    def setUp(self):
        self.config = CryptoLlama(
            input_size=4,
            output_size=4,
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=1024,
        )
        self.model = CryptoLlamaModel(self.config)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, CryptoLlamaModel)
        self.assertEqual(self.model.config.input_size, 4)
        self.assertEqual(self.model.config.output_size, 4)

    def test_model_forward_pass(self):
        batch_size = 2
        seq_len = 100
        inputs = torch.randn(batch_size, seq_len, self.config.input_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        outputs = self.model(inputs=inputs, attention_mask=attention_mask)

        self.assertEqual(outputs.shape, (batch_size, seq_len, self.config.output_size))

    def test_evaluation_metrics_single(self):
        predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        labels = torch.tensor([[1.1, 1.9], [3.1, 3.9]])
        mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)

        evaluation_metrics_single(predictions, labels, mask)

        # self.assertIn("mae", metrics)
        # self.assertIn("mse", metrics)
        # self.assertIn("recall", metrics)
        # self.assertIn("precision", metrics)
        # self.assertIn("f1", metrics)

    def test_evaluation_metrics(self):
        predictions = torch.randn(2, 3, 4)
        labels = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3, dtype=torch.float32)

        evaluation_metrics(predictions, labels, mask)

        # self.assertIn("high/mae", metrics)
        # self.assertIn("low/mse", metrics)
        # self.assertIn("close/recall", metrics)

    def test_get_lr(self):
        config = {
            "optimizer": {"lr": 0.001, "min_lr": 0.0001, "warmup_steps": 1000, "total_steps": 10000}
        }

        lr_start = get_lr(config, 0)
        lr_warmup = get_lr(config, 500)
        lr_middle = get_lr(config, 5000)
        lr_end = get_lr(config, 10000)

        self.assertAlmostEqual(lr_start, 0.0, delta=1e-6)
        self.assertGreater(lr_warmup, lr_start)
        self.assertLess(lr_middle, config["optimizer"]["lr"])
        self.assertGreater(lr_middle, config["optimizer"]["min_lr"])
        self.assertAlmostEqual(lr_end, config["optimizer"]["min_lr"], delta=1e-6)

    def test_ohlc_dataset_mmap(self):
        dataset = OHLCDatasetMmap(
            "memmap_dataset",
            window_range=(30, 100),
            is_train=True,
            filter_intervals=["1h"],
            filter_symbols=["BTCUSDT"],
            first_n=1000,
        )

        self.assertGreater(len(dataset), 0)

        item = dataset[0]
        self.assertIn("inputs", item)
        self.assertIn("symbol", item)
        self.assertIn("interval", item)
        self.assertEqual(item["symbol"], "BTCUSDT")
        self.assertEqual(item["interval"], "1h")
        self.assertGreaterEqual(item["inputs"].shape[0], 30)
        self.assertLessEqual(item["inputs"].shape[0], 100)

    def test_dataloader(self):
        dataset = OHLCDatasetMmap(
            "memmap_dataset",
            window_range=(30, 100),
            is_train=True,
            filter_intervals=["1h"],
            filter_symbols=["BTCUSDT"],
            first_n=1000,
        )
        dataloader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn)

        batch = next(iter(dataloader))
        self.assertIn("inputs", batch)
        self.assertIn("attention_mask", batch)
        self.assertEqual(len(batch["symbol"]), 16)
        self.assertEqual(batch["inputs"].shape[0], 16)
        self.assertEqual(batch["attention_mask"].shape[0], 16)


if __name__ == "__main__":
    unittest.main()
