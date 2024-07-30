import time
import unittest

from torch.utils.data import DataLoader

from training.data import OHLCDatasetMmap


class TestTrain(unittest.TestCase):
    def test_train_dataloading_speed(self):
        global_max_index = -1
        for rank in range(8):
            dataset = OHLCDatasetMmap(
                "memmap_dataset",
                window_range=(1024, 2048),
                is_train=True,
                rank=rank,
                filter_intervals=["1h"],
                world_size=8,
            )
            val_loader = DataLoader(
                dataset, batch_size=32, num_workers=8, collate_fn=dataset.collate_fn
            )
            start = time.time()
            min_index = 1e12
            max_index = 0
            for i, data in enumerate(val_loader):
                max_index = max(*data["index"], max_index)
                min_index = min(min_index, *data["index"])
                if global_max_index > 0:
                    self.assertTrue(
                        global_max_index < min_index,
                        "Data overlapping {} >= {}".format(global_max_index, min_index),
                    )
                diff = time.time() - start
                i / diff
                if i > 1000:
                    break
            global_max_index = max(global_max_index, max_index)
            self.assertTrue(global_max_index, 60)

    def test_val_dataloading_speed(self):
        val_dataset = OHLCDatasetMmap(
            "memmap_dataset",
            window_range=(1024, 2048),
            is_train=True,
            rank=0,
            world_size=8,
            filter_intervals=["1h"],
            filter_symbols=["BTCUSDT", "ETHUSDT"],
            first_n=1000,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, num_workers=4, collate_fn=val_dataset.collate_fn
        )
        min_index = 1e12
        max_index = 0
        try:
            for i, data in enumerate(val_loader):
                max_index = max(*data["index"], max_index)
                min_index = min(min_index, *data["index"])
                if i > 10000:
                    break
        except:
            assert False, "Data loading failed"


if __name__ == "__main__":
    unittest.main()
