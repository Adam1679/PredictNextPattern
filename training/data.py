import os
import random
from datetime import datetime

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class OHLCDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        window=None,
        window_range=(30, 4096),
        max_date=None,
        min_date=None,
        random_seed=42,
        split="spot.1m",
    ):
        worker_info = torch.utils.data.get_worker_info()
        self.num_workers = worker_info.num_workers if worker_info is not None else 1
        self.num_proc = max(1, os.cpu_count() - 4)
        self.worker_id = worker_info.id if worker_info is not None else 0
        self.window_range = window_range
        self.window = window
        self.all_dataset = load_dataset(dataset_name, num_proc=self.num_proc, split=split).shard(
            num_shards=self.num_workers, index=self.worker_id, contiguous=True
        )
        assert not (
            max_date is not None and min_date is not None
        ), "Either max_date or min_date should be None"
        assert not (
            window is not None and window_range is not None
        ), "Either window or window_range should be None"
        if max_date is not None:
            train_split_timestamp_ns = datetime.strptime(max_date, "%Y-%m-%d")
            self.all_dataset = self.all_dataset.filter(
                lambda x: x["open_timestamp"] <= train_split_timestamp_ns, num_proc=self.num_proc
            )
        if min_date is not None:
            train_split_timestamp_ns = datetime.strptime(min_date, "%Y-%m-%d")
            self.all_dataset = self.all_dataset.filter(
                lambda x: x["open_timestamp"] >= train_split_timestamp_ns, num_proc=self.num_proc
            )

        self.rng = random.Random(random_seed + self.worker_id)

    def __len__(self):
        return len(self.all_dataset)

    def __getitem__(self, start_idx):
        # Randomly get a window size
        if self.window:
            window = self.window
        else:
            window = self.rng.randint(*self.window_range)
        # Get data for the selected ticker
        range_data = self.all_dataset[start_idx : start_idx + window]
        symbol, type_str, interval = (
            range_data["symbol"][0],
            range_data["type"][0],
            range_data["interval"][0],
        )
        data = {
            "symbol": [],
            "type": [],
            "interval": [],
            "open_price": [],
            "high_price": [],
            "low_price": [],
            "close_price": [],
            "open_timestamp": [],
        }
        for i in range(len(range_data)):
            if (
                range_data["type"][i] == type_str
                and range_data["interval"][i] == interval
                and range_data["symbol"][i] == symbol
            ):
                for key in data.keys():
                    data[key].append(range_data[key][i])
            else:
                break
        open_t = str(data["open_timestamp"][0])
        close_t = str(data["open_timestamp"][-1])
        # Extract prices and convert to tensor
        o = torch.tensor(data["open_price"], dtype=torch.float32)
        h = torch.tensor(data["high_price"], dtype=torch.float32)
        lo = torch.tensor(data["low_price"], dtype=torch.float32)
        c = torch.tensor(data["close_price"], dtype=torch.float32)

        return {
            "symbol": symbol,
            "type": type_str,
            "interval": interval,
            "open_price": o,
            "high_price": h,
            "low_price": lo,
            "close_price": c,
            "window_start": open_t,
            "window_end": close_t,
        }


if __name__ == "__main__":
    dataset = OHLCDataset(
        "adamzzzz/binance-klines-20240721",
        window_range=(16, 48),
        max_date="2024-01-01",
        split="spot.1h",
    )
    print(len(dataset[0]))
    print(dataset[10])
    print("#" * 100)
    print(dataset[100])
    print("#" * 100)
    print(dataset[10000])
    print("#" * 100)
    print(dataset[100000])
    # dataloader = DataLoader(dataset, batch_size=4, num_workers=0)
    # for i, data in enumerate(dataloader):

    #     print(data)
    #     if i > 5:
    #         break
