import json
import os
import random
from bisect import bisect_right
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class OHLCDatasetHF(Dataset):
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
        raise NotImplementedError("This method should be implemented in the derived class")
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


class OHLCDatasetMmap:
    def __init__(
        self,
        data_root,
        window=None,
        window_range=(30, 4096),
        random_seed=42,
        is_train=True,
        filter_symbols=None,
        filter_types=None,
        filter_intervals=None,
    ):
        worker_info = torch.utils.data.get_worker_info()
        self.num_workers = worker_info.num_workers if worker_info is not None else 1
        self.num_proc = max(1, os.cpu_count() - 4)
        self.worker_id = worker_info.id if worker_info is not None else 0
        self.window_range = window_range
        self.window = window
        self.is_train = is_train
        self.data_root = os.path.join(data_root, "train" if is_train else "test")
        self.rng = random.Random(random_seed + self.worker_id)
        with open(os.path.join(data_root, "metadata.json"), "r") as f:
            metadata = json.load(f)
            self.columns = metadata["columns"]
            if is_train:
                split = "train"
            else:
                split = "test"
            self.split_file_metas = [
                (k, v) for k, v in metadata["splits"].items() if k.endswith(split)
            ]
            self.split_file_metas.sort(key=lambda x: x[0])
            if filter_symbols:
                self.split_file_metas = [
                    (k, v) for k, v in self.split_file_metas if k.split("_")[1] in filter_symbols
                ]
            if filter_types:
                self.split_file_metas = [
                    (k, v) for k, v in self.split_file_metas if k.split("_")[2] in filter_types
                ]
            if filter_intervals:
                self.split_file_metas = [
                    (k, v) for k, v in self.split_file_metas if k.split("_")[3] in filter_intervals
                ]
            self.prefix_sum = [0]
            for _, meta in self.split_file_metas:
                self.prefix_sum.append(self.prefix_sum[-1] + meta["shape"][0])

        assert not (
            window is not None and window_range is not None
        ), "Either window or window_range should be None"

    def __len__(self):
        cnt = 0
        for _, meta in self.split_file_metas:
            cnt += meta["shape"][0]
        return cnt

    def __getitem__(self, index):
        bin_idx = bisect_right(self.prefix_sum, index) - 1
        offset = index - self.prefix_sum[bin_idx]
        split_name, meta = self.split_file_metas[bin_idx]
        _, symbol, type_str, interval, _ = split_name.split("_")
        filename = os.path.join(self.data_root, meta["filename"])
        max_time_len = meta["shape"][0]
        arr = np.memmap(filename, dtype=np.float32, mode="r", shape=tuple(meta["shape"]))
        start = offset
        if self.window:
            window = self.window
        else:
            window = self.rng.randint(*self.window_range)
        end = min(offset + window, max_time_len)
        ohlc = torch.zeros((4, window), dtype=torch.float32)
        volume = torch.zeros(window, dtype=torch.float32)
        ohlc[0, start:end] = torch.tensor(arr[start:end, 0], dtype=torch.float32)
        ohlc[1, start:end] = torch.tensor(arr[start:end, 1], dtype=torch.float32)
        ohlc[2, start:end] = torch.tensor(arr[start:end, 2], dtype=torch.float32)
        ohlc[3, start:end] = torch.tensor(arr[start:end, 3], dtype=torch.float32)
        volume[start:end] = torch.tensor(arr[start:end, 4], dtype=torch.float32)
        return {
            "symbol": symbol,
            "type": type_str,
            "interval": interval,
            "ohlc": ohlc,
            "volume": volume,
        }


if __name__ == "__main__":
    # dataset = OHLCDatasetMmap('memmap_dataset', window_range=(16, 48), is_train=True)
    dataset = OHLCDatasetMmap("memmap_dataset", window=2048, window_range=None, is_train=True)
    # print(len(dataset))
    # print(len(dataset[0]))
    # print(dataset[10])
    # print("#" * 100)
    # print(dataset[100])
    # print("#" * 100)
    # print(dataset[10000])
    # print("#" * 100)
    # print(dataset[100000])
    dataloader = DataLoader(dataset, batch_size=4, num_workers=0)
    for i, data in enumerate(dataloader):
        # print(data)
        print(i)
        if i > 5:
            break
