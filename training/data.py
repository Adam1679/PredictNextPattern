import json
import math
import os
import random
import time
from bisect import bisect_right

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


class Timer:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        print(f"{self.msg}: {time.time() - self.t:.2f} seconds")


class OHLCDatasetMmap(IterableDataset):
    def __init__(
        self,
        data_root,
        window_range=(30, 4096),
        random_seed=42,
        is_train=True,
        filter_symbols=None,
        filter_types=None,
        filter_intervals=None,
        first_n=float("inf"),
    ):
        self.window_range = window_range
        self.window = window_range[0] if window_range[0] == window_range[1] else None
        self.is_train = is_train
        self.data_root = os.path.join(data_root, "train" if is_train else "test")
        self.first_n = first_n
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)

        if not os.path.exists(os.path.join(data_root, "metadata.json")):
            raise ValueError("metadata.json not found in {}".format(data_root))

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
                size = min(meta["shape"][0], self.first_n)
                self.prefix_sum.append(self.prefix_sum[-1] + size)

        self.total_size = self.prefix_sum[-1]

    def __len__(self):
        return self.total_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.random_seed + worker_id)
        i = 0
        per_worker = int(math.ceil(self.total_size / float(num_workers)))
        print(f"Worker {worker_id} will process {per_worker} samples")
        start = worker_id * per_worker
        end = min(start + per_worker, self.total_size)

        while i < per_worker:
            randint = rng.randint(start, end)
            i += 1
            yield self.__getitem__(randint)

    def __getitem__(self, index):
        bin_idx = bisect_right(self.prefix_sum, index) - 1
        offset = index - self.prefix_sum[bin_idx]
        split_name, meta = self.split_file_metas[bin_idx]
        _, symbol, type_str, interval, _ = split_name.split("_")
        filename = os.path.join(self.data_root, meta["filename"])
        max_time_len = min(meta["shape"][0], self.first_n)
        arr = np.memmap(filename, dtype=np.float32, mode="r", shape=tuple(meta["shape"]))
        start = offset
        if self.window:
            window = self.window
        else:
            window = self.rng.randint(*self.window_range)
        end = min(offset + window, max_time_len)
        actual_length = end - start

        ohlcv = torch.zeros((actual_length, 5), dtype=torch.float32)
        ohlcv[:actual_length, 0] = torch.tensor(arr[start:end, 0], dtype=torch.float32)
        ohlcv[:actual_length, 1] = torch.tensor(arr[start:end, 1], dtype=torch.float32)
        ohlcv[:actual_length, 2] = torch.tensor(arr[start:end, 2], dtype=torch.float32)
        ohlcv[:actual_length, 3] = torch.tensor(arr[start:end, 3], dtype=torch.float32)
        ohlcv[:actual_length, 4] = torch.tensor(arr[start:end, 4], dtype=torch.float32)
        return {
            "symbol": symbol,
            "type": type_str,
            "interval": interval,
            "inputs": ohlcv,
            "bar_start": start,
            "bar_end": end,
        }

    def collate_fn(self, batch, pad_value=0):
        # Separate inputs and targets
        inputs = [item["inputs"] for item in batch]

        # Get the maximum sequence length in the batch
        max_len = max(seq.size(0) for seq in inputs)

        # Pad sequences to max_len
        padded_inputs = []
        for seq in inputs:
            # seq: (seq_len, 5)
            if seq.size(0) < max_len:
                padding = torch.full(
                    (max_len - seq.size(0), seq.size(1)),
                    pad_value,
                    dtype=seq.dtype,
                    device=seq.device,
                )
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq[:max_len, :]
            padded_inputs.append(padded_seq)

        # Stack the padded sequences
        stacked_inputs = torch.stack(padded_inputs)
        batched_inputs = {
            "symbol": [item["symbol"] for item in batch],
            "type": [item["type"] for item in batch],
            "interval": [item["interval"] for item in batch],
            "inputs": stacked_inputs,
        }
        return batched_inputs


if __name__ == "__main__":
    dataset = OHLCDatasetMmap("memmap_dataset", window_range=(1600, 4096), is_train=True)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=16, collate_fn=dataset.collate_fn)
    for i, data in enumerate(dataloader):
        print(data["inputs"].shape)
        if i > 500:
            break
