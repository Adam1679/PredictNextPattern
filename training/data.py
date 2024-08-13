import json
import math
import os
import random
import time
from bisect import bisect_right
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from training.plotting import (
    plot_ohlc_candlestick_with_volume,
    plot_ohlc_candlestick_with_volume_and_prediction,
)


class Timer:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        print(f"{self.msg}: {time.time() - self.t:.2f} seconds")


INTERVAL_TO_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


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
        sample_n=None,
        rank=0,
        world_size=1,
        clip=(-50, 50),
        normalize_rescale_price=True,
    ):
        self.window_range = window_range
        self.rank = rank
        self.world_size = world_size
        self.window = window_range[0] if window_range[0] == window_range[1] else None
        self.is_train = is_train
        self.data_root = os.path.join(data_root, "train" if is_train else "test")
        self.first_n = first_n
        self.sample_n = sample_n
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        self.clip = clip
        self.normalize_rescale_price = normalize_rescale_price
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
                    (k, meta)
                    for k, meta in self.split_file_metas
                    if meta["symbol"] in filter_symbols
                ]
            if filter_types:
                self.split_file_metas = [
                    (k, meta) for k, meta in self.split_file_metas if meta["type"] in filter_types
                ]
            if filter_intervals:
                self.split_file_metas = [
                    (k, meta)
                    for k, meta in self.split_file_metas
                    if meta["interval"] in filter_intervals
                ]
            self.prefix_sum = [0]
            for _, meta in self.split_file_metas:
                size = min(meta["shape"][0], self.first_n)
                self.prefix_sum.append(self.prefix_sum[-1] + size)

        self.total_size = self.prefix_sum[-1]
        # Partition the data based on rank and world_size
        self.start = self.rank * (self.total_size // self.world_size)
        self.end = (self.rank + 1) * (self.total_size // self.world_size)
        if self.rank == self.world_size - 1:
            self.end = self.total_size

    def __len__(self):
        if self.sample_n:
            return min(self.end - self.start, self.sample_n)
        return self.end - self.start

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.random_seed + self.rank * num_workers + worker_id)

        per_worker = int(math.ceil((self.end - self.start) / float(num_workers)))
        worker_start = self.start + worker_id * per_worker
        worker_end = min(worker_start + per_worker, self.end)
        sample_per_worker = (
            int(math.ceil(self.sample_n / float(num_workers))) if self.sample_n else None
        )
        i = 0
        while True:
            randint = rng.randint(worker_start, worker_end - 1)
            if sample_per_worker is not None and i > sample_per_worker:
                break
            yield self.__getitem__(randint)
            i += 1

    @staticmethod
    def normalize_rescale(ohlcv):
        # ohlcv = torch.log(ohlcv / (ohlcv[0] + 1e-12))
        # ohlcv: (seq_len, 4)
        open_price = ohlcv[0, 0]  # scalar
        ohlcv = ohlcv / (open_price + 1e-12) - 1
        ohlcv = torch.nan_to_num(ohlcv, nan=0.0, posinf=0.0, neginf=0.0)
        return ohlcv, open_price

    @staticmethod
    def unnormalize_rescale(ohlcv, denominator):
        # ohlcv: (seq_len, 4), denominator: scalar
        return (ohlcv + 1) * (denominator + 1e-12)

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

        timestamp_s_start = meta["open_timestamp_s_start"] + INTERVAL_TO_SECONDS[interval] * start

        ohlcv = torch.zeros((actual_length, 4), dtype=torch.float32)
        ohlcv[:actual_length, 0] = torch.tensor(arr[start:end, 0], dtype=torch.float32)
        ohlcv[:actual_length, 1] = torch.tensor(arr[start:end, 1], dtype=torch.float32)
        ohlcv[:actual_length, 2] = torch.tensor(arr[start:end, 2], dtype=torch.float32)
        ohlcv[:actual_length, 3] = torch.tensor(arr[start:end, 3], dtype=torch.float32)

        volume = torch.tensor(arr[start:end, 4], dtype=torch.float32)
        if self.normalize_rescale_price:
            ohlcv, denominator = self.normalize_rescale(ohlcv)
        else:
            denominator = 1
        if self.clip:
            ohlcv = torch.clamp(ohlcv, self.clip[0], self.clip[1])
        return {
            "symbol": symbol,
            "type": type_str,
            "interval": interval,
            "inputs": ohlcv,
            "bar_start": start,
            "bar_end": end,
            "timestamp_s_start": timestamp_s_start,
            "volume": volume,
            "index": index,
            "seq_len": actual_length,
            "denominator": denominator,
        }

    def collate_fn(self, batch, pad_value=0):
        # Separate inputs and targets
        inputs = [item["inputs"] for item in batch]

        # Get the maximum sequence length in the batch
        max_len = max(seq.size(0) for seq in inputs)

        # Pad sequences to max_len and create attention masks
        padded_inputs = []
        attention_masks = []
        for seq in inputs:
            seq_len = seq.size(0)
            # Create attention mask
            attention_mask = torch.ones(seq_len, dtype=torch.long, device=seq.device)
            if seq_len < max_len:
                padding = torch.full(
                    (max_len - seq_len, seq.size(1)),
                    pad_value,
                    dtype=seq.dtype,
                    device=seq.device,
                )
                padded_seq = torch.cat([seq, padding], dim=0)
                # Update attention mask
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.zeros(max_len - seq_len, dtype=torch.long, device=seq.device),
                    ],
                    dim=0,
                )
            else:
                padded_seq = seq[:max_len, :]
                attention_mask = attention_mask[:max_len]
            padded_inputs.append(padded_seq)
            attention_masks.append(attention_mask)

        # Stack the padded sequences and attention masks
        stacked_inputs = torch.stack(padded_inputs)
        stacked_attention_masks = torch.stack(attention_masks)

        batched_inputs = {
            "symbol": [item["symbol"] for item in batch],
            "seq_len": [item["seq_len"] for item in batch],
            "denominator": [item["denominator"] for item in batch],
            "type": [item["type"] for item in batch],
            "interval": [item["interval"] for item in batch],
            "timestamp_s_start": [item["timestamp_s_start"] for item in batch],
            "inputs": stacked_inputs,
            "attention_mask": stacked_attention_masks,
            "index": [item["index"] for item in batch],
        }

        return batched_inputs

    def plot_kline(self, index, output_file=""):
        item = self.__getitem__(index)
        self.plot_kline_with_prediction(item, output_file=output_file)

    def plot_kline_with_prediction(
        self, item, prediction: Optional[np.ndarray] = None, output_file="", obersevation_length=0
    ):
        symbol = item["symbol"]
        volume = item["volume"].numpy()
        interval = item["interval"]
        normalized_price = item["inputs"].numpy()  # (seq_len, 4)
        if prediction is not None:
            assert len(normalized_price) == len(prediction) + obersevation_length
            if prediction.ndim == 2:
                prediction = prediction[:, 3]  # close price
        data_as_list_of_dict = [
            {
                "timestamp_s": item["timestamp_s_start"] + i * INTERVAL_TO_SECONDS[interval],
                "open": normalized_price[i, 0],
                "high": normalized_price[i, 1],
                "low": normalized_price[i, 2],
                "close": normalized_price[i, 3],
                "volume": volume[i],
            }
            for i in range(normalized_price.shape[0])
        ]
        df = pd.DataFrame(data_as_list_of_dict)
        df["date"] = pd.to_datetime(df["timestamp_s"], unit="s")
        if prediction is not None:
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.to(torch.float32).numpy()
            prediction_as_list_of_dict = [
                {
                    "timestamp_s": item["timestamp_s_start"] + i * INTERVAL_TO_SECONDS[interval],
                    "predicted_price": prediction[i - obersevation_length],
                }
                for i in range(obersevation_length, normalized_price.shape[0])
            ]
            pred_df = pd.DataFrame(prediction_as_list_of_dict)
            pred_df["date"] = pd.to_datetime(pred_df["timestamp_s"], unit="s")
            plot_ohlc_candlestick_with_volume_and_prediction(
                df, pred_df, output_filename=output_file, symbol=symbol, interval=interval
            )
        else:
            plot_ohlc_candlestick_with_volume(
                df, output_filename=output_file, symbol=symbol, interval=interval
            )


def preview_size():
    # dataset = OHLCDatasetMmap(
    #         "memmap_dataset", window_range=(1600, 4096), is_train=True, filter_intervals='1h', filter_types='spot'
    #     )
    # print("len dataset ", len(dataset)) # 10599248

    # dataset = OHLCDatasetMmap(
    #         "memmap_dataset", window_range=(1600, 4096), is_train=True, filter_intervals='1h'
    #     )
    # print("len dataset ", len(dataset)) # 15253874
    pass


def sample_kiline():
    dataset = OHLCDatasetMmap(
        "memmap_dataset", window_range=(128, 512), is_train=True, filter_intervals="1h"
    )
    dataset.plot_kline(100, "test.html")


if __name__ == "__main__":
    # preview_size()
    sample_kiline()
