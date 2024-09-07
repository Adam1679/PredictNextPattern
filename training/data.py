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
        clip=(-10, 10),
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
        if "bar_pct_change" in batch[0]:
            batched_inputs["bar_pct_change"] = [item["bar_pct_change"] for item in batch]
        if "low_bigger_than_high_error_sum" in batch[0]:
            batched_inputs["low_bigger_than_high_error_sum"] = [
                item["low_bigger_than_high_error_sum"] for item in batch
            ]
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


class EnhancedOHLCDataset(OHLCDatasetMmap):
    BAR_HEIGHTS_BUCKETS = torch.tensor([0.2, 0.5, 1, 1.5, 3], dtype=torch.float32)  # 6 classes
    SHADOW_RATIO_BUCKETS = torch.tensor(
        [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95], dtype=torch.float32
    )  # 8 classes
    BODY_RATIO_BUCKETS = torch.tensor(
        [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95], dtype=torch.float32
    )  # 8 classes
    NORMALIZED_CLOSE_BUCKETS = torch.linspace(-50, 50, 4097)[1:-1]  # 4096 classes
    BOOLINGGER_BUCKETS = torch.tensor(
        [0, 0.25, 0.5, 0.75, 1], dtype=torch.float32
    )  # 0-, 0.25, 0.5, 0.75, 1+, 0- & 1+ means breakout. 6 classes
    WINDOW_SIZE = 20

    NAMES = [
        "close_price_buckets",
        "moving_up",
        "up_shadow_ratio_buckets",
        "body_ratio_buckets",
        "bar_height_atr_buckets",
        "higher_high",
        "lower_low",
        "bollinger_buckets",
        "breakout",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_atr(self, high, low, close):
        # TrueRange = max(high - low, abs(high - prev_close), abs(low - prev_close))
        # but because the prev_close == open, so we can simplify to
        # TrueRange = high - low
        ranges = high - low
        return ranges.mean()

    def __getitem__(self, index):
        item = super().__getitem__(index)
        ohlc = item.pop("inputs")  # Shape: (seq_len, 4)
        # print('ohlc', ohlc[:10])
        # 1. Boolean feature for moving up or down
        moving_up = torch.zeros(ohlc.shape[0], dtype=torch.bool)
        moving_up[1:] = ohlc[1:, 3] > ohlc[:-1, 3]  # Compare current close to previous close
        low_bigger_than_high_error_sum = torch.sum((ohlc[:, 1] < ohlc[:, 2]).float())
        bar_pct_change = torch.max((ohlc[:, 1] - ohlc[:, 2]))  # (high - low) / open
        # print('moving_up', moving_up[:10])
        # 2. Higher high or lower high
        higher_high = torch.zeros(ohlc.shape[0], dtype=torch.bool)
        higher_high[1:] = ohlc[1:, 1] > ohlc[:-1, 1]  # Compare current high to previous high
        # print('higher_high', higher_high[:10])
        # 3. Higher low or lower low
        lower_low = torch.zeros(ohlc.shape[0], dtype=torch.bool)
        lower_low[1:] = ohlc[1:, 2] < ohlc[:-1, 2]  # Compare current low to previous low
        # print('lower_low', lower_low[:10])
        # 4. Breakout of 20 rolling window high or low
        breakout = torch.zeros(ohlc.shape[0], dtype=torch.bool)
        window_high = torch.zeros(ohlc.shape[0], dtype=torch.float32)
        window_low = torch.zeros(ohlc.shape[0], dtype=torch.float32)
        bollinger_buckets = torch.zeros(ohlc.shape[0], dtype=torch.float32)
        for i in range(1, ohlc.shape[0]):
            if i > self.WINDOW_SIZE:
                window = ohlc[i - self.WINDOW_SIZE : i]
            else:
                window = ohlc[:i]
            window_high_ = torch.max(window[:, 1])
            window_low_ = torch.min(window[:, 2])
            close_price = ohlc[i, 3]
            breakout[i] = (close_price > window_high_) or (close_price < window_low_)
            window_high[i] = window_high_
            window_low[i] = window_low_
            ratio = (close_price - window_low_) / (window_high_ - window_low_)
            bollinger_buckets[i] = torch.bucketize(ratio, self.BOOLINGGER_BUCKETS, right=True)

        # print('breakout', breakout[:10])
        # print('window_high', window_high[:10])
        # print('window_low', window_low[:10])
        # print('bollinger_buckets', bollinger_buckets[:10])

        # 4.1 add a micro breakout feature
        # 5. Bar height normalized by ATR and bucketized
        atr = self.calculate_atr(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]) + 1e-6
        bar_height = abs(ohlc[:, 1] - ohlc[:, 2])
        bar_height_atr = bar_height / atr
        bar_height_atr_buckets = torch.bucketize(
            torch.clamp_(bar_height_atr, 0, 5), self.BAR_HEIGHTS_BUCKETS, right=True
        )
        # print('bar_height_atr_buckets', bar_height_atr_buckets[:10])
        # 6. get the shape of this bar
        up_shadow_ratio = (ohlc[:, 1] - torch.max(ohlc[:, 0], ohlc[:, 3])) / bar_height
        body_ratio = torch.abs(ohlc[:, 0] - ohlc[:, 3]) / bar_height
        up_shadow_ratio_buckets = torch.bucketize(
            torch.clamp_(
                up_shadow_ratio, self.SHADOW_RATIO_BUCKETS[0] - 1, self.SHADOW_RATIO_BUCKETS[-1] + 1
            ),
            self.SHADOW_RATIO_BUCKETS,
            right=True,
        )
        body_ratio_buckets = torch.bucketize(
            torch.clamp_(
                body_ratio, self.BODY_RATIO_BUCKETS[0] - 1, self.BODY_RATIO_BUCKETS[0] + 1
            ),
            self.BODY_RATIO_BUCKETS,
            right=True,
        )
        # print('up_shadow_ratio_buckets', up_shadow_ratio_buckets[:10])
        # print('body_ratio_buckets', body_ratio_buckets[:10])

        # 7. get the bucketized close price
        close_price_buckets = torch.bucketize(
            torch.clamp_(
                ohlc[:, 3] / ohlc[0, 3],
                self.NORMALIZED_CLOSE_BUCKETS[0] - 1,
                self.NORMALIZED_CLOSE_BUCKETS[-1] + 1,
            ),
            self.NORMALIZED_CLOSE_BUCKETS,
            right=True,
        )
        # print('close_price_buckets', close_price_buckets[:10])
        # Combine all features
        enhanced_features = torch.stack(
            [
                # 价格
                close_price_buckets,  # 4096
                # 方向
                moving_up.long(),  # 2
                # Signal Bar 强弱
                up_shadow_ratio_buckets,  # 8
                body_ratio_buckets,  # 8
                bar_height_atr_buckets,  # 6
                # 市场结构
                higher_high.long(),  # 2
                lower_low.long(),  # 2
                bollinger_buckets,  # 6
                breakout.long(),  # 2
            ],
            dim=1,
        ).long()
        item["inputs"] = enhanced_features  # (seq_len, 9)
        item["bar_pct_change"] = bar_pct_change
        item["low_bigger_than_high_error_sum"] = low_bigger_than_high_error_sum
        return item


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


def sample_one():
    dataset = EnhancedOHLCDataset(
        "memmap_dataset", window_range=(128, 512), is_train=True, filter_intervals="1h"
    )
    print(dataset[1000])


if __name__ == "__main__":
    # preview_size()
    torch.set_printoptions(precision=4)
    sample_one()
