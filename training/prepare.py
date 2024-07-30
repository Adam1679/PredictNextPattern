import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

num_proc = os.cpu_count() - 4

INTERVAL_TO_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

INTERVAL_TO_TIMEDELT = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
}


def load_and_save_to_memmap(
    dataset_name,
    output_dir,
    split_date="",
    columns=["open_price", "high_price", "low_price", "close_price", "volume"],
):
    # Load the dataset
    all_dataset = load_dataset(dataset_name, keep_in_memory=True)
    train_split_timestamp_s = datetime.strptime(split_date, "%Y-%m-%d").timestamp()
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)

    metadata = {
        "columns": columns,
        "dataset_name": dataset_name,
        "split_date": split_date,
        "output_dir": output_dir,
        "create_time": time.time(),
        "splits": {},
    }

    # Process each is_train
    for split_name in all_dataset.keys():
        split_data = all_dataset[split_name].to_pandas()
        split_data["open_timestamp_s"] = split_data["open_timestamp"].astype(int) / 1e9
        print("Grouping data... for {}".format(split_name), end=" ")
        split_data = split_data.sort_values(["symbol", "type", "interval", "open_timestamp"])
        grouped = split_data.groupby(["symbol", "type", "interval"])
        for (symbol, type_str, interval), group in tqdm(
            grouped,
            desc="Processing data for {}".format(split_name),
        ):
            # Sort by timestamp
            group = group.sort_values("open_timestamp")
            # # Create a complete date range
            # start_time = group["open_timestamp"].min()
            # end_time = group["open_timestamp"].max()
            # complete_range = pd.date_range(
            #     start=start_time, end=end_time, freq=INTERVAL_TO_TIMEDELT[interval]
            # )

            # # Reindex the group with the complete range and forward fill
            # filled_group = group.set_index("open_timestamp").reindex(complete_range)
            # # Mark missing data
            # filled_group["missing"] = filled_group["missing"].isna()
            # group = group.ffill().reset_index()

            # Check if all required columns exist
            min_timestamp = group["open_timestamp_s"].iloc[0]
            max_timestamp = group["open_timestamp_s"].iloc[-1]

            num_interval = int((max_timestamp - min_timestamp) // INTERVAL_TO_SECONDS[interval] + 1)
            assert (
                num_interval == group.shape[0]
            ), f"num_interval: {num_interval}, arr.shape[0]: {group.shape[0]}"
            for is_train in [True, False]:

                if is_train:
                    data_selected = group.loc[
                        group.open_timestamp_s <= train_split_timestamp_s, columns
                    ]
                else:
                    data_selected = group.loc[
                        group.open_timestamp_s > train_split_timestamp_s, columns
                    ]
                if data_selected.shape[0] == 0:
                    # print("No data for {}_{}_{}_{}. date range {} to {}".format(split_name, symbol, type_str, interval, group.open_timestamp.min(), group.open_timestamp.max()))
                    continue
                arr = np.array(data_selected.values, dtype=np.float32)  # (n_samples, n_features)
                subdir = "train" if is_train else "test"
                split_key = f"{split_name}_{symbol}_{type_str}_{interval}_{subdir}"
                file_name = f"{split_key}.dat"

                memmap_path = os.path.join(
                    output_dir, "train" if is_train else "test", f"{split_key}.dat"
                )
                memmap_array = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=arr.shape)

                # Write data to memmap file
                memmap_array[:] = arr[:]
                memmap_array.flush()

                metadata["splits"][split_key] = {
                    "filename": file_name,
                    "shape": arr.shape,
                    "dtype": str(np.float32),
                    "subdir": subdir,
                    "open_timestamp_s_start": int(group["open_timestamp_s"].iloc[0]),
                    "open_timestamp_s_end": int(group["open_timestamp_s"].iloc[-1]),
                    "symbol": symbol,
                    "type": type_str,
                    "interval": interval,
                }
                del arr
                del memmap_array
        del grouped
        del split_data

    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def test(input_dir):
    # Load metadata
    # test precision
    with open(os.path.join(input_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    loaded_data = {}

    for split_name, info in metadata["splits"].items():
        memmap_path = os.path.join(input_dir, info["subdir"], info["filename"])
        if not os.path.exists(memmap_path):
            logging.error(f"{memmap_path} for {split_name} not found")
            continue
        size = tuple(info["shape"])[0]
        data = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=tuple(info["shape"]))
        if np.any(data[:, :4] <= 0):
            op_zero = len(np.where(data[:, 0] <= 0))
            hi_zero = len(np.where(data[:, 1] <= 0))
            lo_zero = len(np.where(data[:, 2] <= 0))
            cl_zero = len(np.where(data[:, 3] <= 0))
            print(
                f"Data contains non-positve values in is_train '{split_name}', open: {op_zero}, high: {hi_zero}, low: {lo_zero}, close: {cl_zero}"
            )
        try:
            data[size - 1, 0]
        except IndexError:
            logging.error(f"IndexError: {split_name} {size}")
            assert False

    return loaded_data


def main():
    output_dir = "memmap_dataset2"

    # Load data from Hugging Face and save to memmap
    load_and_save_to_memmap("adamzzzz/binance-klines-20240721", output_dir, split_date="2024-04-01")

    # Load data from memmap
    test(output_dir)


# Example usage
if __name__ == "__main__":
    main()
