import json
import logging
import os
import time
from datetime import datetime

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

num_proc = os.cpu_count() - 4


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
        start_time = time.time()
        grouped = split_data.groupby(["symbol", "type", "interval"])
        time.time() - start_time
        # print("Takes {:.2f} seconds".format(group_time))
        for (symbol, type, interval), group in tqdm(
            grouped,
            desc="Processing data for {}".format(split_name),
        ):
            # Sort by timestamp
            group = group.sort_values("open_timestamp")

            # Check if all required columns exist
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
                    # print("No data for {}_{}_{}_{}. date range {} to {}".format(split_name, symbol, type, interval, group.open_timestamp.min(), group.open_timestamp.max()))
                    continue
                arr = np.array(data_selected.values, dtype=np.float32)  # (n_samples, n_features)
                subdir = "train" if is_train else "test"
                split_key = f"{split_name}_{symbol}_{type}_{interval}_{subdir}"
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
                    "open_timestamp_s_start": group["open_timestamp_s"].iloc[0],
                    "symbol": symbol,
                    "type": type,
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
    output_dir = "memmap_dataset"

    # Load data from Hugging Face and save to memmap
    load_and_save_to_memmap("adamzzzz/binance-klines-20240721", output_dir, split_date="2024-04-01")

    # Load data from memmap
    test(output_dir)


# Example usage
if __name__ == "__main__":
    main()
