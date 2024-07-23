import os
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm
from utils.binance_util import INTERVALS, TRADING_TYPE

INTERVAL_TO_TIMEDELT = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
}


@dataclass
class BinanceKlines:
    open_timestamp: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    close_timestamp: int
    quote_asset_volume: float
    number_of_trade: int
    tbbav: float
    tbqav: float
    ignore: str
    symbol: str
    type: str
    interval: str
    missing: bool


def _add_meta(data, symbol, type, interval):
    data["symbol"] = symbol
    data["type"] = type
    data["interval"] = interval
    data["missing"] = False  # Add missing flag, initially set to False for all original data
    return BinanceKlines(**data)


def get_zip_files(root_dir: str) -> List[str]:
    """
    Get all ZIP files in a directory and its subdirectories.

    :param root_dir: Root directory to search
    :return: List of paths to ZIP files
    """
    zip_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".zip"):
                zip_files.append(os.path.join(root, file))
    return zip_files


def uncompress_zip(zip_file_path: str) -> bytes:
    """
    Uncompresses a ZIP file and returns a dictionary containing the raw content of each file.

    :param zip_file_path: Path to the ZIP file
    :return: Dictionary with filenames as keys and their raw content as values
    """
    content_bytes = None
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            with zip_ref.open(file_info) as file:
                content_bytes = file.read()
    return content_bytes


def parse_csv_content(content: bytes) -> List[Dict[str, Any]]:
    """
    Parses the CSV content from a byte string without using csv.reader.

    :param content: Byte string containing CSV data
    :return: List of dictionaries, each representing a row in the CSV
    """
    headers = [
        "open_timestamp",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
        "close_timestamp",
        "quote_asset_volume",
        "number_of_trade",
        "tbbav",
        "tbqav",
        "ignore",
    ]

    parsed_data = []
    lines = content.decode("utf-8").splitlines()

    for line in lines:
        values = line.split(",")
        if values[0] == "open_time":
            continue
        if len(values) != len(headers):
            continue  # Skip rows that don't match the expected format

        parsed_row = {}
        for header, value in zip(headers, values):
            if header in ["open_timestamp", "close_timestamp"]:
                parsed_row[header] = datetime.fromtimestamp(int(value) / 1000)
            elif header in [
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume",
                "quote_asset_volume",
                "tbbav",
                "tbqav",
            ]:
                parsed_row[header] = float(value)
            elif header == "number_of_trade":
                parsed_row[header] = int(value)
            else:
                parsed_row[header] = value

        parsed_data.append(parsed_row)

    return parsed_data


def get_meta_from_type(path_str):
    # e.g /Users/adam/binance_data/data/futures/um/daily/klines/BCHUSDT/1d/BCHUSDT-1d-2024-06-18.zip
    # e.g binance_data/data/spot/daily/klines/ATAUSDT/1h/ATAUSDT-1h-2024-07-08.zip
    path = path_str.split("/")
    if path[-1].endswith(".zip"):
        interval = path[-2].lower()
        symbol = path[-3].upper()
        type = path[-6].lower()
        assert type in TRADING_TYPE, f"Unknown trading type: {type}"
        assert interval in INTERVALS, f"Unknown interval: {interval}"
        return symbol, type, interval
    else:
        raise ValueError(f"Invalid path: {path_str}")


def process_zip_files(zip_files: List[str]) -> Dataset:
    """
    Process a list of ZIP files, fill missing data using forward fill, add a 'missing' flag,
    and return a Hugging Face Dataset.

    :param zip_files: List of paths to ZIP files
    :return: Hugging Face Dataset
    """
    all_data = []

    with ThreadPoolExecutor(16) as workers:

        def _worker(zip_file):
            raw_content = uncompress_zip(zip_file)
            parsed_data = parse_csv_content(raw_content)
            symbol, type, interval = get_meta_from_type(zip_file)
            parsed_data = list(map(lambda x: _add_meta(x, symbol, type, interval), parsed_data))
            return parsed_data

        futures = [workers.submit(_worker, zip_file) for zip_file in zip_files]
        for future in tqdm(futures, total=len(futures), desc="Processing ZIP files"):
            parsed_data = future.result()
            all_data.extend(parsed_data)

    df = pd.DataFrame(all_data)
    df = df.sort_values(by=["symbol", "type", "interval"])
    df = df.drop_duplicates(subset=["symbol", "type", "interval", "open_timestamp"])

    # Group by symbol, type, and interval
    grouped = df.groupby(["symbol", "type", "interval"])
    n_samples = len(df["symbol"].unique())
    n_type = len(df["type"].unique())
    n_interval = len(df["interval"].unique())
    filled_data = []
    for (symbol, type, interval), group in tqdm(
        grouped,
        total=n_samples * n_type * n_interval,
        desc="Sorting, Reindexing, and Forward Filling...",
    ):
        # Sort by timestamp
        group = group.sort_values("open_timestamp")

        # Create a complete date range
        start_time = group["open_timestamp"].min()
        end_time = group["open_timestamp"].max()
        complete_range = pd.date_range(
            start=start_time, end=end_time, freq=INTERVAL_TO_TIMEDELT[interval]
        )

        # Reindex the group with the complete range and forward fill
        filled_group = group.set_index("open_timestamp").reindex(complete_range)
        # Mark missing data
        filled_group["missing"] = filled_group["missing"].isna()

        # Forward fill
        filled_group = filled_group.ffill()

        # Reset index and add back the metadata columns
        filled_group = filled_group.reset_index()
        filled_group["open_timestamp"] = filled_group["index"]
        filled_group["symbol"] = symbol
        filled_group["type"] = type
        filled_group["interval"] = interval

        filled_data.append(filled_group)

    # Combine all filled data
    final_df = pd.concat(filled_data, ignore_index=True)

    # Remove duplicates and sort
    final_df = final_df.drop_duplicates(subset=["open_timestamp", "symbol", "type", "interval"])
    final_df = final_df.sort_values(by=["symbol", "type", "interval", "open_timestamp"])
    # if "__index_level_0__" in final_df.columns:
    # final_df = final_df.drop(columns=['__index_level_0__'])
    print("Total rows:")
    print(final_df.groupby(["symbol", "type", "interval"])["open_timestamp"].count())
    print("Missing data rows:")
    print(final_df.groupby(["symbol", "type", "interval"])["missing"].sum())
    return Dataset.from_pandas(final_df)


def _worker(local_zip_files, symbol, type, interval):
    all_data = []
    for zip_file in local_zip_files:
        raw_content = uncompress_zip(zip_file)
        parsed_data = parse_csv_content(raw_content)
        parsed_data = list(map(lambda x: _add_meta(x, symbol, type, interval), parsed_data))
        all_data.extend(parsed_data)
    group = pd.DataFrame(all_data)
    group = group.sort_values(by=["symbol", "type", "interval"])
    group = group.drop_duplicates(subset=["symbol", "type", "interval", "open_timestamp"])
    # Create a complete date range
    start_time = group["open_timestamp"].min()
    end_time = group["open_timestamp"].max()
    complete_range = pd.date_range(
        start=start_time, end=end_time, freq=INTERVAL_TO_TIMEDELT[interval]
    )

    # Reindex the group with the complete range and forward fill
    filled_group = group.set_index("open_timestamp").reindex(complete_range)
    # Mark missing data
    filled_group["missing"] = filled_group["missing"].isna()

    # Forward fill
    filled_group = filled_group.ffill()

    # Reset index and add back the metadata columns
    filled_group = filled_group.reset_index()
    filled_group["open_timestamp"] = filled_group["index"]
    filled_group["symbol"] = symbol
    filled_group["type"] = type
    filled_group["interval"] = interval
    return filled_group


def process_zip_files_grouped(zip_files: List[str]) -> Dataset:
    """
    Process a list of ZIP files, fill missing data using forward fill, add a 'missing' flag,
    and return a Hugging Face Dataset.

    :param zip_files: List of paths to ZIP files
    :return: Hugging Face Dataset
    """
    zip_files_grouped = defaultdict(list)
    for zip_file in zip_files:
        symbol, type, interval = get_meta_from_type(zip_file)
        zip_files_grouped[(symbol, type, interval)].append(zip_file)
    print(f"Found {len(zip_files_grouped)} groups of ZIP files")
    with ProcessPoolExecutor(32) as workers:
        futures = []
        all_df_data = []
        for (symbol, type_str, interval), group_zip_files in zip_files_grouped.items():
            futures.append(workers.submit(_worker, group_zip_files, symbol, type_str, interval))

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing grouped ZIP files"
        ):
            parsed_data = future.result()
            all_df_data.append(parsed_data)

    final_df = pd.concat(all_df_data, axis=0, ignore_index=True)

    # Remove duplicates and sort
    final_df = final_df.drop_duplicates(subset=["open_timestamp", "symbol", "type", "interval"])
    final_df = final_df.sort_values(by=["symbol", "type", "interval", "open_timestamp"])

    print("Total rows:")
    print(final_df.groupby(["symbol", "type", "interval"])["open_timestamp"].count())
    print("Missing data rows:")
    print(final_df.groupby(["symbol", "type", "interval"])["missing"].sum())
    return Dataset.from_pandas(final_df)


def upload_to_huggingface(dataset: Dataset, repo_name: str, type_str: str, interval: str):
    """
    Upload a dataset to Hugging Face.

    :param dataset: Hugging Face Dataset
    :param repo_name: Name of the repository to create/use on Hugging Face
    :param token: Hugging Face API token
    """
    # Create the repository if it doesn't exist
    HfApi()
    create_repo(
        repo_name, repo_type="dataset", exist_ok=True, token="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"
    )
    # for type_str, interval in product(types, intervals):
    #     # Filter dataset
    #     filtered_dataset = dataset.filter(
    #         lambda x: x["type"] == type_str and x["interval"] == interval
    #     )
    #     # Push the dataset to the Hugging Face Hub
    dataset = dataset.remove_columns(["__index_level_0__"])
    dataset.push_to_hub(
        repo_name, split=f"{type_str}.{interval}", token="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"
    )


def main(root_dir: str, repo_name: str, type_str, interval, symbols=[]):
    """
    Main function to run the entire pipeline.

    :param root_dir: Root directory containing ZIP files
    :param repo_name: Name of the repository to create/use on Hugging Face
    """
    # Get all ZIP files
    zip_files = get_zip_files(root_dir)
    if symbols:
        zip_files = [f for f in zip_files if any(symbol in f for symbol in symbols)]
    _type_str = "/" + type_str.lower() + "/"
    _interval_str = "/" + interval.lower() + "/"
    zip_files = [f for f in zip_files if _type_str in f and _interval_str in f]
    if len(zip_files) == 0:
        print("No ZIP files found")
        return
    print(f"Found {len(zip_files)} ZIP files")

    # Process ZIP files and create dataset
    # dataset = process_zip_files(zip_files)
    dataset = process_zip_files_grouped(zip_files)
    print(f"Created dataset with {len(dataset)} rows")

    # Upload to Hugging Face
    upload_to_huggingface(dataset, repo_name, type_str, interval)
    print(f"Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{repo_name}")


if __name__ == "__main__":
    # repo_name = "adamzzzz/binance-1m-klines-20240721"
    repo_name = "adamzzzz/binance-klines-20240721"
    # main(os.path.expanduser("~/binance_data/data"), repo_name, type_str='spot', interval='1h')
    # main(os.path.expanduser("~/binance_data/data"), repo_name, type_str='spot', interval='30m')
    # main(os.path.expanduser("~/binance_data/data"), repo_name, type_str='spot', interval='15m')
    # main(os.path.expanduser("~/binance_data/data"), repo_name, type_str="spot", interval="5m")
    # main(os.path.expanduser("~/binance_data/data"), repo_name, type_str="spot", interval="1m")

    # main(os.path.expanduser("~/binance_data/data"), repo_name, type_str="um", interval="1h")
    # main(os.path.expanduser("~/binance_data/data"), repo_name, type_str="um", interval="30m")
    # main(os.path.expanduser("~/binance_data/data"), repo_name, type_str="um", interval="15m")
    # main(os.path.expanduser("~/binance_data/data"), repo_name, type_str="um", interval="5m")
    main(os.path.expanduser("~/binance_data/data"), repo_name, type_str="um", interval="1m")
