import json
import os
import re
import shutil
import sys
import urllib.request
from argparse import ArgumentParser, ArgumentTypeError, RawTextHelpFormatter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytz
import requests

UTC_TZ = pytz.UTC
DATA_ROOT = os.path.expanduser("~/binance_data")


def generate_datetime(timestamp: float) -> datetime:
    """生成时间"""
    dt: datetime = datetime.fromtimestamp(timestamp / 1000, tz=UTC_TZ)
    return dt


@dataclass
class BinanceUmContractInfo:
    symbol: str
    status: str  # TRADING
    pricePrecision: int
    quantityPrecision: int
    baseAssetPrecision: int


# YEARS = ["2017", "2018", "2019", "2020", "2021", "2022", "2023"]
YEARS = ["2020", "2021", "2022", "2023", "2024"]
INTERVALS = [
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1mo",
]
DAILY_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
TRADING_TYPE = ["spot", "um", "cm"]
MONTHS = list(range(1, 13))
MAX_DAYS = 35
BASE_URL = "https://data.binance.vision/"
F_REST_HOST: str = "https://fapi.binance.com"
START_DATE = date(int(YEARS[0]), MONTHS[0], 1)
END_DATE = datetime.date(datetime.now())

UM_ASSET_INFO = {}
try:
    if Path("./utils/exchange_info_usdm_20230823.json").exists():
        with open("./utils/exchange_info_usdm_20230823.json", "r") as f:
            info = json.load(f)
    else:
        print("exchange_info_usdm_20230823.json not found")
        info = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo").json()
    for asset_info in info["symbols"]:
        asset_name = asset_info["symbol"]
        UM_ASSET_INFO[asset_info["symbol"]] = BinanceUmContractInfo(
            asset_name,
            asset_info["status"],
            asset_info["pricePrecision"],
            asset_info["quantityPrecision"],
            asset_info["baseAssetPrecision"],
        )
    print("[INFO] ASSET_INFO updated")
except Exception as e:
    print(f"[WARNING], exception in getting asset_info from binance {str(e)}")


def get_um_asset_info(symbol):
    if len(UM_ASSET_INFO) == 0:
        raise ValueError("asset_info not updated")
    return UM_ASSET_INFO[symbol]


def get_destination_dir(file_url, folder=None):
    store_directory = os.environ.get("STORE_DIRECTORY")
    if folder:
        store_directory = folder
    if not store_directory:
        store_directory = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(store_directory, file_url)


def get_download_url(file_url):
    return "{}{}".format(BASE_URL, file_url)


def symbol_get_most_recent_depth_data(symbol, type, depth):
    if type == "um":
        response = requests.get(
            f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&limit={depth}"
        ).json()
    elif type == "cm":
        response = requests.get(
            f"https://dapi.binance.com/dapi/v1/klines?symbol={symbol}&limit={depth}"
        ).json()
    else:
        response = requests.get(
            f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={depth}"
        ).json()
    if len(response["bids"]) == 0:
        raise ValueError(f"{symbol} got zero depth: {response}")
    d = {"symbol": symbol}
    d["bids_price"], d["bids_qty"] = zip(*response["bids"])
    d["asks_price"], d["asks_qty"] = zip(*response["asks"])
    d["bids_price"] = list(map(float, d["bids_price"]))
    d["bids_qty"] = list(map(float, d["bids_qty"]))
    d["asks_price"] = list(map(float, d["asks_price"]))
    d["asks_qty"] = list(map(float, d["asks_qty"]))
    return d


def symbol_recent_kline(symbol, type, limit, interval):
    if type == "um":
        response = requests.get(
            f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&limit={limit}&interval={interval}"
        ).json()
    elif type == "cm":
        response = requests.get(
            f"https://dapi.binance.com/dapi/v1/klines?symbol={symbol}&limit={limit}&interval={interval}"
        ).json()
    else:
        response = requests.get(
            f"https://api.binance.com/api/v3/klines?symbol={symbol}&limit={limit}&interval={interval}"
        ).json()
    rows = []
    for row in response:
        d = {
            "symbol": symbol,
            "volume": float(row[5]),
            "turnover": float(row[7]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "closed_datetime": datetime.fromtimestamp(float(row[6]) / 1000),
        }
        rows.append(d)
    return pd.DataFrame(rows)


def get_all_symbols(type):
    if type == "um":
        response = urllib.request.urlopen("https://fapi.binance.com/fapi/v1/exchangeInfo").read()
        symbols = list(map(lambda symbol: symbol["symbol"], json.loads(response)["symbols"]))
    elif type == "cm":
        response = urllib.request.urlopen("https://dapi.binance.com/dapi/v1/exchangeInfo").read()
        symbols = list(map(lambda symbol: symbol["symbol"], json.loads(response)["symbols"]))
    elif type == "spot":
        response = urllib.request.urlopen("https://api.binance.com/api/v3/exchangeInfo").read()
        symbols = list(map(lambda symbol: symbol["symbol"], json.loads(response)["symbols"]))
    elif type == "perpetual":
        symbols = []
        response = urllib.request.urlopen("https://fapi.binance.com/fapi/v1/exchangeInfo").read()
        for symbol_info in json.loads(response)["symbols"]:
            if symbol_info["contractType"] == "PERPETUAL" and symbol_info["status"] == "TRADING":
                symbols.append(symbol_info["symbol"])

    else:
        raise ValueError()
    return symbols


def download_file(base_path, file_name, date_range=None, folder=None):
    download_path = "{}{}".format(base_path, file_name)
    if date_range:
        date_range = date_range.replace(" ", "_")
        base_path = os.path.join(base_path, date_range)
    save_path = get_destination_dir(os.path.join(base_path, file_name), folder)

    if os.path.exists(save_path):
        print("\nfile already exists! {}".format(save_path))
        return
    # make the directory
    if not os.path.exists(get_destination_dir(base_path, folder)):
        Path(get_destination_dir(base_path, folder)).mkdir(parents=True, exist_ok=True)

    try:
        download_url = get_download_url(download_path)
        dl_file = urllib.request.urlopen(download_url, timeout=10)
        length = dl_file.getheader("content-length")
        if length:
            length = int(length)
            blocksize = max(4096, length // 100)
        tmp_save_path = save_path + ".unconfirmed"
        with open(tmp_save_path, "wb") as out_file:
            dl_progress = 0
            print("\nFile Download: {}".format(save_path))
            while True:
                buf = dl_file.read(blocksize)
                if not buf:
                    break
                dl_progress += len(buf)
                out_file.write(buf)
                done = int(50 * dl_progress / length)
                sys.stdout.write("\r[%s%s]" % ("#" * done, "." * (50 - done)))
                sys.stdout.flush()
        os.rename(tmp_save_path, save_path)

    except urllib.error.HTTPError:
        print("\nFile not found: {}".format(download_url))
    except Exception as e:
        print("\nError downloading file: {}".format(e))
    finally:
        tmp_save_path = save_path + ".unconfirmed"
        if os.path.exists(tmp_save_path):
            os.remove(tmp_save_path)


def convert_to_date_object(d):
    if isinstance(d, date):
        return d
    year, month, day = [int(x) for x in d.split("-")]
    date_obj = date(year, month, day)
    return date_obj


def get_start_end_date_objects(date_range):
    start, end = date_range.split()
    start_date = convert_to_date_object(start)
    end_date = convert_to_date_object(end)
    return start_date, end_date


def match_date_regex(arg_value, pat=re.compile(r"\d{4}-\d{2}-\d{2}")):
    if not pat.match(arg_value):
        raise ArgumentTypeError
    return arg_value


def check_directory(arg_value):
    if os.path.exists(arg_value):
        while True:
            option = input("Folder already exists! Do you want to overwrite it? y/n  ")
            if option != "y" and option != "n":
                print("Invalid Option!")
                continue
            elif option == "y":
                shutil.rmtree(arg_value)
                break
            else:
                break
    return arg_value


def get_path(trading_type, market_data_type, time_period, symbol, interval=None):
    trading_type_path = "data/spot"
    if trading_type != "spot":
        trading_type_path = f"data/futures/{trading_type}"
    if interval is not None:
        path = f"{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{interval}/"
    else:
        path = f"{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/"
    return path


def get_parser(parser_type):
    parser = ArgumentParser(
        description=("This is a script to download historical {} data").format(parser_type),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-s",
        dest="symbols",
        nargs="+",
        help="Single symbol or multiple symbols separated by space",
        default=None,
    )
    parser.add_argument(
        "-y",
        dest="years",
        default=YEARS,
        nargs="+",
        choices=YEARS,
        help="Single year or multiple years separated by space\n-y 2019 2021 means to download {} from 2019 and 2021".format(
            parser_type
        ),
    )
    parser.add_argument(
        "-m",
        dest="months",
        default=MONTHS,
        nargs="+",
        type=int,
        choices=MONTHS,
        help="Single month or multiple months separated by space\n-m 2 12 means to download {} from feb and dec".format(
            parser_type
        ),
    )
    parser.add_argument(
        "-d",
        dest="dates",
        nargs="+",
        type=match_date_regex,
        help="Date to download in [YYYY-MM-DD] format\nsingle date or multiple dates separated by space\ndownload past 35 days if no argument is parsed",
    )

    parser.add_argument(
        "-startDate",
        dest="startDate",
        type=match_date_regex,
        help="Starting date to download in [YYYY-MM-DD] format",
    )
    parser.add_argument(
        "-endDate",
        dest="endDate",
        type=match_date_regex,
        help="Ending date to download in [YYYY-MM-DD] format",
    )
    parser.add_argument(
        "-folder",
        dest="folder",
        type=check_directory,
        default=DATA_ROOT,
        help="Directory to store the downloaded data",
    )
    parser.add_argument(
        "-c",
        dest="checksum",
        default=0,
        type=int,
        choices=[0, 1],
        help="1 to download checksum file, default 0",
    )
    parser.add_argument(
        "-t",
        dest="type",
        default="um",
        choices=TRADING_TYPE,
        help="Valid trading types: {}".format(TRADING_TYPE),
    )

    if parser_type == "klines":
        parser.add_argument(
            "-i",
            dest="intervals",
            default=INTERVALS,
            nargs="+",
            choices=INTERVALS,
            help="single kline interval or multiple intervals separated by space\n-i 1m 1w means to download klines interval of 1minute and 1week",
        )
    return parser


if __name__ == "__main__":
    # print all symbols
    all_spot = get_all_symbols("spot")
    all_spot = [symbol for symbol in all_spot if symbol.endswith("USDT")]
    # print("len=", len(all_spot))
    # print(all_spot)
    pds = []
    # workers = ThreadPoolExecutor(max_workers=8)

    # for spot_symbol in tqdm(all_spot):
    #     # one_month_daily_turnover = workers.submit(symbol_recent_kline, spot_symbol, "spot", interval="1d", limit=30)
    #     try:
    #         one_month_daily_turnover = symbol_recent_kline(spot_symbol, "spot", interval="1d", limit=30)
    #         pds.append(one_month_daily_turnover)
    #     except Exception as e:
    #         print(f"{spot_symbol} failed. {e}")
    # # pds = [a.result() for a in pds]
    # # workers.shutdown()
    # final_df = pd.concat(pds, axis=0)
    # final_df.to_csv("./final_csv.csv")

    # get all spot symbol with avg most recent turnover > 1M
    # for spot_symbol in tqdm(all_spot):
    #     # one_month_daily_turnover = workers.submit(symbol_recent_kline, spot_symbol, "spot", interval="1d", limit=30)
    #     try:
    #         depth = symbol_get_most_recent_depth_data(spot_symbol, "spot", depth=5)
    #         pds.append(depth)
    #     except Exception as e:
    #         print(f"{spot_symbol} failed. {e}")
    #     # pds = [a.result() for a in pds]
    #     # workers.shutdown()
    # final_df = pd.DataFrame(pds)
    # final_df.to_csv(
    #     "/Users/bytedance/MLTrader-deploy/research/data_binance/final_depth_csv.csv"
    # )

    # get futures
    symbols = get_all_symbols("um")
    print("#usdt symbols=", len(symbols))
    usdt_symbols = [s for s in symbols if s.endswith("USDT")]
    busd_symbols = [s for s in symbols if s.endswith("BUSD")]
    print("#usdt symbols=", len(usdt_symbols))
    print(usdt_symbols)
    print("#busd symbols=", len(busd_symbols))
    print(busd_symbols)
