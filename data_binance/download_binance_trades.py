import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import END_DATE, START_DATE

import pandas as pd
from utils.binance_util import (
    convert_to_date_object,
    datetime,
    download_file,
    get_parser,
    get_path,
    json,
    urllib,
)


# python download_binance.py -s ETHUSDT XRPUSDT LTCUSDT EOSUSDT ETCUSDT -startDate 2022-02-01 -folder .vscode -i 1m
def get_all_symbols(type):
    if type == "um":
        response = urllib.request.urlopen("https://fapi.binance.com/fapi/v1/exchangeInfo").read()
    elif type == "cm":
        response = urllib.request.urlopen("https://dapi.binance.com/dapi/v1/exchangeInfo").read()
    else:
        response = urllib.request.urlopen("https://api.binance.com/api/v3/exchangeInfo").read()
    return list(map(lambda symbol: symbol["symbol"], json.loads(response)["symbols"]))


# def download_monthly_aggTrades(
#     trading_type,
#     symbols,
#     num_symbols,
#     years,
#     months,
#     start_date,
#     end_date,
#     folder,
#     checksum,
# ):
#     current = 0
#     date_range = None

#     if start_date and end_date:
#         date_range = start_date + " " + end_date

#     if not start_date:
#         start_date = START_DATE
#     else:
#         start_date = convert_to_date_object(start_date)

#     if not end_date:
#         end_date = END_DATE
#     else:
#         end_date = convert_to_date_object(end_date)

#     print("Found {} symbols".format(num_symbols))
#     print("Folder = ", folder)
#     excecutors = ThreadPoolExecutor(16)
#     for symbol in symbols:
#         print(
#             "[{}/{}] - start download monthly {} klines ".format(
#                 current + 1, num_symbols, symbol
#             )
#         )
#         for year in years:
#             for month in months:
#                 current_date = convert_to_date_object("{}-{}-01".format(year, month))
#                 if current_date >= start_date and current_date <= end_date:
#                     path = get_path(trading_type, "aggTrades", "monthly", symbol)
#                     file_name = "{}-{}-{}-{}.zip".format(
#                         symbol.upper(), "aggTrades", year, "{:02d}".format(month)
#                     )
#                     excecutors.submit(
#                         download_file, path, file_name, date_range, folder
#                     )
#                     # download_file(path, file_name, date_range, folder)

#                     if checksum == 1:
#                         checksum_path = get_path(
#                             trading_type, "aggTrades", "monthly", symbol
#                         )
#                         checksum_file_name = "{}-{}-{}-{}.zip.CHECKSUM".format(
#                             symbol.upper(), "aggTrades", year, "{:02d}".format(month)
#                         )
#                         # download_file(
#                         #     checksum_path, checksum_file_name, date_range, folder
#                         # )
#                         excecutors.submit(
#                             download_file,
#                             checksum_path,
#                             checksum_file_name,
#                             date_range,
#                             folder,
#                         )

#         current += 1
#     excecutors.shutdown(wait=True)


def download_daily_aggTrades(
    trading_type,
    symbols,
    num_symbols,
    dates,
    start_date,
    end_date,
    folder,
    checksum,
):
    current = 0
    date_range = None

    if not start_date:
        start_date = START_DATE
    else:
        start_date = convert_to_date_object(start_date)

    if not end_date:
        end_date = END_DATE
    else:
        end_date = convert_to_date_object(end_date)

    # Get valid intervals for daily
    print("Found {} symbols".format(num_symbols))
    excecutors = ThreadPoolExecutor(16)
    for symbol in symbols:
        print(
            "[{}/{}] - start download daily {} aggTrades ".format(current + 1, num_symbols, symbol)
        )
        for date in dates:
            current_date = convert_to_date_object(date)
            if current_date >= start_date and current_date <= end_date:
                path = get_path(trading_type, "aggTrades", "daily", symbol)
                file_name = "{}-aggTrades-{}.zip".format(symbol.upper(), date)
                # download_file(path, file_name, date_range, folder)
                excecutors.submit(download_file, path, file_name, date_range, folder)
                if checksum == 1:
                    checksum_path = get_path(trading_type, "aggTrades", "daily", symbol)
                    checksum_file_name = "{}-aggTrades-{}.zip.CHECKSUM".format(symbol.upper(), date)
                    excecutors.submit(
                        download_file,
                        checksum_path,
                        checksum_file_name,
                        date_range,
                        folder,
                    )

        current += 1
    excecutors.shutdown(wait=True)


def download_daily_trades(
    trading_type,
    symbols,
    num_symbols,
    dates,
    start_date,
    end_date,
    folder,
    checksum,
):
    current = 0
    date_range = None

    if not start_date:
        start_date = START_DATE
    else:
        start_date = convert_to_date_object(start_date)

    if not end_date:
        end_date = END_DATE
    else:
        end_date = convert_to_date_object(end_date)

    # Get valid intervals for daily
    print("Found {} symbols".format(num_symbols))
    excecutors = ThreadPoolExecutor(16)
    for symbol in symbols:
        print("[{}/{}] - start download daily {} trades ".format(current + 1, num_symbols, symbol))
        for date in dates:
            current_date = convert_to_date_object(date)
            if current_date >= start_date and current_date <= end_date:
                path = get_path(trading_type, "trades", "daily", symbol)
                file_name = "{}-trades-{}.zip".format(symbol.upper(), date)
                # download_file(path, file_name, date_range, folder)
                excecutors.submit(download_file, path, file_name, date_range, folder)
                if checksum == 1:
                    checksum_path = get_path(trading_type, "trades", "daily", symbol)
                    checksum_file_name = "{}-trades-{}.zip.CHECKSUM".format(symbol.upper(), date)
                    excecutors.submit(
                        download_file,
                        checksum_path,
                        checksum_file_name,
                        date_range,
                        folder,
                    )

        current += 1
    excecutors.shutdown(wait=True)


if __name__ == "__main__":
    parser = get_parser("trades")
    parser.add_argument(
        "-ld",
        dest="latest_n_days",
        help="lates N days",
        default="-1",
        type=int,
    )
    parser.add_argument(
        "--trades",
        dest="trades",
        help="",
        action="store_true",
    )  # aggTrades 数据源有问题，暂时不用
    args = parser.parse_args(sys.argv[1:])

    if not args.symbols:

        symbols = get_all_symbols(args.type)
        symbols = [symbol for symbol in symbols if symbol.upper().endswith("USDT")]
        print(
            "fetching all USDT symbols from exchange. #={}, usdt_symbols={}".format(
                len(symbols), symbols
            )
        )
        num_symbols = len(symbols)
    else:
        symbols = args.symbols
        num_symbols = len(symbols)

    if args.dates:
        dates = args.dates
    else:
        dates = (
            pd.date_range(
                end=datetime.today(),
                periods=args.latest_n_days if args.latest_n_days > 0 else 500,
            )
            .to_pydatetime()
            .tolist()
        )
        dates = [date.strftime("%Y-%m-%d") for date in dates]
        if args.trades:
            download_daily_trades(
                args.type,
                symbols,
                num_symbols,
                dates,
                args.startDate,
                args.endDate,
                args.folder,
                args.checksum,
            )
        else:
            download_daily_aggTrades(
                args.type,
                symbols,
                num_symbols,
                dates,
                args.startDate,
                args.endDate,
                args.folder,
                args.checksum,
            )
