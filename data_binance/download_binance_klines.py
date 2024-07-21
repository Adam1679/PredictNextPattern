import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import DAILY_INTERVALS, END_DATE, MAX_DAYS, START_DATE, datetime, query_ls_history

import pandas as pd
from utils.binance_util import (
    convert_to_date_object,
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


def download_monthly_klines(
    trading_type,
    symbols,
    num_symbols,
    intervals,
    years,
    months,
    start_date,
    end_date,
    folder,
    checksum,
):
    current = 0
    date_range = None

    if start_date and end_date:
        date_range = start_date + " " + end_date

    if not start_date:
        start_date = START_DATE
    else:
        start_date = convert_to_date_object(start_date)

    if not end_date:
        end_date = END_DATE
    else:
        end_date = convert_to_date_object(end_date)

    print("Found {} symbols".format(num_symbols))
    print("Folder = ", folder)
    for symbol in symbols:
        print(
            "[{}/{}] - start download monthly {} klines ".format(current + 1, num_symbols, symbol)
        )
        excecutors = ThreadPoolExecutor(32)
        for interval in intervals:
            for year in years:
                for month in months:
                    current_date = convert_to_date_object("{}-{}-01".format(year, month))
                    if current_date >= start_date and current_date <= end_date:
                        path = get_path(trading_type, "klines", "monthly", symbol, interval)
                        file_name = "{}-{}-{}-{}.zip".format(
                            symbol.upper(), interval, year, "{:02d}".format(month)
                        )
                        excecutors.submit(download_file, path, file_name, date_range, folder)

                        if checksum == 1:
                            checksum_path = get_path(
                                trading_type, "klines", "monthly", symbol, interval
                            )
                            checksum_file_name = "{}-{}-{}-{}.zip.CHECKSUM".format(
                                symbol.upper(), interval, year, "{:02d}".format(month)
                            )
                            excecutors.submit(
                                download_file,
                                checksum_path,
                                checksum_file_name,
                                date_range,
                                folder,
                            )
        excecutors.shutdown(wait=True)
        current += 1


def download_daily_klines(
    trading_type,
    symbols,
    num_symbols,
    intervals,
    dates,
    start_date,
    end_date,
    folder,
    checksum,
):
    current = 0
    date_range = None

    if start_date and end_date:
        date_range = start_date + " " + end_date

    if not start_date:
        start_date = START_DATE
    else:
        start_date = convert_to_date_object(start_date)

    if not end_date:
        end_date = END_DATE
    else:
        end_date = convert_to_date_object(end_date)

    # Get valid intervals for daily
    intervals = list(set(intervals) & set(DAILY_INTERVALS))
    print("Found {} symbols".format(num_symbols))
    excecutors = ThreadPoolExecutor(16)
    for symbol in symbols:
        print("[{}/{}] - start download daily {} klines ".format(current + 1, num_symbols, symbol))
        for interval in intervals:
            for date in dates:
                current_date = convert_to_date_object(date)
                if current_date >= start_date and current_date <= end_date:
                    path = get_path(trading_type, "klines", "daily", symbol, interval)
                    file_name = "{}-{}-{}.zip".format(symbol.upper(), interval, date)
                    # download_file(path, file_name, date_range, folder)
                    excecutors.submit(download_file, path, file_name, date_range, folder)
                    if checksum == 1:
                        checksum_path = get_path(trading_type, "klines", "daily", symbol, interval)
                        checksum_file_name = "{}-{}-{}.zip.CHECKSUM".format(
                            symbol.upper(), interval, date
                        )
                        # download_file(
                        #     checksum_path, checksum_file_name, date_range, folder
                        # )
                        excecutors.submit(
                            download_file,
                            checksum_path,
                            checksum_file_name,
                            date_range,
                            folder,
                        )

        current += 1


def download_5m_ls_data(symbols, folder):
    with ThreadPoolExecutor(16) as excecutors:
        for symbol in symbols:
            excecutors.submit(query_ls_history, symbol, folder)
            # query_ls_history(symbol, folder)


if __name__ == "__main__":
    parser = get_parser("klines")
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
        dates = pd.date_range(end=datetime.today(), periods=MAX_DAYS).to_pydatetime().tolist()
        dates = [date.strftime("%Y-%m-%d") for date in dates]
        download_monthly_klines(
            args.type,
            symbols,
            num_symbols,
            args.intervals,
            args.years,
            args.months,
            args.startDate,
            args.endDate,
            args.folder,
            args.checksum,
        )
        download_daily_klines(
            args.type,
            symbols,
            num_symbols,
            args.intervals,
            dates,
            args.startDate,
            args.endDate,
            args.folder,
            args.checksum,
        )
