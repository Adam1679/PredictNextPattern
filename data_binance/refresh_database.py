import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

root = Path(os.path.expanduser("~/binance_data"))
binance_data_root = Path(os.path.expanduser("~/binance_data/data"))


def delete_if_exist(path, file):
    fullpath = os.path.join(path, file)
    if os.path.exists(fullpath):
        os.remove(fullpath)


def removeEmptyFolders(root, removebinance_data_root=True):
    "Function to remove empty folders"
    if not os.path.isdir(root):
        return False
    delete_if_exist(root, ".DS_Store")
    # remove empty subfolders
    paths = os.listdir(root)
    for name in paths:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            removeEmptyFolders(path)

    if len(os.listdir(root)) == 0:
        os.rmdir(root)
        print(f"remove {root}")
        return True
    else:
        return False


def get_symbols(path):
    files = os.listdir(path)
    files = [f for f in files if f.endswith("USDT") or f.endswith("BUSD")]
    symbols = set()
    for file in files:
        symbols.add(file[:-4])
    return symbols


def download_futures(symbols, trades=True):
    f_s = " ".join(symbols)
    print(f"download data for {f_s}")
    if trades:
        cmd = f"python ./download_binance_trades.py -s {f_s} -folder ~/binance_data -t um -trades"
    else:
        cmd = f"python ./download_binance_klines.py -s {f_s} -folder ~/binance_data -t um -i 1m"
    print(cmd)
    if os.system(cmd) < 0:
        raise ValueError("download_futures failed")


def download_spots(symbols):
    f_s = " ".join(symbols)
    print(f"download data for {f_s}")
    cmd = f"python ./download_binance.py -s {f_s} -folder ~/binance_data -t spot -i 1m"
    print(cmd)
    if os.system(cmd) < 0:
        raise ValueError("download_spots failed")


def delete_duplicate_data(daily_binance_data_root, monthly_binance_data_root):
    print("delete duplicate data ...")
    dir_binance_data_root = monthly_binance_data_root
    daily_binance_data_root = daily_binance_data_root
    if not daily_binance_data_root.exists():
        return
    if not monthly_binance_data_root.exists():
        return
    all_spot_symbols = os.listdir(dir_binance_data_root)
    all_spot_symbols = [s for s in all_spot_symbols if "USDT" in s or "BUSD" in s]
    for symbol in all_spot_symbols:
        path = dir_binance_data_root.joinpath(symbol).joinpath("1m")
        all_files = os.listdir(path)
        max_dt = None
        for file in all_files:
            if file.endswith(".zip") or file.endswith(".csv"):
                dt = file.replace(f"{symbol}-1m-", "")
                dt = dt.replace(".zip", "")
                dt = dt.replace(".csv", "")
                dt = datetime.datetime.strptime(dt, "%Y-%m")
                if max_dt is None:
                    max_dt = dt
                else:
                    max_dt = max(dt, max_dt)
        if max_dt is None:
            continue
        if max_dt.month < 12:
            max_dt = max_dt.replace(month=max_dt.month + 1)
        else:
            max_dt = max_dt.replace(month=1, year=max_dt.year + 1)
        print(f"max_dt={max_dt} for {symbol}")
        all_dates = {max_dt}
        daily_dir = daily_binance_data_root.joinpath(symbol).joinpath("1m")
        if not daily_dir.exists():
            continue
        for file in os.listdir(daily_dir):
            if file.endswith(".csv") or file.endswith(".zip"):
                dt = file.replace(f"{symbol}-1m-", "")
                dt = dt.replace(".zip", "")
                dt = dt.replace(".csv", "")
                dt = datetime.datetime.fromisoformat(dt)
                if dt < max_dt:
                    os.remove(daily_dir.joinpath(file))
                    print(f"remove data {file}")
                else:
                    all_dates.add(dt)
        all_dates = sorted(list(all_dates))
        # 保证日期连续
        for i in range(len(all_dates) - 1):
            assert (all_dates[i + 1] - all_dates[i]).total_seconds() == 60 * 24 * 60, all_dates


def _fake_data(symbol, late_days, root):
    """
    1. 从BUSD合约数据中截取最近late_days天的数据
    2. 从USDT合约数据中截取最近late_days天的数据
    3. 从现货数据中截取最近late_days天的数据
    4. 合并数据
    5. 保存到fake数据文件中
    """
    print(f"processing {symbol} ... ")
    num_minutes = late_days * 24 * 60
    future_data_path = root / f"{symbol}BUSD-1m-perpetual.csv"
    usdt_future_data_path = root / f"{symbol}USDT-1m-perpetual.csv"
    spot_data_path = root / f"{symbol}USDT-1m.csv"

    if not future_data_path.exists():
        print(f"{symbol}没有BUSD合约")
        fake_data_path = root / f"{symbol}USDT-1m-fake.csv"
        if not usdt_future_data_path.exists():
            print(f"** {symbol}没有任何永续合约, 无法fake 数据 **")
            return
        else:
            future_data_path = None
    else:
        fake_data_path = root / f"{symbol}BUSD-1m-fake.csv"
    if fake_data_path.exists():
        if os.system(f"rm -f {fake_data_path.absolute()}") < 0:
            raise ValueError(f"rm -f {fake_data_path.absolute()}" + " failed")

    if future_data_path is not None:
        BUSD = pd.read_csv(future_data_path, header=None, index_col=0)
        BUSD = BUSD.sort_index()
        if len(BUSD) > num_minutes:
            BUSD = BUSD.iloc[num_minutes:]
        else:
            print(f"** {symbol} BUSD 期货数据不够 **")
        busd_start_time = BUSD.index.min()
        busd_end_time = BUSD.index.max()
        busd_early_time_dt = datetime.datetime.fromtimestamp(busd_start_time // 1_000)
        busd_late_time_dt = datetime.datetime.fromtimestamp(busd_end_time // 1_000)
    else:
        busd_late_time_dt, busd_early_time_dt = None, None

    USDT = pd.read_csv(usdt_future_data_path, header=None, index_col=0)
    USDT = USDT.sort_index()
    if len(USDT) > num_minutes:
        USDT = USDT.iloc[num_minutes:]
    else:
        print(f"** {symbol} USDT 期货数据不够 **")

    SPOT = pd.read_csv(spot_data_path, header=None, index_col=0)

    usdt_start_time = USDT.index.min()
    spot_time_start = SPOT.index.min()
    usdt_start_time_dt = datetime.datetime.fromtimestamp(usdt_start_time // 1_000)
    spot_time_start_dt = datetime.datetime.fromtimestamp(spot_time_start // 1_000)
    print(
        f"{symbol} SPOT: {spot_time_start_dt} to {usdt_start_time_dt}, FUTRES(USDT): {usdt_start_time_dt} to {busd_early_time_dt}, FUTURES(BUSD): {busd_early_time_dt} to {busd_late_time_dt}"
    )
    if future_data_path is not None:
        usdt_append = USDT.loc[USDT.index < busd_start_time]
        USDT = pd.concat([BUSD, usdt_append], axis=0)

    spot_append = SPOT.loc[SPOT.index < usdt_start_time]
    USDT = pd.concat([USDT, spot_append], axis=0)
    USDT = USDT.sort_index()

    USDT.to_csv(
        fake_data_path,
        header=False,
        index=True,
    )


def fake_all_data(all_spot_symbols, late_days, root):
    print(f"Fake data for {all_spot_symbols} with {late_days} late days")
    root = root / "csv_data"
    executors = ThreadPoolExecutor(max_workers=8)
    for symbol in all_spot_symbols:
        executors.submit(_fake_data, symbol, late_days, root)
        # _fake_data(symbol, late_days, root)
    executors.shutdown()


def replace_volume_for_futures_data(symbol):
    future_data_path = root / f"{symbol}BUSD-1m-perpetual.csv"
    usdt_future_data_path = root / f"{symbol}USDT-1m-perpetual.csv"
    usdt_fake_future_data_path = root / f"{symbol}USDT-1m-fake.csv"
    spot_data_path = root / f"{symbol}USDT-1m.csv"

    if not spot_data_path.exists():
        print(f"{symbol} has not spot data")
        return
    spot = pd.read_csv(spot_data_path, header=None, index_col=0)
    index_set = set(spot.index)

    def _replace_path(path):
        df = pd.read_csv(path, header=None, index_col=0)
        df_index = set(df.index)
        if not df_index.issubset(index_set):
            print("{path}'s index is fully covered.")
            return
        volume = spot.loc[df_index, 0]
        df.loc[df_index, 0] = volume
        df.to_csv(path, header=False, index=True)

    if usdt_future_data_path.exists():
        _replace_path(usdt_future_data_path)
    if future_data_path.exists():
        _replace_path(future_data_path)
    if usdt_fake_future_data_path.exists():
        _replace_path(usdt_fake_future_data_path)


if __name__ == "__main__":
    # ---- * ---- macro variables
    removeEmptyFolders(binance_data_root)
    spot_path1 = binance_data_root / "spot" / "monthly" / "klines"
    spot_path2 = binance_data_root / "spot" / "daily" / "klines"
    futures_path1 = binance_data_root / "futures" / "um" / "monthly" / "klines"
    futures_path2 = binance_data_root / "futures" / "um" / "daily" / "klines"

    # ---- * ---- for refresh current spot dataset
    symbols1 = get_symbols(futures_path1)  # pure symbols like BTC, ETH
    symbols2 = get_symbols(futures_path2)

    spot_symbols = symbols1 | symbols2
    all_spot_symbols = []
    for symbol in spot_symbols:
        all_spot_symbols.append(symbol + "USDT")
        all_spot_symbols.append(symbol + "BUSD")
    download_spots(all_spot_symbols)

    symbols1 = get_symbols(futures_path1)
    symbols2 = get_symbols(futures_path2)
    future_symbols = symbols1 | symbols2
    all_symbols = []
    for symbol in future_symbols:
        all_symbols.append(symbol + "USDT")
        all_symbols.append(symbol + "BUSD")
    download_futures(all_symbols)
    delete_duplicate_data(monthly_binance_data_root=spot_path1, daily_binance_data_root=spot_path2)
    delete_duplicate_data(
        monthly_binance_data_root=futures_path1, daily_binance_data_root=futures_path2
    )
    os.system("bash ./concat_all_zip_files.bash")
    # print("beging faking data...")
    # fake_all_data(spot_symbols, late_days=3, root=root)
    # print("beging faking data ... done")
    # print("beging replace volume data ...")
    # for symbol in future_symbols:
    #     replace_volume_for_futures_data(symbol)
    # print("beging replace volume data ... done")
    removeEmptyFolders(binance_data_root)

    # ---- * ---- for refresh current future dataset
    # removeEmptyFolders(binance_data_root)
    # future_symbols = S_DEFAULTS
    # download_futures(future_symbols, trades=False)
    # download_5m_ls_data(future_symbols, binance_data_root)
    # delete_duplicate_data(
    #     monthly_binance_data_root=futures_path1, daily_binance_data_root=futures_path2
    # )
    # os.system("bash ./utils/concat_all_zip_files.bash")
    # # print("beging faking data...")
    # # fake_all_data(future_symbols, late_days=3, root=root)
    # print("beging faking data ... done")
    # # print("beging replace volume data ...")
    # # for symbol in future_symbols:
    # # replace_volume_for_futures_data(symbol)
    # # print("beging replace volume data ... done")
    # removeEmptyFolders(binance_data_root)

    # ---- * ---- for download new spot dataset

    # removeEmptyFolders(binance_data_root)
    # spot_path1 = binance_data_root / "spot" / "monthly" / "klines"
    # spot_path2 = binance_data_root / "spot" / "daily" / "klines"
    # futures_path1 = binance_data_root / "futures" / "um" / "monthly" / "klines"
    # futures_path2 = binance_data_root / "futures" / "um" / "daily" / "klines"

    # all_spot_symbols = [
    #     "AAVEUSDT",
    #     "ADAUSDT",
    #     "ALGOUSDT",
    #     "ALICEUSDT",
    #     "APEUSDT",
    #     "APTUSDT",
    #     "ARUSDT",
    #     "ATOMUSDT",
    #     "AUDUSDT",
    #     "AVAXUSDT",
    #     "AXSUSDT",
    #     "BANDUSDT",
    #     "BCHUSDT",
    #     "BNBUSDT",
    #     "BNXUSDT",
    #     "BTCUSDT",
    #     "BURGERUSDT",
    #     "C98USDT",
    #     "CAKEUSDT",
    #     "CELOUSDT",
    #     "CHZUSDT",
    #     "CRVUSDT",
    #     "DASHUSDT",
    #     "DOGEUSDT",
    #     "DOTUSDT",
    #     "DYDXUSDT",
    #     "EGLDUSDT",
    #     "ENSUSDT",
    #     "EOSUSDT",
    #     "ETCUSDT",
    #     "ETHUSDT",
    #     "FILUSDT",
    #     "FLOWUSDT",
    #     "FTMUSDT",
    #     "GALAUSDT",
    #     "GALUSDT",
    #     "GBPUSDT",
    #     "GMTUSDT",
    #     "GMXUSDT",
    #     "GRTUSDT",
    #     "HFTUSDT",
    #     "ICPUSDT",
    #     "INJUSDT",
    #     "JASMYUSDT",
    #     "KAVAUSDT",
    #     "KLAYUSDT",
    #     "LAZIOUSDT",
    #     "LDOUSDT",
    #     "LINKUSDT",
    #     "LITUSDT",
    #     "LRCUSDT",
    #     "LTCUSDT",
    #     "LUNAUSDT",
    #     "LUNCUSDT",
    #     "MANAUSDT",
    #     "MASKUSDT",
    #     "MATICUSDT",
    #     "MDXUSDT",
    #     "NEARUSDT",
    #     "OPUSDT",
    #     "PEOPLEUSDT",
    #     "PORTOUSDT",
    #     "QNTUSDT",
    #     "REEFUSDT",
    #     "ROSEUSDT",
    #     "RUNEUSDT",
    #     "SANDUSDT",
    #     "SANTOSUSDT",
    #     "SFPUSDT",
    #     "SHIBUSDT",
    #     "SNXUSDT",
    #     "SOLUSDT",
    #     "SUSHIUSDT",
    #     "SXPUSDT",
    #     "THETAUSDT",
    #     "TRXUSDT",
    #     "TWTUSDT",
    #     "UNFIUSDT",
    #     "UNIUSDT",
    #     "VETUSDT",
    #     "VGXUSDT",
    #     "VIDTUSDT",
    #     "WAVESUSDT",
    #     "XLMUSDT",
    #     "XMRUSDT",
    #     "XRPUSDT",
    #     "YFIIUSDT",
    #     "YFIUSDT",
    #     "ZILUSDT",
    # ]
    # # download_spots(all_spot_symbols)

    # # symbols1 = get_symbols(futures_path1)
    # # symbols2 = get_symbols(futures_path2)
    # # future_symbols = symbols1 | symbols2
    # # all_symbols = []
    # # for symbol in future_symbols:
    # #     all_symbols.append(symbol + "USDT")
    # #     all_symbols.append(symbol + "BUSD")
    # # download_futures(all_symbols)
    # # for symbol in all_spot_symbols:
    # # query_ls_history(symbol, folder=binance_data_root)
    # removeEmptyFolders(binance_data_root)
    # delete_duplicate_data(
    #     monthly_binance_data_root=spot_path1, daily_binance_data_root=spot_path2
    # )
    # delete_duplicate_data(
    #     monthly_binance_data_root=futures_path1, daily_binance_data_root=futures_path2
    # )
    # os.system("bash ./concat_all_zip_files.bash")
