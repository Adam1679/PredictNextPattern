import pandas as pd

from training.data import OHLCDatasetMmap
from training.plotting import plot_ohlc_candlestick_with_volume


def visualize():
    dataset = OHLCDatasetMmap(
        "memmap_dataset",
        window_range=(400, 400),
        is_train=True,
        filter_intervals="1h",
        filter_symbols=["BTCUSDT", "ETHUSDT"],
        filter_types="um",
        normalize_price=False,
        clip=None,
    )
    for idx in [
        100,
    ]:
        _data = dataset[idx]
        symbol = _data["symbol"]
        interval = _data["interval"]
        type_str = _data["type"]
        close_price = _data["inputs"][:, 0]
        high_price = _data["inputs"][:, 1]
        low_price = _data["inputs"][:, 2]
        open_price = _data["inputs"][:, 3]
        # volume = _data['inputs'][:, 4]
        df = pd.DataFrame(
            {
                "timestam_s": _data["timestamp_s"],
                "close": close_price,
                "high": high_price,
                "low": low_price,
                "open": open_price,
                # 'volume': volume
            }
        )
        df["date"] = pd.to_datetime(df["timestam_s"], unit="s")
        file_name = f"kline_{symbol}_{interval}_{type_str}.html"
        plot_ohlc_candlestick_with_volume(df, output_filename=file_name)


if __name__ == "__main__":
    visualize()
