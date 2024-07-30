import unittest

from training.data import OHLCDatasetMmap
from training.plotting import (
    plot_ohlc_candlestick_with_volume,
    plot_ohlc_candlestick_with_volume_and_prediction,
)


class TestPrepare(unittest.TestCase):
    def test_plot_run_success1(self):
        sample_data = [
            {"date": "2023-01-01", "open": 100, "high": 110, "low": 95, "close": 105, "volume": 10},
            {
                "date": "2023-01-02",
                "open": 105,
                "high": 115,
                "low": 100,
                "close": 110,
                "volume": 10,
            },
            {
                "date": "2023-01-03",
                "open": 110,
                "high": 120,
                "low": 105,
                "close": 115,
                "volume": 10,
            },
            {
                "date": "2023-01-04",
                "open": 115,
                "high": 125,
                "low": 110,
                "close": 120,
                "volume": 10,
            },
            {
                "date": "2023-01-05",
                "open": 120,
                "high": 130,
                "low": 115,
                "close": 125,
                "volume": 10,
            },
            {"date": "2023-01-06", "open": 120, "high": 130, "low": 76, "close": 90, "volume": 10},
        ]
        plot_ohlc_candlestick_with_volume(sample_data)

    def test_plot_run_success2(self):
        sample_data = [
            {"date": "2023-01-01", "open": 100, "high": 110, "low": 95, "close": 105, "volume": 10},
            {
                "date": "2023-01-02",
                "open": 105,
                "high": 115,
                "low": 100,
                "close": 110,
                "volume": 10,
            },
            {
                "date": "2023-01-03",
                "open": 110,
                "high": 120,
                "low": 105,
                "close": 115,
                "volume": 10,
            },
            {
                "date": "2023-01-04",
                "open": 115,
                "high": 125,
                "low": 110,
                "close": 120,
                "volume": 10,
            },
            {
                "date": "2023-01-05",
                "open": 120,
                "high": 130,
                "low": 115,
                "close": 125,
                "volume": 10,
            },
            {"date": "2023-01-06", "open": 120, "high": 130, "low": 76, "close": 90, "volume": 10},
        ]
        prediction = [
            {"date": "2023-01-01", "predicted_price": 100},
            {"date": "2023-01-02", "predicted_price": 105},
            {"date": "2023-01-03", "predicted_price": 110},
            {"date": "2023-01-04", "predicted_price": 115},
            {"date": "2023-01-05", "predicted_price": 120},
            {"date": "2023-01-06", "predicted_price": 120},
        ]
        plot_ohlc_candlestick_with_volume_and_prediction(sample_data, prediction)

    def test_dataset_to_plot(self):
        for interval in ["1m", "5m", "15m", "30m", "1h"]:
            dataset = OHLCDatasetMmap(
                "memmap_dataset",
                window_range=(1024, 2048),
                is_train=True,
                rank=0,
                filter_intervals=[interval],
                world_size=8,
            )
            dataset.plot_kline(1000)


if __name__ == "__main__":
    unittest.main()
