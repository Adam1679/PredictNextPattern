import json

import requests


def fetch_binance_klines(symbol, interval, limit=1000):
    base_url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    symbol = "ETHUSDT"
    interval = "1h"
    with open("training/get_binance_test_data_dumped.jsonl", "w") as f:
        data = fetch_binance_klines(symbol, interval, 1000)
        print("#lines", len(data))
        for line in data:
            open_time_s = line[0] // 1000
            open_price = float(line[1])
            high_price = float(line[2])
            low_price = float(line[3])
            close_price = float(line[4])
            volume = float(line[5])
            data = {
                "open_time": open_time_s,
                "open_price": open_price,
                "high_price": high_price,
                "low_price": low_price,
                "close_price": close_price,
                "volume": volume,
                "interval": interval,
                "symbol": symbol,
            }

            f.write(json.dumps(data) + "\n")
