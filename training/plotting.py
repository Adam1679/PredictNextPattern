from math import pi

import numpy as np
import pandas as pd
from bokeh.io import save
from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    DataRange1d,
    DataTable,
    HoverTool,
    Legend,
    NumeralTickFormatter,
    TableColumn,
)
from bokeh.plotting import figure


def plot_ohlc_candlestick_with_volume(data, symbol="", interval="", output_filename=""):
    return plot_ohlc_candlestick(
        data, symbol=symbol, interval=interval, output_filename=output_filename
    )


def plot_ohlc_candlestick_with_volume_and_prediction(
    data,
    predicted_data,
    symbol="",
    interval="",
    output_filename="",
):
    return plot_ohlc_candlestick(
        data,
        predicted_data=predicted_data,
        symbol=symbol,
        interval=interval,
        output_filename=output_filename,
    )


def plot_ohlc_candlestick(data, predicted_data=None, symbol="", interval="", output_filename=""):
    # Convert data to pandas DataFrame if it's not already
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    # Sort the dataframe by date
    df = df.sort_values("date")

    df["change"] = df["close"] - df["open"]
    df["color"] = [
        "limegreen" if close > open else "red" for close, open in zip(df["close"], df["open"])
    ]

    # Calculate the width of each bar
    time_diff = df["date"].diff().min()
    w = time_diff.total_seconds() * 1000 * 0.8  # 80% of the minimum time difference

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    source = ColumnDataSource(df)

    # Candlestick chart
    p1 = figure(
        x_axis_type="datetime",
        tools=TOOLS,
        width=1000,
        height=400,
        title=f"Candlestick with Volume {symbol}@{interval}",
    )
    p1.x_range = DataRange1d(range_padding=0.05)
    p1.y_range = DataRange1d(range_padding=0.05)
    p1.xaxis.major_label_orientation = pi / 4
    p1.grid.grid_line_alpha = 0.3

    # Segment (High-Low)
    p1.segment("date", "high", "date", "low", color="black", source=source)

    # Candlestick bodies
    candlesticks = p1.vbar(
        "date", w, "open", "close", fill_color="color", line_color="black", source=source
    )

    # Candlestick hover tool
    hover_candlestick = HoverTool(
        renderers=[candlesticks],
        tooltips=[
            ("Date", "@date{%Y-%m-%d %H:%M:%S}"),
            ("Open", "@open{0.0000}"),
            ("High", "@high{0.0000}"),
            ("Low", "@low{0.0000}"),
            ("Close", "@close{0.0000}"),
            ("Change", "@change{0.0000}"),
            ("Type", "@color{bull(#D5E1DD),bear(#F2583E)}"),
        ],
        formatters={"@date": "datetime", "@color": "printf"},
        mode="vline",
    )
    p1.add_tools(hover_candlestick)

    # Format y-axis ticks
    p1.yaxis.formatter = NumeralTickFormatter(format="0.0000")

    # Add predicted price line if predicted_data is provided
    if predicted_data is not None:
        pred_df = pd.DataFrame(predicted_data)

        # the predicted price is shifted by 1 to align with the actual price
        pred_df["predicted_price"] = pred_df["predicted_price"].shift(1)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        pred_df = pred_df.sort_values("date")
        # Merge actual and predicted data
        merged_df = pd.merge(df, pred_df, on="date", how="inner")
        predicted_return = (
            merged_df["predicted_price"].values - merged_df["open"].values
        ) / merged_df["open"].values
        predicted_return = np.nan_to_num(predicted_return)
        # assuming buy and sell at open price
        pnl_ratio = (merged_df["close"].values - merged_df["open"].values) / merged_df[
            "open"
        ].values
        signal = np.where(np.abs(predicted_return) > 0, np.sign(predicted_return), 0.0)
        pnl = signal * pnl_ratio
        merged_df["signal"] = signal
        merged_df["pnl"] = pnl
        merged_df["pnl"] = merged_df["pnl"].fillna(0)
        merged_df["cost"] = 8e-4
        merged_df["cumulative_pnl"] = merged_df["pnl"].cumsum()
        merged_df["cumulative_pnl_with_cost"] = (
            merged_df["pnl"].cumsum() - merged_df["cost"].cumsum()
        )
        merged_source = ColumnDataSource(merged_df)
        predicted_line = p1.line(
            "date",
            "predicted_price",
            line_color="purple",
            line_alpha=0.5,
            line_dash="dashed",
            line_width=2,
            source=merged_source,
            name="Predicted Price",
        )

        # Predicted price hover tool
        hover_predicted = HoverTool(
            renderers=[predicted_line],
            tooltips=[
                ("Date", "@date{%Y-%m-%d %H:%M:%S}"),
                ("Predicted Price", "@predicted_price{0.0000}"),
            ],
            formatters={"@date": "datetime"},
            mode="vline",
        )
        p1.add_tools(hover_predicted)

        # Add legend
        legend = Legend(items=[("Actual", [candlesticks]), ("Predicted", [predicted_line])])
        p1.add_layout(legend, "right")
    else:
        merged_source = None
    # Volume chart
    p2 = figure(x_axis_type="datetime", tools=TOOLS, width=1000, height=200, x_range=p1.x_range)
    p2.y_range = DataRange1d(range_padding=0.05, start=0)
    p2.xaxis.major_label_orientation = pi / 4
    p2.grid.grid_line_alpha = 0.3

    # Use the same x-axis formatter for the volume chart
    p2.xaxis.formatter = p1.xaxis.formatter

    # Volume bars
    volume_bars = p2.vbar(
        "date", w, "volume", 0, fill_color="color", line_color="black", source=source, alpha=0.8
    )

    # Volume hover tool
    hover_volume = HoverTool(
        renderers=[volume_bars],
        tooltips=[("Date", "@date{%Y-%m-%d %H:%M:%S}"), ("Volume", "@volume{0,0}")],
        formatters={"@date": "datetime"},
        mode="vline",
    )
    p2.add_tools(hover_volume)

    p2.yaxis.axis_label = "Volume"

    if merged_source is not None:
        # Price difference chart
        p3 = figure(x_axis_type="datetime", tools=TOOLS, width=1000, height=200, x_range=p1.x_range)
        p3.y_range = DataRange1d(range_padding=0.05)
        p3.xaxis.major_label_orientation = pi / 4
        p3.grid.grid_line_alpha = 0.3

        # Use the same x-axis formatter for the price difference chart
        p3.xaxis.formatter = p1.xaxis.formatter

        # Price difference line
        cumulative_pnl_line = p3.line(
            "date",
            "cumulative_pnl",
            line_color="blue",
            line_width=2,
            source=merged_source,
            name="Cum PnL",
        )
        cumulative_pnl_with_cost = p3.line(
            "date",
            "cumulative_pnl_with_cost",
            line_color="green",
            line_width=2,
            source=merged_source,
            name="Cum PnL with Fees",
        )

        # Price difference hover tool
        hover_price_diff = HoverTool(
            renderers=[cumulative_pnl_line, cumulative_pnl_with_cost],
            tooltips=[
                ("Date", "@date{%Y-%m-%d %H:%M:%S}"),
                ("Cum Pnl", "@cumulative_pnl{0.0000}"),
                ("Cum Pnl With Fee", "@cumulative_pnl_with_cost{0.0000}"),
            ],
            formatters={"@date": "datetime"},
            mode="vline",
        )
        p3.add_tools(hover_price_diff)

        p3.yaxis.axis_label = "Cumulative PnL"

        # summary stats
        stats = {
            "total_pnl": merged_df["cumulative_pnl"].iloc[-1],
            "total_pnl_with_cost": merged_df["cumulative_pnl_with_cost"].iloc[-1],
            "average_pnl": merged_df["pnl"].mean(),
            "average_cost": merged_df["cost"].mean(),
            "pnl_max": merged_df["pnl"].max(),
            "pnl_min": merged_df["pnl"].min(),
            "pnl_p90": merged_df["pnl"].quantile(0.9),
            "pnl_p10": merged_df["pnl"].quantile(0.1),
            "num_trades": (merged_df["signal"] != 0).sum(),
            "num_long": (merged_df["signal"] > 0).sum(),
            "num_short": (merged_df["signal"] < 0).sum(),
            "win_rate": (merged_df["pnl"] > 0).mean(),
        }
        column_stats = {
            "Metric": [
                "Total PnL",
                "Total PnL with Cost",
                "Average PnL",
                "Average Cost",
                "Max PnL",
                "Min PnL",
                "P90 PnL",
                "P10 PnL",
                "Number of Trades",
                "Number of (Long) Trades",
                "Number of (Short) Trades",
                "Win Rate",
            ],
            "Values": [
                stats["total_pnl"],
                stats["total_pnl_with_cost"],
                stats["average_pnl"],
                stats["average_cost"],
                stats["pnl_max"],
                stats["pnl_min"],
                stats["pnl_p90"],
                stats["pnl_p10"],
                stats["num_trades"],
                stats["num_long"],
                stats["num_short"],
                stats["win_rate"],
            ],
        }
        summary_source = ColumnDataSource(column_stats)
        columns = [
            TableColumn(field="Metric", title="Metric"),
            TableColumn(field="Values", title="Values"),
        ]
        summary_table = DataTable(source=summary_source, columns=columns, width=400, height=150)

        # Align the charts
        charts = column(p1, p2, p3, summary_table)
    else:
        charts = column(p1, p2)

    if output_filename:
        save(charts, filename=output_filename)
        print(f"Chart saved to {output_filename}")

    return charts


# Example usage
if __name__ == "__main__":
    # Sample data (replace this with your actual data)
    sample_data = [
        {"date": "2023-01-01", "open": 100, "high": 110, "low": 95, "close": 105, "volume": 10},
        {"date": "2023-01-02", "open": 105, "high": 115, "low": 100, "close": 110, "volume": 10},
        {"date": "2023-01-03", "open": 110, "high": 120, "low": 105, "close": 115, "volume": 10},
        {"date": "2023-01-04", "open": 115, "high": 125, "low": 110, "close": 120, "volume": 10},
        {"date": "2023-01-05", "open": 120, "high": 130, "low": 115, "close": 125, "volume": 10},
        {"date": "2023-01-06", "open": 120, "high": 130, "low": 76, "close": 90, "volume": 10},
    ]

    plot_ohlc_candlestick_with_volume(sample_data)
