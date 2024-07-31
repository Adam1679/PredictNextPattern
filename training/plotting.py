from math import pi

import pandas as pd
from bokeh.io import save
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, DataRange1d, HoverTool, Legend, NumeralTickFormatter
from bokeh.plotting import figure


def plot_ohlc_candlestick_with_volume(data, symbol="", interval="", output_filename=""):
    # Convert data to pandas DataFrame if it's not already
    df = pd.DataFrame(data)
    date = pd.to_datetime(df["date"])
    df["date"] = pd.to_datetime(df["date"])

    # Sort the dataframe by date
    df = df.sort_values("date")

    df["change"] = df["close"] - df["open"]
    df["color"] = [
        "#D5E1DD" if close > open else "#F2583E" for close, open in zip(df["close"], df["open"])
    ]

    # Calculate the width of each bar
    time_diff = date.diff().min()
    w = time_diff.total_seconds() * 1000 * 0.8  # 80% of the minimum time difference

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    source = ColumnDataSource(df)

    # Candlestick chart
    p1 = figure(
        x_axis_type="datetime",
        tools=TOOLS,
        width=1000,
        height=400,
        title="Candlestick with Volume {}@{}".format(symbol, interval),
    )
    p1.x_range = DataRange1d(range_padding=0.05)
    p1.y_range = DataRange1d(range_padding=0.05)
    p1.xaxis.major_label_orientation = pi / 4
    p1.grid.grid_line_alpha = 0.3

    # Customize x-axis tick formatter
    # p1.xaxis.formatter = DatetimeTickFormatter(
    #     seconds=["%Y-%m-%d %H:%M:%S"],
    #     minutes=["%Y-%m-%d %H:%M:%S"],
    #     hours=["%Y-%m-%d %H:%M:%S"],
    #     days=["%Y-%m-%d %H:%M:%S"],
    #     months=["%Y-%m-%d %H:%M:%S"],
    #     years=["%Y-%m-%d %H:%M:%S"]
    # )

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

    # Align the charts
    charts = column(p1, p2)
    if output_filename:
        save(charts, filename=output_filename)
        print(f"Chart saved to {output_filename}")


def plot_ohlc_candlestick_with_volume_and_prediction(
    data, predicted_data, symbol="", interval="", output_filename=""
):
    # Convert data to pandas DataFrame if it's not already
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    # Sort the dataframe by date
    df = df.sort_values("date")

    df["change"] = df["close"] - df["open"]
    df["color"] = [
        "#D5E1DD" if close > open else "#F2583E" for close, open in zip(df["close"], df["open"])
    ]

    # Process predicted data
    pred_df = pd.DataFrame(predicted_data)
    assert (
        "predicted_price" in pred_df.columns
    ), "predicted_data must contain 'predicted_price' column but have {}".format(pred_df.columns)
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df = pred_df.sort_values("date")
    # Calculate the width of each bar
    time_diff = df["date"].diff().min()
    w = time_diff.total_seconds() * 1000 * 0.8  # 80% of the minimum time difference

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    source = ColumnDataSource(df)
    pred_source = ColumnDataSource(pred_df)

    # Candlestick chart
    p1 = figure(
        x_axis_type="datetime",
        tools=TOOLS,
        width=1000,
        height=400,
        title="Candlestick with Volume and Predicted Price {}@{}".format(symbol, interval),
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

    # Predicted price line
    predicted_line = p1.line(
        "date",
        "predicted_price",
        line_color="purple",
        line_alpha=0.5,
        line_dash="dashed",
        line_width=2,
        source=pred_source,
        name="Predicted Price",
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

    # Format y-axis ticks
    p1.yaxis.formatter = NumeralTickFormatter(format="0.0000")

    # Add legend
    legend = Legend(items=[("Actual", [candlesticks]), ("Predicted", [predicted_line])])
    p1.add_layout(legend, "right")

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

    # Align the charts
    charts = column(p1, p2)
    if output_filename:
        save(charts, output_filename)
        print(f"Chart saved to {output_filename}")


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
