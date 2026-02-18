"""Visualization utilities for strategy verification artifacts."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _compound_return(returns: pd.Series) -> float:
    """Compute compounded return from a return series."""
    if returns.empty:
        return 0.0
    return float((1.0 + returns.astype(float)).prod() - 1.0)


def build_performance_frame(
    compare_df: pd.DataFrame,
    rebalance_counts: Optional[Sequence[int]] = None,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Build time-series frame used by visualization and period summaries."""
    required_cols = {"Date", "Sim_Return", "Backtest_Return", "Sim_Equity"}
    missing = required_cols - set(compare_df.columns)
    if missing:
        raise ValueError(f"compare_df missing required columns: {sorted(missing)}")

    perf_df = compare_df.copy()
    perf_df["Date"] = pd.to_datetime(perf_df["Date"])
    perf_df = perf_df.sort_values("Date").reset_index(drop=True)

    for col in ("Sim_Return", "Backtest_Return", "Sim_Equity"):
        perf_df[col] = pd.to_numeric(perf_df[col], errors="coerce").fillna(0.0)

    if "Benchmark_Return" in perf_df.columns:
        perf_df["Benchmark_Return"] = pd.to_numeric(
            perf_df["Benchmark_Return"], errors="coerce"
        ).fillna(0.0)
    else:
        perf_df["Benchmark_Return"] = 0.0

    perf_df["Sim_CumNav"] = (1.0 + perf_df["Sim_Return"]).cumprod()
    perf_df["Sim_CumReturn"] = perf_df["Sim_CumNav"] - 1.0

    perf_df["Backtest_CumNav"] = (1.0 + perf_df["Backtest_Return"]).cumprod()
    perf_df["Backtest_CumReturn"] = perf_df["Backtest_CumNav"] - 1.0

    perf_df["Benchmark_CumNav"] = (1.0 + perf_df["Benchmark_Return"]).cumprod()
    perf_df["Benchmark_CumReturn"] = perf_df["Benchmark_CumNav"] - 1.0
    perf_df["Excess_CumReturn"] = (
        perf_df["Sim_CumReturn"] - perf_df["Benchmark_CumReturn"]
    )

    perf_df["Sim_RunningPeak"] = perf_df["Sim_CumNav"].cummax()
    perf_df["Sim_Drawdown"] = perf_df["Sim_CumNav"] / perf_df["Sim_RunningPeak"] - 1.0

    perf_df["Rebalance_Count"] = 0.0
    perf_df["Sim_Turnover"] = np.nan
    k_value = int(top_k) if top_k else 0
    if rebalance_counts is not None:
        count_arr = np.asarray(list(rebalance_counts), dtype=float)
        n = len(perf_df)
        if len(count_arr) >= n:
            count_arr = count_arr[:n]
        else:
            count_arr = np.pad(count_arr, (0, n - len(count_arr)), constant_values=0.0)
        perf_df["Rebalance_Count"] = count_arr
        if k_value > 0:
            perf_df["Sim_Turnover"] = perf_df["Rebalance_Count"] / float(k_value)

    return perf_df


def build_period_return_tables(
    perf_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build yearly and monthly return tables."""
    work_df = perf_df.copy()
    work_df["Year"] = work_df["Date"].dt.year
    work_df["Month"] = work_df["Date"].dt.month
    work_df["YearMonth"] = work_df["Date"].dt.strftime("%Y-%m")

    yearly_strategy = (
        work_df.groupby("Year", sort=True)["Sim_Return"]
        .apply(_compound_return)
        .rename("Strategy_Return")
    )
    yearly_benchmark = (
        work_df.groupby("Year", sort=True)["Benchmark_Return"]
        .apply(_compound_return)
        .rename("Benchmark_Return")
    )
    yearly_df = pd.concat([yearly_strategy, yearly_benchmark], axis=1).reset_index()
    yearly_df["Excess_Return"] = (
        yearly_df["Strategy_Return"] - yearly_df["Benchmark_Return"]
    )

    monthly_strategy = (
        work_df.groupby(["Year", "Month", "YearMonth"], sort=True)["Sim_Return"]
        .apply(_compound_return)
        .rename("Strategy_Return")
    )
    monthly_benchmark = (
        work_df.groupby(["Year", "Month", "YearMonth"], sort=True)["Benchmark_Return"]
        .apply(_compound_return)
        .rename("Benchmark_Return")
    )
    monthly_df = pd.concat([monthly_strategy, monthly_benchmark], axis=1).reset_index()
    monthly_df["Excess_Return"] = (
        monthly_df["Strategy_Return"] - monthly_df["Benchmark_Return"]
    )
    monthly_pivot = (
        monthly_df.pivot(index="Year", columns="Month", values="Strategy_Return")
        .reindex(columns=list(range(1, 13)))
        .sort_index()
    )
    return yearly_df, monthly_df, monthly_pivot


def build_summary_metrics(perf_df: pd.DataFrame, top_k: Optional[int] = None) -> pd.DataFrame:
    """Build strategy-level and benchmark-level summary metrics."""
    n = len(perf_df)
    if n == 0:
        return pd.DataFrame(
            {
                "Metric": [
                    "Strategy Total Return",
                    "Strategy Annualized Return",
                    "Strategy Sharpe Ratio",
                    "Strategy Max Drawdown",
                    "Benchmark Total Return",
                    "Benchmark Annualized Return",
                    "Benchmark Sharpe Ratio",
                    "Benchmark Max Drawdown",
                    "Excess Total Return vs Benchmark",
                    "Total Rebalance Count",
                    "Avg Daily Turnover",
                    "Aggregate Turnover",
                ],
                "Value": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    np.nan,
                    np.nan,
                ],
            }
        )

    strategy_ret = perf_df["Sim_Return"].astype(float).to_numpy()
    benchmark_ret = perf_df["Benchmark_Return"].astype(float).to_numpy()

    strategy_total = float(perf_df["Sim_CumNav"].iloc[-1] - 1.0)
    benchmark_total = float(perf_df["Benchmark_CumNav"].iloc[-1] - 1.0)
    excess_total = strategy_total - benchmark_total

    strategy_ann = float((1.0 + strategy_total) ** (252.0 / n) - 1.0)
    benchmark_ann = float((1.0 + benchmark_total) ** (252.0 / n) - 1.0)

    strategy_std = float(np.std(strategy_ret))
    benchmark_std = float(np.std(benchmark_ret))
    strategy_sharpe = float(np.mean(strategy_ret) / (strategy_std + 1e-9) * np.sqrt(252.0))
    benchmark_sharpe = float(np.mean(benchmark_ret) / (benchmark_std + 1e-9) * np.sqrt(252.0))
    strategy_max_drawdown = float(abs(perf_df["Sim_Drawdown"].min()))
    benchmark_running_peak = perf_df["Benchmark_CumNav"].cummax()
    benchmark_drawdown = perf_df["Benchmark_CumNav"] / (benchmark_running_peak + 1e-12) - 1.0
    benchmark_max_drawdown = float(abs(benchmark_drawdown.min()))

    rebalance_counts = perf_df["Rebalance_Count"].astype(float).to_numpy()
    total_rebalance_count = float(np.nansum(rebalance_counts))

    turnover = perf_df["Sim_Turnover"].astype(float).to_numpy()
    turnover = turnover[np.isfinite(turnover)]
    avg_daily_turnover = float(np.mean(turnover)) if turnover.size > 0 else np.nan

    k_value = int(top_k) if top_k else 0
    if k_value > 0 and n > 0:
        aggregate_turnover = total_rebalance_count / float(k_value * n)
    else:
        aggregate_turnover = np.nan

    return pd.DataFrame(
        {
            "Metric": [
                "Strategy Total Return",
                "Strategy Annualized Return",
                "Strategy Sharpe Ratio",
                "Strategy Max Drawdown",
                "Benchmark Total Return",
                "Benchmark Annualized Return",
                "Benchmark Sharpe Ratio",
                "Benchmark Max Drawdown",
                "Excess Total Return vs Benchmark",
                "Total Rebalance Count",
                "Avg Daily Turnover",
                "Aggregate Turnover",
            ],
            "Value": [
                strategy_total,
                strategy_ann,
                strategy_sharpe,
                strategy_max_drawdown,
                benchmark_total,
                benchmark_ann,
                benchmark_sharpe,
                benchmark_max_drawdown,
                excess_total,
                total_rebalance_count,
                avg_daily_turnover,
                aggregate_turnover,
            ],
        }
    )


def _build_curve_figure(perf_df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.65, 0.35],
        subplot_titles=("Cumulative Return Curve", "Drawdown Curve"),
    )

    fig.add_trace(
        go.Scatter(
            x=perf_df["Date"],
            y=perf_df["Sim_CumReturn"],
            mode="lines",
            name="Strategy Cumulative Return",
            line=dict(width=2, color="#005f73"),
            hovertemplate="%{x|%Y-%m-%d}<br>Strategy Cum Return: %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=perf_df["Date"],
            y=perf_df["Backtest_CumReturn"],
            mode="lines",
            name="Vector Backtest Cumulative Return",
            line=dict(width=2, color="#ee9b00"),
            hovertemplate="%{x|%Y-%m-%d}<br>Vector Cum Return: %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=perf_df["Date"],
            y=perf_df["Benchmark_CumReturn"],
            mode="lines",
            name="Benchmark Cumulative Return (index_jsl)",
            line=dict(width=2, color="#0a9396"),
            hovertemplate="%{x|%Y-%m-%d}<br>Benchmark Cum Return: %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=perf_df["Date"],
            y=perf_df["Excess_CumReturn"],
            mode="lines",
            name="Excess Cumulative Return (Strategy-Benchmark)",
            line=dict(width=2, color="#1d3557", dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>Excess Cum Return: %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=perf_df["Date"],
            y=perf_df["Sim_Drawdown"],
            mode="lines",
            name="Strategy Drawdown",
            line=dict(width=1.5, color="#ae2012"),
            fill="tozeroy",
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        height=860,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=30, t=80, b=35),
    )
    fig.update_yaxes(tickformat=".2%", row=1, col=1)
    fig.update_yaxes(tickformat=".2%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig


def _build_period_figure(
    yearly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    monthly_pivot: pd.DataFrame,
) -> go.Figure:
    month_labels = [f"{m:02d}" for m in range(1, 13)]
    z_values = monthly_pivot.to_numpy(dtype=float) * 100.0

    text_values = np.full_like(z_values, "", dtype=object)
    valid = np.isfinite(z_values)
    text_values[valid] = np.vectorize(lambda x: f"{x:.1f}%")(z_values[valid])

    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.12,
        row_heights=[0.30, 0.22, 0.48],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "heatmap"}]],
        subplot_titles=(
            "Return Comparison (Year/Month): Strategy vs Benchmark",
            "Excess Return (Year/Month): Strategy - Benchmark",
            "Monthly Strategy Return Heatmap",
        ),
    )

    # Yearly comparison (default visible)
    fig.add_trace(
        go.Bar(
            x=yearly_df["Year"].astype(str),
            y=yearly_df["Strategy_Return"],
            name="Strategy (Yearly)",
            marker_color="#005f73",
            offsetgroup="strategy_yearly",
            hovertemplate="Year %{x}<br>Strategy: %{y:.2%}<extra></extra>",
            visible=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=yearly_df["Year"].astype(str),
            y=yearly_df["Benchmark_Return"],
            name="Benchmark(index_jsl) (Yearly)",
            marker_color="#0a9396",
            offsetgroup="benchmark_yearly",
            hovertemplate="Year %{x}<br>Benchmark: %{y:.2%}<extra></extra>",
            visible=True,
        ),
        row=1,
        col=1,
    )

    # Monthly comparison (hidden by default)
    fig.add_trace(
        go.Bar(
            x=monthly_df["YearMonth"].astype(str),
            y=monthly_df["Strategy_Return"],
            name="Strategy (Monthly)",
            marker_color="#005f73",
            offsetgroup="strategy_monthly",
            hovertemplate="Month %{x}<br>Strategy: %{y:.2%}<extra></extra>",
            visible=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=monthly_df["YearMonth"].astype(str),
            y=monthly_df["Benchmark_Return"],
            name="Benchmark(index_jsl) (Monthly)",
            marker_color="#0a9396",
            offsetgroup="benchmark_monthly",
            hovertemplate="Month %{x}<br>Benchmark: %{y:.2%}<extra></extra>",
            visible=False,
        ),
        row=1,
        col=1,
    )

    # Yearly excess (default visible)
    excess_colors = ["#1d3557" if v >= 0 else "#e63946" for v in yearly_df["Excess_Return"]]
    fig.add_trace(
        go.Bar(
            x=yearly_df["Year"].astype(str),
            y=yearly_df["Excess_Return"],
            name="Excess Return (Yearly)",
            marker_color=excess_colors,
            hovertemplate="Year %{x}<br>Excess: %{y:.2%}<extra></extra>",
            visible=True,
        ),
        row=2,
        col=1,
    )

    # Monthly excess (hidden by default)
    monthly_excess_colors = [
        "#1d3557" if v >= 0 else "#e63946"
        for v in monthly_df["Excess_Return"].tolist()
    ]
    fig.add_trace(
        go.Bar(
            x=monthly_df["YearMonth"].astype(str),
            y=monthly_df["Excess_Return"],
            name="Excess Return (Monthly)",
            marker_color=monthly_excess_colors,
            hovertemplate="Month %{x}<br>Excess: %{y:.2%}<extra></extra>",
            visible=False,
        ),
        row=2,
        col=1,
    )

    # Monthly heatmap (always visible)
    fig.add_trace(
        go.Heatmap(
            z=z_values,
            x=month_labels,
            y=monthly_pivot.index.astype(str).tolist(),
            text=text_values,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale="RdYlGn",
            colorbar=dict(title="Monthly Return (%)"),
            zmid=0.0,
            hovertemplate="Year %{y} / Month %{x}<br>Return: %{z:.2f}%<extra></extra>",
            name="Monthly Return",
            visible=True,
        ),
        row=3,
        col=1,
    )

    # trace order:
    # 0 yearly strategy, 1 yearly benchmark, 2 monthly strategy, 3 monthly benchmark,
    # 4 yearly excess, 5 monthly excess, 6 heatmap
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=1.0,
                y=1.18,
                xanchor="right",
                yanchor="top",
                showactive=True,
                buttons=[
                    dict(
                        label="按年",
                        method="update",
                        args=[
                            {"visible": [True, True, False, False, True, False, True]},
                            {
                                "xaxis.title.text": "Year",
                                "xaxis2.title.text": "Year",
                            },
                        ],
                    ),
                    dict(
                        label="按月",
                        method="update",
                        args=[
                            {"visible": [False, False, True, True, False, True, True]},
                            {
                                "xaxis.title.text": "Year-Month",
                                "xaxis2.title.text": "Year-Month",
                            },
                        ],
                    ),
                ],
            )
        ],
    )

    fig.update_layout(
        template="plotly_white",
        height=1050,
        margin=dict(l=55, r=50, t=80, b=45),
        barmode="group",
    )
    fig.update_yaxes(tickformat=".2%", row=1, col=1)
    fig.update_yaxes(tickformat=".2%", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=3, col=1)
    fig.update_yaxes(title_text="Year", row=3, col=1)
    return fig


def _format_metric(metric: str, value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    metric_l = metric.lower()
    if metric_l == "total rebalance count":
        return f"{value:.0f}"
    if "corr" in metric_l or "correlation" in metric_l or "sharpe" in metric_l:
        return f"{value:.4f}"
    if (
        "return" in metric_l
        or "drawdown" in metric_l
        or "turnover" in metric_l
        or "overlap" in metric_l
    ):
        return f"{value:.2%}"
    return f"{value:.4f}"


def _write_html_bundle(
    html_path: str,
    curve_fig: go.Figure,
    period_fig: go.Figure,
    summary_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    strategy_name: str,
) -> None:
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    summary_render = summary_df.copy()
    summary_render["Value"] = [
        _format_metric(m, float(v)) for m, v in zip(summary_render["Metric"], summary_render["Value"])
    ]

    summary_table_html = summary_render.to_html(index=False, border=0)
    yearly_table_html = yearly_df.to_html(
        index=False,
        float_format=lambda x: f"{x:.4%}",
        border=0,
    )
    monthly_table_html = monthly_df.to_html(
        index=False,
        float_format=lambda x: f"{x:.4%}",
        border=0,
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        f.write("<title>Strategy Verification Visualization</title>")
        f.write(
            "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:20px;color:#111827;}"
            "h1{margin-bottom:6px;} .meta{color:#4b5563;margin-bottom:18px;}"
            "h2{margin-top:26px;} table{border-collapse:collapse;font-size:13px;}"
            "th,td{border:1px solid #d1d5db;padding:6px 10px;text-align:right;}"
            "th:first-child,td:first-child{text-align:left;}</style>"
        )
        f.write("</head><body>")
        f.write("<h1>Backtest Visualization Report</h1>")
        f.write(
            f"<div class='meta'>Strategy: {strategy_name} | Generated: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>"
        )
        f.write("<h2>Strategy Summary</h2>")
        f.write(summary_table_html)
        f.write(
            "<div class='meta'>Turnover definition: daily turnover = "
            "max(unique buys, unique sells) / top_k (TP sells excluded); "
            "aggregate turnover = total rebalance count / (top_k * days).</div>"
        )
        f.write(curve_fig.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write(period_fig.to_html(full_html=False, include_plotlyjs=False))
        f.write("<h2>Yearly Return Table (Strategy / Benchmark / Excess)</h2>")
        f.write(yearly_table_html)
        f.write("<h2>Monthly Return Table (Strategy / Benchmark / Excess)</h2>")
        f.write(monthly_table_html)
        f.write("</body></html>")


def generate_verification_visuals(
    compare_df: pd.DataFrame,
    output_dir: str,
    suffix: str,
    strategy_name: str,
    top_k: Optional[int] = None,
    rebalance_counts: Optional[Sequence[int]] = None,
    extra_summary_rows: Optional[Sequence[Tuple[str, float]]] = None,
) -> Dict[str, str]:
    """Generate visualization artifacts based on verification comparison output."""
    os.makedirs(output_dir, exist_ok=True)

    perf_df = build_performance_frame(
        compare_df=compare_df,
        rebalance_counts=rebalance_counts,
        top_k=top_k,
    )
    summary_df = build_summary_metrics(perf_df, top_k=top_k)
    if extra_summary_rows:
        extra_df = pd.DataFrame(list(extra_summary_rows), columns=["Metric", "Value"])
        if not extra_df.empty:
            extra_df["Value"] = pd.to_numeric(extra_df["Value"], errors="coerce")
            summary_df = pd.concat([summary_df, extra_df], ignore_index=True)
    yearly_df, monthly_df, monthly_pivot = build_period_return_tables(perf_df)

    perf_csv_path = os.path.join(output_dir, f"performance_detail{suffix}.csv")
    summary_csv_path = os.path.join(output_dir, f"strategy_summary{suffix}.csv")
    yearly_csv_path = os.path.join(output_dir, f"period_returns_yearly{suffix}.csv")
    monthly_csv_path = os.path.join(output_dir, f"period_returns_monthly{suffix}.csv")
    html_path = os.path.join(output_dir, f"performance_visualization{suffix}.html")

    perf_df.to_csv(perf_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    yearly_df.to_csv(yearly_csv_path, index=False)
    monthly_df.to_csv(monthly_csv_path, index=False)

    figure_title = f"{strategy_name} | Strategy / Vector / Benchmark Return and Drawdown"
    curve_fig = _build_curve_figure(perf_df, title=figure_title)
    period_fig = _build_period_figure(yearly_df, monthly_df, monthly_pivot)
    _write_html_bundle(
        html_path=html_path,
        curve_fig=curve_fig,
        period_fig=period_fig,
        summary_df=summary_df,
        yearly_df=yearly_df,
        monthly_df=monthly_df,
        strategy_name=strategy_name,
    )

    return {
        "performance_detail_csv": perf_csv_path,
        "strategy_summary_csv": summary_csv_path,
        "yearly_returns_csv": yearly_csv_path,
        "monthly_returns_csv": monthly_csv_path,
        "visualization_html": html_path,
    }
