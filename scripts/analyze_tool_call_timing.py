#!/usr/bin/env python3
"""Analyze node tool-call timing JSONL logs and generate latency plots.

This script reads JSONL entries from tool_call_timing.jsonl, extracts
prediction_to_end_latency_ns values, and generates two plots:
- Supervisor node latency tolerance plot
- Researcher node latency tolerance plot

Each plot shows per-request latency on the Y-axis and request index on the X-axis,
with extra spacing on the top and right for readability, and P05/P10 annotations
in the top-right corner.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate supervisor/researcher latency plots from timing JSONL logs."
    )
    parser.add_argument(
        "--input",
        default="tool_call_timing.jsonl",
        help="Path to the JSONL input file (default: tool_call_timing.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default="timing_plots",
        help="Directory to write plot images (default: timing_plots)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI for output images (default: 180)",
    )
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float:
    """Return percentile p (0 to 100) using linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_vals) - 1)
    weight = rank - low
    return sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight


def load_latency_by_node(jsonl_path: Path) -> dict[str, list[int]]:
    """Load prediction_to_end_latency_ns values grouped by node."""
    data: dict[str, list[int]] = {"supervisor": [], "researcher": []}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            node = row.get("node")
            latency = row.get("prediction_to_end_latency_ns")

            if node not in data:
                continue
            if latency is None:
                continue

            data[node].append(int(latency))

    return data


def plot_node_latencies(
    latencies_ns: list[int],
    node_name: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Create and save a single node latency plot."""
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc

    if not latencies_ns:
        raise ValueError(f"No entries found for node: {node_name}")

    x = list(range(1, len(latencies_ns) + 1))
    y_sec = [float(v) / 1_000_000_000.0 for v in latencies_ns]

    p05_sec = percentile(y_sec, 5.0)
    p10_sec = percentile(y_sec, 10.0)

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(
        x,
        y_sec,
        color="#1f77b4",
        edgecolor="#1a4f8b",
        linewidth=0.5,
        alpha=0.9,
        width=0.72,
    )

    ax.set_title(f"{node_name.capitalize()} Node: Prediction-to-End Latency", fontsize=14)
    ax.set_xlabel("Request Entry Index", fontsize=11)
    ax.set_ylabel("Latency Tolerance (seconds)", fontsize=11)

    # Add demarcation space on right and top for cleaner visuals.
    right_padding = max(1, int(len(x) * 0.06))
    ax.set_xlim(1, len(x) + right_padding)

    y_min = min(y_sec)
    y_max = max(y_sec)
    y_range = y_max - y_min
    top_padding = y_range * 0.12 if y_range > 0 else max(1.0, y_max * 0.12)
    ax.set_ylim(max(0.0, y_min - (y_range * 0.02)), y_max + top_padding)

    # Force major ticks at 10, 20, 30 seconds, etc.
    y_tick_max = max(10, int(math.ceil((y_max + top_padding) / 10.0) * 10))
    ax.set_yticks(list(range(0, y_tick_max + 1, 10)))

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)

    # Label each bar with its latency tolerance in seconds.
    for bar in bars:
        bar_height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar_height + max(0.15, y_tick_max * 0.004),
            f"{bar_height:.1f}s",
            ha="center",
            va="bottom",
            fontsize=6.5,
            rotation=90,
            color="#173a63",
            clip_on=False,
        )

    stats_text = f"P05: {p05_sec:.2f}s\nP10: {p10_sec:.2f}s"
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#aaaaaa"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    node_data = load_latency_by_node(input_path)

    supervisor_out = output_dir / "supervisor_latency_tolerance.png"
    researcher_out = output_dir / "researcher_latency_tolerance.png"

    plot_node_latencies(node_data["supervisor"], "supervisor", supervisor_out, args.dpi)
    plot_node_latencies(node_data["researcher"], "researcher", researcher_out, args.dpi)

    print(f"Wrote: {supervisor_out}")
    print(f"Wrote: {researcher_out}")


if __name__ == "__main__":
    main()
