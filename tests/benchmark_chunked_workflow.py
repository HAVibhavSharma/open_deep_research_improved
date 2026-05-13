"""End-to-end benchmark for the deep researcher workflow running through
the vLLM `/v1/chunked_chat/completions` endpoint.

Per-request metrics are collected at the httpx layer (inside
ChunkedChatHTTPClient.metrics) rather than via LangChain callbacks, because
LangGraph in this project does not reliably propagate callbacks through
all nested model.ainvoke calls.

Metrics per request (one HTTP request per LLM call):
    - latency_to_headers   wall time from request-sent to first response
                           header. For streaming this is ~TTFT; for
                           non-streaming it's "time to first byte".
    - status               HTTP status code (200 on success)
    - model                model name from the request body
    - n_chunks             number of chunks the rewriter produced
    - n_anchor_indices     how many of those are anchors (system prompt)
    - anchor_chars         total chars in anchor chunks
    - dynamic_chars        total chars in non-anchor chunks
    - stream               True if the openai SDK requested streaming
    - request_bytes        size of the rewritten request body
    - has_tools            True if `tools` was present in the request

Aggregates printed:
    - total wall time of the workflow
    - total HTTP requests (= LLM calls)
    - sum of anchor_chars, dynamic_chars, request_bytes
    - mean / median / max latency_to_headers
    - count of streaming vs non-streaming
    - count of tool-using vs not

Server-side anchor pool stats (cache hits, base captures, admit/activate)
are NOT visible here -- grep the vLLM server log for `[chunked_chat]` and
`[anchor-pool]` lines.

Usage:
    CHUNKED_CHAT_DEBUG=1 OPENAI_BASE_URL=http://localhost:8000/v1 \\
    .venv/bin/python tests/benchmark_chunked_workflow.py \\
        --query "What are the latest developments in fusion energy?" \\
        --runs 2
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

# Importing deep_researcher also creates the shared _CHUNKED_HTTP_CLIENT
# that the graph wires into every chat model.
from open_deep_research.deep_researcher import _CHUNKED_HTTP_CLIENT, deep_researcher
from open_deep_research.chunked_chat_client import ChunkedRequestMetric


def _fmt_secs(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    return (
        f"mean={statistics.mean(xs):.3f}s "
        f"median={statistics.median(xs):.3f}s "
        f"max={max(xs):.3f}s"
    )


def format_report(
    metrics: list[ChunkedRequestMetric], wall_s: float, label: str
) -> str:
    header = f"DEEP RESEARCHER WORKFLOW BENCHMARK (chunked_chat)  --  {label}"
    bar = "=" * len(header)
    out = [bar, header, bar]
    out.append(f"  total_wall_s         : {wall_s:.3f}")
    out.append(f"  total_http_requests  : {len(metrics)}")

    statuses = Counter(m.status for m in metrics)
    out.append(f"  status_counts        : {dict(statuses)}")
    out.append(
        f"  streaming / non-stream: "
        f"{sum(1 for m in metrics if m.stream)} / "
        f"{sum(1 for m in metrics if not m.stream)}"
    )
    out.append(
        f"  tool_using / not     : "
        f"{sum(1 for m in metrics if m.has_tools)} / "
        f"{sum(1 for m in metrics if not m.has_tools)}"
    )

    total_anchor = sum(m.anchor_chars for m in metrics)
    total_dynamic = sum(m.dynamic_chars for m in metrics)
    total_bytes = sum(m.request_bytes for m in metrics)
    out.append(f"  total_anchor_chars   : {total_anchor}")
    out.append(f"  total_dynamic_chars  : {total_dynamic}")
    out.append(f"  total_request_bytes  : {total_bytes}")

    lats = [m.latency_to_headers for m in metrics if m.latency_to_headers is not None]
    out.append(f"  latency_to_headers   : {_fmt_secs(lats)}")

    out.append("")
    out.append("Per-request detail:")
    out.append(
        f"  {'#':>3}  {'status':>6}  {'lat_s':>7}  "
        f"{'anc_ch':>7}  {'dyn_ch':>7}  {'req_B':>7}  "
        f"{'stream':>6}  {'tools':>5}"
    )
    for i, m in enumerate(metrics, 1):
        lat = m.latency_to_headers
        out.append(
            f"  {i:>3}  {str(m.status or '?'):>6}  "
            f"{(f'{lat:.3f}' if lat is not None else 'n/a'):>7}  "
            f"{m.anchor_chars:>7}  {m.dynamic_chars:>7}  "
            f"{m.request_bytes:>7}  "
            f"{('yes' if m.stream else 'no'):>6}  "
            f"{('yes' if m.has_tools else 'no'):>5}"
        )
    return "\n".join(out)


async def run_once(query: str, allow_clarification: bool, label: str) -> tuple[list[ChunkedRequestMetric], float, str]:
    _CHUNKED_HTTP_CLIENT.reset_metrics()
    config: dict[str, Any] = {
        "configurable": {"allow_clarification": allow_clarification},
    }
    t0 = time.perf_counter()
    result = await deep_researcher.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )
    wall = time.perf_counter() - t0
    # Snapshot metrics now in case a later run resets the list.
    metrics = list(_CHUNKED_HTTP_CLIENT.metrics)
    final_report = result.get("final_report", "") or ""
    print(format_report(metrics, wall, label=label))
    print(f"\n[{label}] final_report length: {len(final_report)} chars")
    return metrics, wall, final_report


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="What are the latest developments in fusion energy in 2026?",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="How many times to run the same query (>=2 lets you see cold vs warm anchor pool).",
    )
    parser.add_argument(
        "--allow-clarification",
        action="store_true",
        help="Let clarify_with_user ask a question (default off so the workflow runs end-to-end).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write per-run final_report files into. "
             "Defaults to tests/benchmark_outputs/<timestamp>/.",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "benchmark_outputs" / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> writing per-run outputs to {output_dir}")

    runs: list[tuple[list[ChunkedRequestMetric], float]] = []
    for i in range(1, args.runs + 1):
        label = f"run {i}/{args.runs}"
        print(f"\n>>> starting {label}")
        metrics, wall, final_report = await run_once(
            args.query, args.allow_clarification, label
        )
        runs.append((metrics, wall))

        out_path = output_dir / f"run_{i:02d}.md"
        report_text = format_report(metrics, wall, label=label)
        with out_path.open("w") as f:
            f.write(f"# {label}\n\n")
            f.write(f"**Query:** {args.query}\n\n")
            f.write(f"**Wall time:** {wall:.3f}s\n\n")
            f.write(f"**HTTP requests:** {len(metrics)}\n\n")
            f.write("## Metrics\n\n```\n")
            f.write(report_text)
            f.write("\n```\n\n")
            f.write("## Final Report\n\n")
            f.write(final_report if final_report else "_(empty)_")
            f.write("\n")
        print(f"[{label}] wrote {out_path}")

    if len(runs) >= 2:
        print("\n" + "=" * 60)
        print("COLD vs WARM SUMMARY")
        print("=" * 60)
        for i, (metrics, wall) in enumerate(runs, 1):
            lats = [m.latency_to_headers for m in metrics if m.latency_to_headers is not None]
            mean_lat = statistics.mean(lats) if lats else float("nan")
            print(
                f"  run {i}: wall={wall:6.2f}s  "
                f"http_reqs={len(metrics):>3}  "
                f"mean_lat_to_headers={mean_lat:.3f}s"
            )
        base_lats = [
            m.latency_to_headers
            for m in runs[0][0]
            if m.latency_to_headers is not None
        ]
        if base_lats:
            baseline = statistics.mean(base_lats)
            for i, (metrics, _) in enumerate(runs[1:], 2):
                lats = [
                    m.latency_to_headers
                    for m in metrics
                    if m.latency_to_headers is not None
                ]
                if not lats:
                    continue
                m = statistics.mean(lats)
                if m > 0:
                    print(f"  run {i} latency speedup vs run 1: {baseline / m:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
