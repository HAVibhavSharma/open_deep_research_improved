"""End-to-end benchmark for the deep researcher workflow running through
the vLLM `/v1/chunked_chat/completions` endpoint.

Captures per-LLM-call metrics via a LangChain async callback handler and
prints an aggregated report at the end. Optionally repeats the same query
N times so you can see how TTFT improves once the anchor pool warms up.

Metrics collected (per LLM call):
    - node             which LangGraph node issued the call
    - ttft             time from llm_start to the first streamed token
                       (only meaningful for streaming calls)
    - latency          llm_start -> llm_end wall time
    - prompt_tokens    from response usage_metadata
    - completion_tok   from response usage_metadata
    - had_system       True if a SystemMessage was in the prompt
                       (-> an anchor chunk was sent)

Aggregates printed:
    - total wall time
    - total LLM calls + per-node counts
    - sum of prompt/completion tokens
    - mean / median / max TTFT and latency, overall and per-node

Server-side anchor-pool stats (cache hits, base captures, admit/activate
decisions) are NOT collected here -- read them from the vLLM server log
by greping for `[chunked_chat]` and `[anchor-pool]`.

Usage:
    .venv/bin/python tests/benchmark_chunked_workflow.py \\
        --query "What are the latest developments in fusion energy?" \\
        --runs 2
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage

from open_deep_research.deep_researcher import deep_researcher


@dataclass
class LLMCall:
    run_id: str
    node: str
    started: float
    first_token: float | None = None
    ended: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    had_system: bool = False
    n_messages: int = 0


class MetricsHandler(AsyncCallbackHandler):
    """Records per-LLM-call timing and token usage."""

    def __init__(self) -> None:
        self.calls: list[LLMCall] = []
        self._by_run: dict[str, LLMCall] = {}
        self._node_stack: list[str] = []

    async def on_chain_start(
        self,
        serialized: dict | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> None:
        name = None
        if serialized:
            name = serialized.get("name")
        name = name or kwargs.get("name") or "?"
        self._node_stack.append(name)

    async def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if self._node_stack:
            self._node_stack.pop()

    async def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        if self._node_stack:
            self._node_stack.pop()

    async def on_chat_model_start(
        self,
        serialized: dict,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        had_system = False
        n = 0
        for msg_list in messages or []:
            n += len(msg_list)
            for m in msg_list:
                if isinstance(m, SystemMessage) or getattr(m, "type", "") == "system":
                    had_system = True
        node = self._node_stack[-1] if self._node_stack else "?"
        call = LLMCall(
            run_id=str(run_id),
            node=node,
            started=time.perf_counter(),
            had_system=had_system,
            n_messages=n,
        )
        self._by_run[str(run_id)] = call
        self.calls.append(call)

    async def on_llm_new_token(self, token: str, *, run_id: UUID, **kwargs: Any) -> None:
        call = self._by_run.get(str(run_id))
        if call and call.first_token is None:
            call.first_token = time.perf_counter()

    async def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        call = self._by_run.get(str(run_id))
        if not call:
            return
        call.ended = time.perf_counter()

        # Prefer usage_metadata on the AIMessage; fall back to llm_output.token_usage.
        try:
            for gen_list in getattr(response, "generations", []) or []:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg is None:
                        continue
                    usage = getattr(msg, "usage_metadata", None) or {}
                    if usage:
                        call.prompt_tokens = usage.get("input_tokens") or call.prompt_tokens
                        call.completion_tokens = usage.get("output_tokens") or call.completion_tokens
                        call.total_tokens = usage.get("total_tokens") or call.total_tokens
        except Exception:
            pass

        try:
            llm_output = getattr(response, "llm_output", None) or {}
            token_usage = llm_output.get("token_usage") or {}
            if token_usage:
                call.prompt_tokens = call.prompt_tokens or token_usage.get("prompt_tokens")
                call.completion_tokens = (
                    call.completion_tokens or token_usage.get("completion_tokens")
                )
                call.total_tokens = call.total_tokens or token_usage.get("total_tokens")
        except Exception:
            pass


def _fmt_secs(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    return (
        f"mean={statistics.mean(xs):.3f}s "
        f"median={statistics.median(xs):.3f}s "
        f"max={max(xs):.3f}s"
    )


def format_report(calls: list[LLMCall], wall_s: float, label: str = "") -> str:
    lines: list[str] = []
    header = "DEEP RESEARCHER WORKFLOW BENCHMARK (chunked_chat)"
    if label:
        header += f"  --  {label}"
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("=" * len(header))

    lines.append(f"  total_wall_s         : {wall_s:.3f}")
    lines.append(f"  total_llm_calls      : {len(calls)}")
    lines.append(
        f"  total_prompt_tokens  : {sum((c.prompt_tokens or 0) for c in calls)}"
    )
    lines.append(
        f"  total_completion_tok : {sum((c.completion_tokens or 0) for c in calls)}"
    )

    ttfts = [c.first_token - c.started for c in calls if c.first_token is not None]
    latencies = [c.ended - c.started for c in calls if c.ended is not None]
    lines.append(f"  ttft     ({len(ttfts):>2} calls): {_fmt_secs(ttfts)}")
    lines.append(f"  latency  ({len(latencies):>2} calls): {_fmt_secs(latencies)}")

    lines.append("")
    lines.append("Per-node breakdown:")
    lines.append(
        f"  {'node':<32} {'n':>3}  "
        f"{'mean_ttft':>10}  {'mean_lat':>10}  "
        f"{'sum_p_tok':>10}  {'sum_c_tok':>10}"
    )
    by_node: dict[str, list[LLMCall]] = defaultdict(list)
    for c in calls:
        by_node[c.node].append(c)
    for node, cs in sorted(by_node.items()):
        ttfts_n = [c.first_token - c.started for c in cs if c.first_token is not None]
        lats_n = [c.ended - c.started for c in cs if c.ended is not None]
        pt = sum((c.prompt_tokens or 0) for c in cs)
        ct = sum((c.completion_tokens or 0) for c in cs)
        m_ttft = f"{statistics.mean(ttfts_n):.3f}s" if ttfts_n else "n/a"
        m_lat = f"{statistics.mean(lats_n):.3f}s" if lats_n else "n/a"
        lines.append(
            f"  {node[:32]:<32} {len(cs):>3}  "
            f"{m_ttft:>10}  {m_lat:>10}  "
            f"{pt:>10}  {ct:>10}"
        )

    lines.append("")
    lines.append("Per-call detail:")
    lines.append(
        f"  {'#':>3} {'node':<28} {'ttft':>8} {'lat':>8} "
        f"{'p_tok':>6} {'c_tok':>6} {'sys?':>4}"
    )
    for i, c in enumerate(calls, 1):
        ttft = (c.first_token - c.started) if c.first_token is not None else None
        lat = (c.ended - c.started) if c.ended is not None else None
        lines.append(
            f"  {i:>3} {c.node[:28]:<28} "
            f"{(f'{ttft:.3f}s' if ttft is not None else 'n/a'):>8} "
            f"{(f'{lat:.3f}s' if lat is not None else 'n/a'):>8} "
            f"{(c.prompt_tokens or 0):>6} {(c.completion_tokens or 0):>6} "
            f"{('yes' if c.had_system else 'no'):>4}"
        )
    return "\n".join(lines)


async def run_once(query: str, allow_clarification: bool, label: str) -> tuple[list[LLMCall], float, str]:
    handler = MetricsHandler()
    config: dict[str, Any] = {
        "callbacks": [handler],
        "configurable": {
            "allow_clarification": allow_clarification,
        },
    }
    t0 = time.perf_counter()
    result = await deep_researcher.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )
    wall = time.perf_counter() - t0
    final_report = result.get("final_report", "") or ""
    print(format_report(handler.calls, wall, label=label))
    print(f"\n[{label}] final_report length: {len(final_report)} chars")
    return handler.calls, wall, final_report


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="What are the latest developments in fusion energy in 2026?",
        help="Research query to drive the workflow.",
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
        help="Let the clarify_with_user node ask a clarifying question instead of skipping it.",
    )
    args = parser.parse_args()

    runs_metrics: list[tuple[list[LLMCall], float]] = []
    for i in range(1, args.runs + 1):
        label = f"run {i}/{args.runs}"
        print(f"\n>>> starting {label}")
        calls, wall, _ = await run_once(args.query, args.allow_clarification, label)
        runs_metrics.append((calls, wall))

    if len(runs_metrics) >= 2:
        print("\n" + "=" * 60)
        print("COLD vs WARM SUMMARY (run 1 vs subsequent runs)")
        print("=" * 60)
        for i, (calls, wall) in enumerate(runs_metrics, 1):
            ttfts = [c.first_token - c.started for c in calls if c.first_token is not None]
            mean_ttft = statistics.mean(ttfts) if ttfts else float("nan")
            print(
                f"  run {i}: wall={wall:.2f}s  llm_calls={len(calls):>3}  "
                f"mean_ttft={mean_ttft:.3f}s"
            )
        first_ttfts = [
            c.first_token - c.started
            for c in runs_metrics[0][0]
            if c.first_token is not None
        ]
        if first_ttfts:
            baseline = statistics.mean(first_ttfts)
            for i, (calls, _) in enumerate(runs_metrics[1:], 2):
                ttfts = [
                    c.first_token - c.started for c in calls if c.first_token is not None
                ]
                if not ttfts:
                    continue
                m = statistics.mean(ttfts)
                if m > 0:
                    print(f"  run {i} ttft speedup vs run 1: {baseline / m:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
