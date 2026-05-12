import asyncio
import itertools
import os
import re
import uuid

from dotenv import find_dotenv, load_dotenv
from evaluators import (
    eval_completeness,
    eval_correctness,
    eval_groundedness,
    eval_overall_quality,
    eval_relevance,
    eval_structure,
)
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client

from open_deep_research.deep_researcher import deep_researcher_builder

load_dotenv(find_dotenv())
model = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-72B-Instruct-AWQ")

client = Client()

# NOTE: Configure the right dataset and evaluators
dataset_name = "Deep Research Bench"
dataset = client.clone_public_dataset(
    "https://smith.langchain.com/public/c5e7a6ad-fdba-478c-88e6-3a388459ce8b/d"
)
print(dataset)
example = next(client.list_examples(dataset_id=dataset.id))
print(example.metadata)
print(example.inputs)


def _is_english_example(ds_example) -> bool:
    """Return True for English examples and False for Chinese examples."""
    metadata = ds_example.metadata or {}

    # Prefer explicit language metadata when available.
    for key in ("language", "lang", "locale"):
        value = metadata.get(key)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"en", "en-us", "en-gb", "english"}:
                return True
            if normalized in {"zh", "zh-cn", "zh-tw", "chinese", "mandarin"}:
                return False

    # Fallback heuristic: treat examples containing CJK Unified Ideographs as non-English.
    user_prompt = ""
    messages = (
        ds_example.inputs.get("messages")
        if isinstance(ds_example.inputs, dict)
        else None
    )
    if isinstance(messages, list) and messages:
        first_message = messages[0]
        if isinstance(first_message, dict):
            user_prompt = first_message.get("content", "")

    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", user_prompt))
    return not has_cjk


all_examples = list(client.list_examples(dataset_id=dataset.id))
english_examples = [e for e in all_examples if _is_english_example(e)]
print(
    f"Loaded {len(all_examples)} total examples; {len(english_examples)} English examples"
)

evaluators = [
    eval_overall_quality,
    eval_relevance,
    eval_structure,
    eval_correctness,
    eval_groundedness,
    eval_completeness,
]
# NOTE: Configure the right parameters for the experiment, these will be logged in the metadata
max_structured_output_retries = 3
allow_clarification = False
max_concurrent_research_units = 10
search_api = "tavily"  # NOTE: We use Tavily to stay consistent
max_researcher_iterations = 6
max_react_tool_calls = 10
summarization_model = "openai:gpt-4.1-mini"
summarization_model_max_tokens = 8192
research_model = "openai:gpt-5"  # "anthropic:claude-sonnet-4-20250514"
research_model_max_tokens = 10000
compression_model = "openai:gpt-4.1"
compression_model_max_tokens = 10000
final_report_model = "openai:gpt-4.1"
final_report_model_max_tokens = 10000

job_counter = itertools.count(1)


async def target(
    inputs: dict,
):
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    job_id = str(next(job_counter))
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "job_id": job_id,
        }
    }
    # NOTE: Configure the right dataset and evaluators
    config["configurable"]["max_structured_output_retries"] = (
        max_structured_output_retries
    )
    config["configurable"]["allow_clarification"] = allow_clarification
    config["configurable"]["max_concurrent_research_units"] = (
        max_concurrent_research_units
    )
    config["configurable"]["search_api"] = search_api
    config["configurable"]["max_researcher_iterations"] = max_researcher_iterations
    config["configurable"]["max_react_tool_calls"] = max_react_tool_calls
    config["configurable"]["summarization_model"] = model
    config["configurable"]["summarization_model_max_tokens"] = (
        summarization_model_max_tokens
    )
    config["configurable"]["research_model"] = model
    config["configurable"]["research_model_max_tokens"] = research_model_max_tokens
    config["configurable"]["compression_model"] = model
    config["configurable"]["compression_model_max_tokens"] = (
        compression_model_max_tokens
    )
    config["configurable"]["final_report_model"] = model
    config["configurable"]["final_report_model_max_tokens"] = (
        final_report_model_max_tokens
    )
    # NOTE: We do not use MCP tools to stay consistent
    final_state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]},
        config,
    )
    return final_state


async def main():
    return await client.aevaluate(
        target,
        data=english_examples,
        evaluators=evaluators,
        experiment_prefix=f"ODR GPT-5, Tavily Search",
        max_concurrency=10,
        metadata={
            "max_structured_output_retries": max_structured_output_retries,
            "allow_clarification": allow_clarification,
            "max_concurrent_research_units": max_concurrent_research_units,
            "search_api": search_api,
            "max_researcher_iterations": max_researcher_iterations,
            "max_react_tool_calls": max_react_tool_calls,
            "dataset_filter": "english-only",
            "evaluated_examples": len(english_examples),
            "summarization_model": model,
            "summarization_model_max_tokens": summarization_model_max_tokens,
            "research_model": model,
            "research_model_max_tokens": research_model_max_tokens,
            "compression_model": model,
            "compression_model_max_tokens": compression_model_max_tokens,
            "final_report_model": model,
            "final_report_model_max_tokens": final_report_model_max_tokens,
        },
    )


if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)
