#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from aria_esc.agents import DialogueAgent, DialogueAgentSuite
from aria_esc.evaluation.metrics import RESULT_VALIDATORS, aggregate_paper_metrics
from aria_esc.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three evaluators and aggregate the seven main-paper metrics."
    )
    parser.add_argument("--input", required=True, help="Dialogue JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--served-model-name",
        help="Model ID exposed by /v1/models, if different from --model",
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    parser.add_argument("--prompt-dir", default=str(Path(__file__).resolve().parents[3] / "prompts"))
    parser.add_argument("--intervention-component-max-tokens", type=int, default=400)
    parser.add_argument("--genuine-progress-max-tokens", type=int, default=500)
    parser.add_argument("--openness-transition-max-tokens", type=int, default=350)
    parser.add_argument("--recent-messages", type=int, default=6)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=0.5)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true", help="Render requests without calling a model.")
    return parser.parse_args()


def requests_for_dialogue(
    dialogue: dict,
    agents: dict[str, DialogueAgent],
    recent_messages: int = 6,
) -> list[dict]:
    turns = dialogue.get("turns", [])
    requests = []
    previous_client = None
    previous_provider_for_client = None
    latest_provider = None
    flow = []
    previous_client_flow_index = None
    previous_client_index = None
    latest_provider_index = None
    for turn_index, turn in enumerate(turns):
        speaker = str(turn.get("speaker", "")).lower()
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        if speaker in {"support_provider", "provider", "assistant"}:
            if previous_client is not None:
                requests.append({
                    "dialogue_id": dialogue["dialogue_id"],
                    "turn_index": turn_index,
                    "baseline_turn_index": previous_client_index,
                    "evaluator": "intervention_component",
                    "agent": "intervention_component",
                    "messages": agents["intervention_component"].build_messages(**{
                        "previous_client_utterance": previous_client,
                        "support_provider_response": text,
                    }),
                })
            latest_provider = text
            latest_provider_index = turn_index
        elif speaker in {"client", "seeker", "user"}:
            if previous_client is not None and latest_provider is not None:
                progress_values = {
                    "previous_client_utterance": previous_client,
                    "current_client_utterance": text,
                    "intervening_support_provider_response": latest_provider,
                    "flow_context_before_baseline": (
                        "\n".join(flow[:previous_client_flow_index][-recent_messages:])
                        if previous_client_flow_index is not None and flow[:previous_client_flow_index]
                        else "(no prior context available)"
                    ),
                }
                requests.append({
                    "dialogue_id": dialogue["dialogue_id"],
                    "turn_index": turn_index,
                    "evaluator": "genuine_progress",
                    "baseline_turn_index": previous_client_index,
                    "provider_turn_index": latest_provider_index,
                    "agent": "genuine_progress",
                    "messages": agents["genuine_progress"].build_messages(**progress_values),
                })
                openness_values = {
                    "previous_support_provider_utterance": (
                        previous_provider_for_client or "(no previous support-provider turn)"
                    ),
                    "previous_client_utterance": previous_client,
                    "current_support_provider_utterance": latest_provider,
                    "current_client_utterance": text,
                }
                requests.append({
                    "dialogue_id": dialogue["dialogue_id"],
                    "turn_index": turn_index,
                    "evaluator": "openness_transition",
                    "agent": "openness_transition",
                    "messages": agents["openness_transition"].build_messages(**openness_values),
                })
            previous_provider_for_client = latest_provider
            previous_client = text
            previous_client_index = turn_index
            previous_client_flow_index = len(flow)
        else:
            raise ValueError(f"Unsupported speaker {speaker!r} in dialogue {dialogue.get('dialogue_id')}")
        flow.append(f"{speaker.title()}: {text}")
    return requests


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = DialogueAgentSuite.from_prompt_dir(
        prompt_dir=args.prompt_dir,
        model=args.model,
        base_url=args.base_url,
        served_model_name=args.served_model_name,
        api_key=args.api_key,
        intervention_component_max_tokens=args.intervention_component_max_tokens,
        genuine_progress_max_tokens=args.genuine_progress_max_tokens,
        openness_transition_max_tokens=args.openness_transition_max_tokens,
        retries=args.retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    agents = suite.evaluators()
    requests = [
        item
        for dialogue in read_jsonl(args.input)
        for item in requests_for_dialogue(dialogue, agents, recent_messages=args.recent_messages)
    ]
    if args.dry_run:
        output = output_dir / "judge_requests.jsonl"
        write_jsonl(output, requests)
        print(f"Rendered {len(requests)} evaluator requests to {output}")
        return
    results = []
    for request in requests:
        result = agents[request["agent"]].invoke_messages(
            request["messages"], validate=RESULT_VALIDATORS[request["evaluator"]]
        )
        results.append(request | {"result": result})
    judge_output = output_dir / "judge_outputs.jsonl"
    metrics_output = output_dir / "paper_metrics.json"
    transitions_output = output_dir / "eligible_transitions.jsonl"
    write_jsonl(judge_output, results)
    summary, transitions = aggregate_paper_metrics(results)
    metrics_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_jsonl(transitions_output, transitions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
