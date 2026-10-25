#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from aria_esc.agents import SeekerAgent, SupportProviderAgent
from aria_esc.backends import AnthropicBackend, TransformersBackend
from aria_esc.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out support-provider and seeker turns.")
    parser.add_argument(
        "--input",
        required=True,
        help="Scenario JSONL with scenario_id, scenario_name, and opening_client_utterance",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--provider-backend",
        choices=("checkpoint", "openai", "anthropic"),
        default="checkpoint",
    )
    parser.add_argument("--provider-model")
    parser.add_argument("--provider-checkpoint", help="Local full checkpoint or LoRA adapter directory")
    parser.add_argument("--provider-base-model", help="Optional base-model override for a LoRA adapter")
    parser.add_argument("--provider-device-map", default="auto")
    parser.add_argument("--seeker-backend", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--seeker-model", required=True)
    parser.add_argument("--provider-served-model-name")
    parser.add_argument("--seeker-served-model-name")
    parser.add_argument("--provider-base-url", default=os.environ.get("PROVIDER_BASE_URL"))
    parser.add_argument("--seeker-base-url", default=os.environ.get("SEEKER_BASE_URL"))
    parser.add_argument("--provider-api-key")
    parser.add_argument("--seeker-api-key")
    parser.add_argument("--turns", type=int, default=20, help="Number of provider-seeker exchanges per case")
    parser.add_argument("--num-rollouts", type=int, default=4, help="Rollouts per scenario")
    parser.add_argument("--provider-temperature", type=float,
                        help="Default: 0.7 for a local checkpoint; provider default for an API model")
    parser.add_argument("--provider-top-p", type=float, default=0.95, help="Top-p for a local checkpoint provider")
    parser.add_argument("--seeker-temperature", type=float, default=0.2)
    parser.add_argument("--prompt-dir", default=str(Path(__file__).resolve().parents[3] / "prompts"))
    args = parser.parse_args()
    if args.turns < 1 or args.num_rollouts < 1:
        parser.error("--turns and --num-rollouts must be positive")
    if args.provider_backend == "checkpoint" and not args.provider_checkpoint:
        parser.error("--provider-checkpoint is required for the checkpoint backend")
    if args.provider_backend != "checkpoint" and not args.provider_model:
        parser.error("--provider-model is required for an API backend")
    return args


def api_key(backend: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    variable = "ANTHROPIC_API_KEY" if backend == "anthropic" else "OPENAI_API_KEY"
    return os.environ.get(variable, "local" if backend == "openai" else None)


def api_base_url(backend: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    variable = "ANTHROPIC_BASE_URL" if backend == "anthropic" else "OPENAI_BASE_URL"
    return os.environ.get(variable)


def generation_backend(
    backend: str,
    *,
    model: str | None,
    checkpoint: str | None = None,
    base_model: str | None = None,
    device_map: str = "auto",
    top_p: float | None = None,
    base_url: str | None = None,
    key: str | None = None,
):
    if backend == "checkpoint":
        return TransformersBackend(checkpoint, base_model=base_model, device_map=device_map, top_p=top_p)
    if backend == "anthropic":
        return AnthropicBackend(model, api_key=key, base_url=base_url)
    return None


def runtime_case_summary(row: dict, dialogue: list[dict]) -> dict:
    client_turns = [turn["text"] for turn in dialogue if turn["speaker"] == "client"]
    provider_turns = [turn["text"] for turn in dialogue if turn["speaker"] == "support_provider"]
    return {
        "scenario_name": row.get("scenario_name", ""),
        # The scenario opening stays visible to the seeker at every turn.
        "first_client_utterance": row["opening_client_utterance"],
        "latest_client_utterance": client_turns[-1] if client_turns else "",
        "latest_support_provider_response": provider_turns[-1] if provider_turns else "",
    }


def rollout_case(row: dict, provider: SupportProviderAgent, seeker: SeekerAgent, turns: int,
                 repeat: int = 1) -> dict:
    opening = str(row.get("opening_client_utterance", "")).strip()
    if not opening:
        raise ValueError("Each scenario requires opening_client_utterance")
    dialogue = [{"speaker": "client", "text": opening}]
    for _ in range(turns):
        provider_text = provider.respond(recent_turns=dialogue)
        dialogue.append({"speaker": "support_provider", "text": provider_text})
        seeker_result = seeker.respond(
            case_summary=runtime_case_summary(row, dialogue),
            recent_turns=dialogue[:-1],
            latest_support_provider_response=provider_text,
        )
        client_text = str(seeker_result.get("next_client_turn", "")).strip()
        if not client_text:
            raise ValueError("Seeker output requires a non-empty next_client_turn")
        dialogue.append({"speaker": "client", "text": client_text})
    return {
        "dialogue_id": f"{row['scenario_id']}-r{repeat}",
        "scenario_id": row["scenario_id"],
        "scenario_name": row.get("scenario_name"),
        "repeat": repeat,
        "turns": dialogue,
    }


def main() -> None:
    args = parse_args()
    prompt_dir = Path(args.prompt_dir)
    provider_key = api_key(args.provider_backend, args.provider_api_key)
    seeker_key = api_key(args.seeker_backend, args.seeker_api_key)
    provider_base_url = api_base_url(args.provider_backend, args.provider_base_url)
    seeker_base_url = api_base_url(args.seeker_backend, args.seeker_base_url)
    provider_backend = generation_backend(
        args.provider_backend,
        model=args.provider_model,
        checkpoint=args.provider_checkpoint,
        base_model=args.provider_base_model,
        device_map=args.provider_device_map,
        top_p=args.provider_top_p,
        base_url=provider_base_url,
        key=provider_key,
    )
    seeker_backend = generation_backend(
        args.seeker_backend,
        model=args.seeker_model,
        base_url=seeker_base_url,
        key=seeker_key,
    )
    provider = SupportProviderAgent.from_prompt_dir(
        prompt_dir,
        model=args.provider_model or args.provider_checkpoint,
        served_model_name=args.provider_served_model_name,
        base_url=provider_base_url,
        api_key=provider_key or "local",
        backend=provider_backend,
    )
    local = args.provider_backend == "checkpoint"
    provider.temperature = (args.provider_temperature if args.provider_temperature is not None
                            else 0.7 if local else None)
    # API providers use provider-default decoding.
    provider.max_tokens = provider.max_tokens if local else None
    seeker = SeekerAgent.from_prompt_dir(
        prompt_dir,
        model=args.seeker_model,
        served_model_name=args.seeker_served_model_name,
        base_url=seeker_base_url,
        api_key=seeker_key or "local",
        backend=seeker_backend,
        temperature=args.seeker_temperature,
    )

    outputs = [
        rollout_case(row, provider, seeker, args.turns, repeat)
        for row in read_jsonl(args.input)
        for repeat in range(1, args.num_rollouts + 1)
    ]

    write_jsonl(args.output, outputs)
    print(f"Wrote {len(outputs)} rollout dialogues to {args.output}")


if __name__ == "__main__":
    main()
