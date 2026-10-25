"""Enumerate human support-provider turns and apply the minimal sanity and admissibility filter.

Admissibility excludes clearly role-invalid or non-client-facing turns. Decisions come from a
human admissibility review file (``--human-admissibility``); without it, an LLM reviewer is used.
Review file rows: {"sample_id": "<dialogue_id>:<turn_index>", "decision": "PASS" | "UNCERTAIN" | "CLEAR_FAIL"}.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from aria_esc.agents import DialogueAgent
from aria_esc.process_data.preferences import MiningConfig, deterministic_sanity_check, load_mining_config
from aria_esc.utils.io import read_jsonl, write_jsonl


def role_of(turn: dict) -> str:
    role = str(turn.get("speaker", turn.get("role", ""))).lower()
    if role in {"support_provider", "provider", "therapist", "assistant"}:
        return "assistant"
    if role in {"client", "seeker", "user"}:
        return "user"
    if role == "system":
        return "system"
    raise ValueError(f"Unknown turn role: {role!r}")


def enumerate_human_turns(dialogues: list[dict], config: MiningConfig) -> list[dict]:
    """Input: ordered complete dialogues of authorized observed human speech."""
    rows, seen = [], set()
    for dialogue in dialogues:
        history = []
        for index, turn in enumerate(dialogue["turns"]):
            role = role_of(turn)
            text = str(turn.get("text", turn.get("content", ""))).strip()
            if role == "assistant":
                sample_id = f"{dialogue['dialogue_id']}:{index}"
                if sample_id in seen:
                    raise ValueError(f"Duplicate support-provider turn identity: {sample_id}")
                seen.add(sample_id)
                valid, cleaned, reason = deterministic_sanity_check(text, config)
                has_client = bool(history and history[-1]["role"] == "user" and history[-1]["content"].strip())
                rows.append({
                    "sample_id": sample_id,
                    "dialogue_id": dialogue["dialogue_id"], "turn_index": index,
                    "context": [dict(t) for t in history[:-1]] if has_client else [dict(t) for t in history],
                    "client_utterance": history[-1]["content"] if has_client else "",
                    "original_text": text, "text": cleaned, "sanity_pass": valid and has_client,
                    "sanity_reason": reason if not valid else (None if has_client else "missing_adjacent_client"),
                })
            history.append({"role": role, "content": text})
    return rows


def apply_admissibility(row: dict, decision: dict | None) -> dict:
    if not row["sanity_pass"]:
        return {**row, "admissibility": None, "eligible": False}
    if not isinstance(decision, dict) or decision.get("decision") not in {"PASS", "UNCERTAIN", "CLEAR_FAIL"}:
        raise ValueError(f"{row['sample_id']}: missing or invalid admissibility decision")
    return {**row, "admissibility": decision, "eligible": decision["decision"] != "CLEAR_FAIL"}


class LlmAdmissibilityAgent(DialogueAgent):
    """Optional LLM-based admissibility review of human support-provider turns."""

    agent_name = "human_anchor_admissibility"
    prompt_filename = "llm_admissibility.txt"
    default_max_tokens = 500

    def build_messages(self, *, row: dict) -> list[dict]:
        data = {"conversation_context": row["context"], "client_utterance": row["client_utterance"],
                "therapist_response": row["text"]}
        return [{"role": "system", "content": self.system_message},
                {"role": "user", "content": self.prompt_template + "\n\nINPUT\n" + json.dumps(data, ensure_ascii=False)}]

    @staticmethod
    def parse_json_response(content: str) -> dict:
        result = DialogueAgent.parse_json_response(content)
        if result.get("decision") not in {"PASS", "UNCERTAIN", "CLEAR_FAIL"}:
            raise ValueError("Invalid admissibility decision")
        if not isinstance(result.get("fail_categories"), list) or not isinstance(result.get("reason"), str):
            raise ValueError("Admissibility output requires fail_categories and reason")
        return result


def attach_human_anchors(samples: list[dict], anchors: list[dict]) -> list[dict]:
    by_id = {str(row["sample_id"]): row for row in anchors}
    if len(by_id) != len(anchors):
        raise ValueError("Duplicate human-anchor identity")
    result, matched = [], 0
    for sample in samples:
        row = by_id.get(str(sample["sample_id"]))
        if row is None:
            continue
        matched += 1
        references = [c for c in sample["candidates"] if c["kind"] == "reference"]
        if len(references) != 1 or references[0]["text"] not in {row["original_text"], row["text"]}:
            raise ValueError(f"{sample['sample_id']}: reference text does not match the human anchor")
        if not row["sanity_pass"]:
            continue
        checked = apply_admissibility(row, row.get("admissibility"))
        if not checked["eligible"]:
            continue
        candidates = [{**c, "admissibility": row["admissibility"]} if c["kind"] == "reference" else dict(c)
                      for c in sample["candidates"]]
        result.append({**sample, "candidates": candidates})
    if samples and not matched:
        raise ValueError("No candidate sample_id matches a human-anchor sample_id")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare human anchors with the sanity and admissibility filter.")
    parser.add_argument("--input", required=True, help="Observed-human dialogues JSONL")
    parser.add_argument("--output", required=True, help="Eligible human anchors JSONL")
    parser.add_argument("--human-admissibility", help="Human admissibility review JSONL; if omitted, an LLM reviews")
    parser.add_argument("--model", help="LLM reviewer model (required without --human-admissibility)")
    parser.add_argument("--config", help="MiningConfig JSON")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    parser.add_argument("--prompt-dir", default=str(Path(__file__).resolve().parents[3] / "prompts"))
    parser.add_argument("--dry-run", action="store_true", help="Render LLM requests without calling the model")
    args = parser.parse_args()
    if not args.human_admissibility and not args.model:
        parser.error("--model is required without --human-admissibility")
    if args.human_admissibility and args.dry_run:
        parser.error("--dry-run applies to LLM review only")

    rows = enumerate_human_turns(read_jsonl(args.input), load_mining_config(args.config))
    agent, human = None, None
    if args.human_admissibility:
        human = {str(row["sample_id"]): row for row in read_jsonl(args.human_admissibility)}
    else:
        agent = LlmAdmissibilityAgent.from_prompt_dir(
            args.prompt_dir, model=args.model, base_url=args.base_url, api_key=args.api_key
        )
    outputs = []
    for row in rows:
        if not row["sanity_pass"]:
            outputs.append(apply_admissibility(row, None))
        elif human is not None:
            outputs.append(apply_admissibility(row, human.get(row["sample_id"])))
        elif args.dry_run:
            outputs.append({**row, "admissibility": None, "eligible": None,
                            "llm_messages": agent.build_messages(row=row)})
        else:
            outputs.append(apply_admissibility(row, agent.invoke_messages(agent.build_messages(row=row))))
    if args.dry_run and agent is not None:
        write_jsonl(args.output, outputs)
        print(f"Prepared LLM review requests for {len(outputs)} provider turns")
    else:
        anchors = [row for row in outputs if row.get("eligible") is True]
        write_jsonl(args.output, anchors)
        print(f"Prepared {len(anchors)} human anchors from {len(outputs)} provider turns")


if __name__ == "__main__":
    main()
