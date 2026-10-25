"""Discounted delayed client-valence targets (Eq. 1).

Input JSONL, one dialogue per line, with utterance-level client valence in [-1, 1]
(``null`` when no usable audio/visual estimate exists)::

    {"dialogue_id": "<id>", "turns": [
      {"speaker": "client", "text": "...", "valence": -0.21},
      {"speaker": "support_provider", "text": "..."},
      {"speaker": "client", "text": "...", "valence": null}, ...]}

Output rows are predictor-training records: ``{"context": [...], "delayed_valence_reward": R}``.
Run once per valence source (audio-derived, visual-derived, optionally smoothed)
by pointing ``--valence-field`` at the corresponding per-turn field.
"""
from __future__ import annotations

import argparse
import json
import math

from aria_esc.process_data.anchors import role_of
from aria_esc.utils.io import read_jsonl, write_jsonl


def valid_valence(value) -> float | None:
    if value is None or value == "":
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    if not -1 <= value <= 1:
        raise ValueError("Valence must be normalized to [-1, 1]; convert missing sentinels to null")
    return value


def discounted_target(previous, future: list, *, within: int = 5, horizon: int = 3,
                      gamma: float = 0.7) -> float | None:
    if horizon < 1 or within < horizon or not 0 <= gamma <= 1:
        raise ValueError("Require 1 <= horizon <= within and gamma in [0, 1]")
    baseline = valid_valence(previous)
    values = [v for value in future[:within] if (v := valid_valence(value)) is not None][:horizon]
    if baseline is None or len(values) != horizon:
        return None
    path = [baseline, *values]
    return sum(gamma ** k * (path[k + 1] - path[k]) for k in range(horizon))


def build_target_rows(dialogues: list[dict], *, valence_field: str = "valence", within: int = 5,
                      horizon: int = 3, gamma: float = 0.7) -> tuple[list[dict], dict]:
    """A dialogue is a boundary: no baseline/future observation crosses it.

    Missing future observations consume client-turn slots but not valid-observation
    slots. The baseline is the last valid client valence before the response; if the
    adjacent client turn has no valence, an earlier client turn supplies the baseline.
    Only the immediately preceding client text and the current provider text are
    predictor inputs; future text and valence are never features.
    """
    rows, total, skipped = [], 0, 0
    for dialogue in dialogues:
        turns = dialogue["turns"]
        roles = [role_of(t) for t in turns]
        clients = [i for i, role in enumerate(roles) if role == "user"]
        for index, role in enumerate(roles):
            if role != "assistant":
                continue
            total += 1
            previous = [i for i in clients if i < index]
            valid_previous = [i for i in previous if valid_valence(turns[i].get(valence_field)) is not None]
            future = [i for i in clients if i > index][:within]
            baseline = turns[valid_previous[-1]].get(valence_field) if valid_previous else None
            target = discounted_target(baseline, [turns[i].get(valence_field) for i in future],
                                       within=within, horizon=horizon, gamma=gamma)
            if target is None or index == 0 or roles[index - 1] != "user":
                skipped += 1
                continue
            used = [i for i in future if valid_valence(turns[i].get(valence_field)) is not None][:horizon]
            rows.append({
                "sample_id": f"{dialogue['dialogue_id']}:{index}",
                "dialogue_id": dialogue["dialogue_id"],
                "context": [{"speaker": "Client", "utterance": turns[index - 1]["text"]},
                            {"speaker": "Support provider", "utterance": turns[index]["text"]}],
                "delayed_valence_reward": target,
                "target_metadata": {"within": within, "horizon": horizon, "gamma": gamma,
                                    "valence_field": valence_field, "baseline_turn_index": valid_previous[-1],
                                    "future_turn_indices": used},
            })
    return rows, {"provider_turns": total, "retained": len(rows), "missing_context_or_target": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build discounted delayed-valence targets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window", choices=["within4-next2", "within5-next3"], default="within5-next3")
    parser.add_argument("--valence-field", default="valence")
    args = parser.parse_args()
    within, horizon = (4, 2) if args.window == "within4-next2" else (5, 3)
    rows, summary = build_target_rows(
        read_jsonl(args.input), valence_field=args.valence_field,
        within=within, horizon=horizon, gamma=0.7,
    )
    write_jsonl(args.output, rows)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
