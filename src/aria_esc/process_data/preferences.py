from __future__ import annotations

import argparse
import json
import math
import re
from html import unescape
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

from aria_esc.utils.io import read_jsonl, write_jsonl


@dataclass(frozen=True)
class MiningConfig:
    positive_score: float = 0.75
    positive_lcb: float = 0.15
    human_positive_score: float = 0.75
    human_positive_lcb: float = 0.15
    generated_to_human_ratio: float = 0.25
    winner_loser_score_margin: float = 0.10
    winner_loser_lcb_margin: float = 0.12
    reference_score_margin: float = 0.05
    reference_lcb_margin: float = 0.05
    max_rejected_per_winner: int = 2
    # The published deterministic sanity rule removes responses with fewer than 3 words.
    min_words: int = 3
    max_words: int = 120
    max_chars: int = 1200
    beta: float = 0.5
    eta: float = 0.35
    rank_disagreement_weight: float = 0.5
    visual_rescue_uncertainty_discount: float = 0.04
    visual_rescue_blend: float = 0.25
    audio_ambiguity_threshold: float = 0.18
    visual_rescue_confidence_threshold: float = 0.34
    conflict_uncertainty_penalty: float = 0.10

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid finite range for {name}")
        for name in ("positive_score", "human_positive_score", "visual_rescue_blend",
                     "audio_ambiguity_threshold", "visual_rescue_confidence_threshold"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("max_rejected_per_winner", "min_words", "max_words", "max_chars"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("at least one score is required")
    return sum(values) / len(values)


def _std(values: Iterable[float]) -> float:
    values = list(values)
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def rank_borda(candidates: list[dict], member_names: list[str]) -> dict[str, float]:
    """Return a [0,1] mean within-prompt Borda score for each candidate."""
    if not candidates or not member_names:
        raise ValueError("Candidates and predictor members must be nonempty")
    if len({str(c["candidate_id"]) for c in candidates}) != len(candidates):
        raise ValueError("candidate_id must be unique within a context")
    totals: dict[str, list[float]] = defaultdict(list)
    n = len(candidates)
    for member in member_names:
        if any(not math.isfinite(float(c["scores"][member])) or not 0 <= float(c["scores"][member]) <= 1
               for c in candidates):
            raise ValueError("Member scores must be calibrated probabilities in [0, 1]")
        ranked = sorted(candidates, key=lambda row: float(row["scores"][member]), reverse=True)
        start = 0
        while start < n:
            end = start + 1
            while end < n and float(ranked[end]["scores"][member]) == float(ranked[start]["scores"][member]):
                end += 1
            average_rank = (start + end - 1) / 2
            value = (n - 1 - average_rank) / (n - 1) if n > 1 else 0.5
            for candidate in ranked[start:end]:
                totals[str(candidate["candidate_id"])].append(value)
            start = end
    return {candidate_id: _mean(scores) for candidate_id, scores in totals.items()}


def route_multimodal_score(
    audio_score: float,
    visual_score: float | None,
    config: MiningConfig,
    *,
    raw_uncertainty: float,
    rank_disagreement: float,
) -> dict:
    """Audio-primary conservative routing with limited visual rescue."""
    audio_confidence = abs(2 * audio_score - 1)
    routed = audio_score
    route = "audio_primary"
    penalty = 0.0
    if visual_score is not None:
        visual_confidence = abs(2 * visual_score - 1)
        if (
            (audio_score - 0.5) * (visual_score - 0.5) >= 0
            and audio_confidence <= config.audio_ambiguity_threshold
            and visual_confidence >= config.visual_rescue_confidence_threshold
        ):
            routed = (1 - config.visual_rescue_blend) * audio_score + config.visual_rescue_blend * visual_score
            route = "visual_rescue"
        if (audio_score - 0.5) * (visual_score - 0.5) < 0:
            penalty = config.conflict_uncertainty_penalty
            route = "audio_fallback_conflict"
    uncertainty = raw_uncertainty + config.rank_disagreement_weight * rank_disagreement
    if route == "visual_rescue":
        uncertainty = max(0.0, uncertainty - config.visual_rescue_uncertainty_discount)
    uncertainty += penalty
    disagreement = abs(audio_score - visual_score) if visual_score is not None else 0.0
    lcb = (2 * routed - 1) - config.beta * uncertainty - config.eta * disagreement
    return {"score": routed, "lcb": lcb, "route": route,
            "audio_borda": audio_score, "visual_borda": visual_score,
            "raw_uncertainty": raw_uncertainty, "rank_disagreement": rank_disagreement,
            "uncertainty": uncertainty, "cross_modal_disagreement": disagreement}


_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")


def sanitize_response(text: str) -> tuple[str, str | None]:
    """Clean recoverable HTML/markup and reject unrecoverable markup artifacts."""
    value = unescape(str(text)).strip()
    if not _HTML_TAG_RE.search(value):
        return value, None
    cleaned = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    cleaned = re.sub(r"</p\s*>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", unescape(cleaned)).strip()
    if not cleaned or _HTML_TAG_RE.search(cleaned) or re.search(r"(?:<|>)\s*$", cleaned):
        return value, "unrecoverable_markup"
    return cleaned, None


def deterministic_sanity_check(text: str, config: MiningConfig) -> tuple[bool, str, str | None]:
    """Apply the published deterministic response sanity filter."""
    cleaned, markup_error = sanitize_response(text)
    if markup_error:
        return False, cleaned, markup_error
    words = cleaned.split()
    if len(words) < config.min_words:
        return False, cleaned, "too_short"
    if len(words) > config.max_words:
        return False, cleaned, "too_long_words"
    if len(cleaned) > config.max_chars:
        return False, cleaned, "too_long_chars"
    normalized = re.sub(r"\s+", " ", cleaned.lower()).split()
    trigrams = [tuple(normalized[i : i + 3]) for i in range(max(0, len(normalized) - 2))]
    if any(trigrams.count(item) >= 3 for item in set(trigrams)):
        return False, cleaned, "repeated_3gram"
    return True, cleaned, None


def score_sample(sample: dict, config: MiningConfig) -> dict:
    candidates = [dict(candidate) for candidate in sample["candidates"]]
    audio_members = list(sample["members"]["audio"])
    visual_members = list(sample["members"].get("visual", []))
    audio_borda = rank_borda(candidates, audio_members)
    visual_borda = rank_borda(candidates, visual_members) if visual_members else {}
    member_ranks = {m: rank_borda(candidates, [m]) for m in audio_members}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        raw_uncertainty = _mean(1 - abs(2 * float(candidate["scores"][m]) - 1) for m in audio_members)
        rank_disagreement = _std(member_ranks[m][candidate_id] for m in audio_members)
        audio, visual = audio_borda[candidate_id], visual_borda.get(candidate_id)
        candidate.update(route_multimodal_score(audio, visual, config,
            raw_uncertainty=raw_uncertainty, rank_disagreement=rank_disagreement))
    return {**sample, "candidates": candidates}


def mine_sample(sample: dict, config: MiningConfig) -> list[dict]:
    """Return local human/generated pairs; use mine_dataset to enforce the global cap."""
    scored = score_sample(sample, config)
    references = [item for item in scored["candidates"] if item.get("kind") == "reference"]
    if len(references) != 1:
        raise ValueError("Each context requires exactly one observed human reference")
    reference = references[0]
    decision = (reference.get("admissibility") or {}).get("decision")
    if decision not in {"PASS", "UNCERTAIN", "CLEAR_FAIL"}:
        raise ValueError("Human reference requires an admissibility decision; missing is not PASS")
    sane, cleaned, _ = deterministic_sanity_check(str(reference["text"]), config)
    if not sane or decision == "CLEAR_FAIL":
        return []
    reference["text"] = cleaned
    generated = [item for item in scored["candidates"] if item.get("kind") == "generated"]
    # The sanity filter defines the human-anchor pool only (Appendix C).
    eligible = [
        item for item in generated
        if item["score"] >= config.positive_score
        and item["lcb"] >= config.positive_lcb
        and item["score"] > reference["score"]
        and item["lcb"] > reference["lcb"]
        and item["score"] - reference["score"] >= config.reference_score_margin
        and item["lcb"] - reference["lcb"] >= config.reference_lcb_margin
    ]
    winners = []
    if reference["score"] >= config.human_positive_score and reference["lcb"] >= config.human_positive_lcb:
        winners.append(reference)
    if eligible:
        winners.append(sorted(eligible, key=lambda item: (-item["lcb"], -item["score"], str(item["candidate_id"])))[0])
    rows = []
    for winner in winners:
        losers = []
        for item in generated:
            if (item["candidate_id"] != winner["candidate_id"]
                    and item["text"] != winner["text"]
                    and winner["score"] > item["score"]
                    and winner["lcb"] > item["lcb"]
                    and winner["score"] - item["score"] >= config.winner_loser_score_margin
                    and winner["lcb"] - item["lcb"] >= config.winner_loser_lcb_margin):
                losers.append(item)
        losers.sort(key=lambda item: (item["lcb"], item["score"], str(item["candidate_id"])))
        for loser in losers[: config.max_rejected_per_winner]:
            rows.append({
                "sample_id": sample["sample_id"],
                "prompt": sample["prompt"],
                "chosen": winner["text"],
                "rejected": loser["text"],
                "metadata": {
                    "winner_id": winner["candidate_id"],
                    "positive_source": "human" if winner["kind"] == "reference" else "generated",
                    "admissibility": decision,
                    "score_margin": winner["score"] - loser["score"],
                    "lcb_margin": winner["lcb"] - loser["lcb"],
                    "uncertainty": winner["uncertainty"],
                    "rank_disagreement": winner["rank_disagreement"],
                    "cross_modal_disagreement": winner["cross_modal_disagreement"],
                    "rejected_id": loser["candidate_id"],
                    "winner_score": winner["score"],
                    "rejected_score": loser["score"],
                    "winner_lcb": winner["lcb"],
                    "rejected_lcb": loser["lcb"],
                    "route": winner["route"],
                },
            })
    return rows


def load_mining_config(path: str | None) -> MiningConfig:
    settings = json.loads(Path(path).read_text()) if path else {}
    unknown = set(settings) - {field.name for field in fields(MiningConfig)}
    if unknown:
        raise ValueError(f"Unknown MiningConfig fields: {sorted(unknown)}")
    return MiningConfig(**settings)


def mine_dataset(samples: list[dict], config: MiningConfig) -> tuple[list[dict], dict]:
    if len({str(s["sample_id"]) for s in samples}) != len(samples):
        raise ValueError("sample_id must be unique across contexts")
    rows = [row for sample in samples for row in mine_sample(sample, config)]
    human = [r for r in rows if r["metadata"]["positive_source"] == "human"]
    generated = [r for r in rows if r["metadata"]["positive_source"] == "generated"]
    generated.sort(key=lambda r: (-r["metadata"]["lcb_margin"], -r["metadata"]["score_margin"],
                                   str(r["sample_id"]), str(r["metadata"]["winner_id"]),
                                   str(r["metadata"]["rejected_id"])))
    cap = math.ceil(len(human) * config.generated_to_human_ratio)
    selected = human + generated[:cap]
    selected.sort(key=lambda r: (str(r["sample_id"]), str(r["metadata"]["winner_id"]), str(r["metadata"]["rejected_id"])))
    return selected, {"human_anchors": len({str(r["sample_id"]) for r in human}),
                      "human_pairs": len(human), "generated_eligible_pairs": len(generated),
                      "generated_cap": cap, "generated_pairs": min(cap, len(generated)),
                      "total_pairs": len(selected)}


def main() -> None:
    from aria_esc.process_data.anchors import attach_human_anchors

    parser = argparse.ArgumentParser(description="Mine conservative DPO pairs from pre-scored candidates.")
    parser.add_argument("--input", required=True, help="Candidate JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="MiningConfig JSON")
    parser.add_argument("--human-anchors", required=True, help="Human-anchor JSONL")
    args = parser.parse_args()

    config = load_mining_config(args.config)
    samples = attach_human_anchors(read_jsonl(args.input), read_jsonl(args.human_anchors))
    pairs, summary = mine_dataset(samples, config)
    write_jsonl(args.output, pairs)
    Path(str(args.output) + ".summary.json").write_text(
        json.dumps({**summary, "config": asdict(config)}, indent=2)
    )
    print(f"Wrote {len(pairs)} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
