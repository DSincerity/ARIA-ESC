"""Compute the seven main-paper metrics from the three evaluator outputs."""
from collections import defaultdict


INTERVENTION_COMPONENT = "intervention_component"
GENUINE_PROGRESS = "genuine_progress"
OPENNESS_TRANSITION = "openness_transition"
EVALUATORS = {INTERVENTION_COMPONENT, GENUINE_PROGRESS, OPENNESS_TRANSITION}

FAMILIES = {
    "support": ("empathy", "reflection", "validation", "reassurance"),
    "exploration": ("exploratory_question",),
    "action": ("advice_or_suggestion", "action_planning"),
    "meaning_shift": ("interpretation", "reframing"),
    "teaching": ("psychoeducation",),
}
MIN_CLOSING_CLIENT_TURN = 14
DIRECTIVE = FAMILIES["action"] + FAMILIES["meaning_shift"] + FAMILIES["teaching"]


def require_bool(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean, not {value!r}")
    return value


def intervention_flags(result):
    components = {
        key: require_bool(result[key], key)
        for family in FAMILIES.values()
        for key in family
    }
    families = {
        name: any(components[key] for key in family)
        for name, family in FAMILIES.items()
    }
    directive_count = sum(components[key] for key in DIRECTIVE)
    dense = sum(families.values()) >= 3
    return {
        "action": families["action"],
        "over_directive": directive_count > 0
        and (families["action"] or directive_count >= 2 or dense),
        "support_only": families["support"]
        and not families["exploration"]
        and directive_count == 0,
    }


def progress_flags(result):
    for key in ("genuine_progress", "surface_compliance"):
        if type(result.get(key)) is not int or not 0 <= result[key] <= 3:
            raise ValueError(f"{key} must be an integer in [0, 3]")
    allowed = {"progress", "compliance_only", "resistance", "mixed", "no_change", "regression"}
    if result.get("change_type") not in allowed:
        raise ValueError("Invalid change_type")
    return {
        "gp": result["genuine_progress"],
        "gp_sc": result["genuine_progress"] - result["surface_compliance"],
        "sc_only": result["change_type"] == "compliance_only",
    }


def openness_flags(result):
    return {
        "is_shifting_point": require_bool(result.get("is_shifting_point"), "is_shifting_point"),
        "is_closing": require_bool(result.get("is_closing", False), "is_closing"),
    }


RESULT_VALIDATORS = {
    INTERVENTION_COMPONENT: intervention_flags,
    GENUINE_PROGRESS: progress_flags,
    OPENNESS_TRANSITION: openness_flags,
}


def second_consecutive_closing(client_indices, closing_indices, min_client_turn=MIN_CLOSING_CLIENT_TURN):
    """Transcript index of the second of two consecutive closing client turns.

    Client turns are numbered by order (scenario opening = 0); only
    turns numbered ``min_client_turn`` or later can end the evaluated dialogue.
    """
    for number in range(max(1, min_client_turn), len(client_indices)):
        if client_indices[number] in closing_indices and client_indices[number - 1] in closing_indices:
            return client_indices[number]
    return None


def aggregate_paper_metrics(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Join eligible ``(C_t, T_t, C_next)`` transitions and aggregate them.

    The phase of ``T_t`` comes from its preceding client turn ``C_t``: transitions
    before the first detected openness signal are pre-open, and those from that
    point onward are open. The scenario opening is the client's initial situation
    and is pre-open. Missing evaluator outputs exclude a transition rather than
    contributing zero. Transitions after the second of two consecutive closing
    client turns (client turn 14 or later, counting the opening as 0) are excluded.
    """
    indexed = {}
    first_open = {}
    closing_turns = defaultdict(set)
    client_turns = defaultdict(set)
    for row in rows:
        evaluator = row.get("evaluator")
        key = (str(row["dialogue_id"]), evaluator, row["turn_index"])
        if key in indexed:
            raise ValueError(f"Duplicate evaluator annotation: {key}")
        if evaluator not in EVALUATORS or "result" not in row:
            raise ValueError("Expected completed named evaluator results")
        indexed[key] = row
        if evaluator == INTERVENTION_COMPONENT:
            intervention_flags(row["result"])
        elif evaluator == GENUINE_PROGRESS:
            progress_flags(row["result"])
            client_turns[str(row["dialogue_id"])].update(
                index for index in (row.get("baseline_turn_index"), row["turn_index"]) if index is not None
            )
        else:
            dialogue_id = str(row["dialogue_id"])
            openness = openness_flags(row["result"])
            if openness["is_shifting_point"]:
                first_open[dialogue_id] = min(
                    first_open.get(dialogue_id, row["turn_index"]), row["turn_index"]
                )
            client_turns[dialogue_id].add(row["turn_index"])
            if openness["is_closing"]:
                closing_turns[dialogue_id].add(row["turn_index"])

    closing_cutoff = {
        dialogue_id: cutoff
        for dialogue_id in closing_turns
        if (cutoff := second_consecutive_closing(sorted(client_turns[dialogue_id]), closing_turns[dialogue_id]))
        is not None
    }

    transitions = []
    excluded = defaultdict(int)
    for row in sorted(
        rows,
        key=lambda item: (str(item["dialogue_id"]), item["turn_index"], str(item.get("evaluator"))),
    ):
        if row.get("evaluator") != GENUINE_PROGRESS:
            continue
        dialogue_id = str(row["dialogue_id"])
        current = row["turn_index"]
        if dialogue_id in closing_cutoff and current > closing_cutoff[dialogue_id]:
            excluded["after_closing"] += 1
            continue
        baseline = row.get("baseline_turn_index")
        provider = row.get("provider_turn_index")
        if baseline is None or provider is None or not baseline < provider < current:
            raise ValueError("genuine-progress rows require baseline < provider < current indices")

        component = indexed.get((dialogue_id, INTERVENTION_COMPONENT, provider))
        if component is None or (dialogue_id, OPENNESS_TRANSITION, current) not in indexed:
            excluded["missing_component_or_openness"] += 1
            continue
        if component.get("baseline_turn_index") != baseline:
            raise ValueError("component and progress evaluators refer to different baseline turns")

        phase = (
            "open"
            if dialogue_id in first_open and baseline >= first_open[dialogue_id]
            else "pre_open"
        )
        transitions.append({
            "dialogue_id": dialogue_id,
            "baseline_turn_index": baseline,
            "provider_turn_index": provider,
            "current_turn_index": current,
            "phase": phase,
            **intervention_flags(component["result"]),
            **progress_flags(row["result"]),
        })

    pre_open = [row for row in transitions if row["phase"] == "pre_open"]
    open_phase = [row for row in transitions if row["phase"] == "open"]

    def mean(items, key):
        return sum(item[key] for item in items) / len(items) if items else None

    return {
        "metrics": {
            "GP": mean(transitions, "gp"),
            "GP-SC": mean(transitions, "gp_sc"),
            "SC-Only": mean(transitions, "sc_only"),
            "Action@Pre-open": mean(pre_open, "action"),
            "Over-Directive@Pre-open": mean(pre_open, "over_directive"),
            "Action@Open": mean(open_phase, "action"),
            "Support-Only@Open": mean(open_phase, "support_only"),
        },
        "denominators": {
            "eligible_transitions": len(transitions),
            "pre_open": len(pre_open),
            "open": len(open_phase),
        },
        "excluded_transitions": dict(excluded),
        "first_open_turn_index": first_open,
        "closing_cutoff_turn_index": closing_cutoff,
    }, transitions
