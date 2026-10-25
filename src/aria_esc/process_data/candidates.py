"""Sample policy candidates for each support-provider turn and score them with
calibrated delayed-valence predictors.

Input JSONL, one support-provider turn per line::

    {"sample_id": "<dialogue_id>:<turn_index>",   # same id as the human anchor
     "messages": [{"role": "user", "content": "<client>"}, ...],   # ends with the client turn
     "reference": "<observed human support-provider response>"}

Output JSONL, the input of ``aria_esc.process_data.preferences``::

    {"sample_id": "<id>", "prompt": "<chat-templated policy input>",
     "members": {"audio": ["<member>", ...], "visual": ["<member>", ...]},
     "candidates": [
       {"candidate_id": "reference", "kind": "reference", "text": "...", "scores": {"<member>": 0.63, ...}},
       {"candidate_id": "generated-0", "kind": "generated", "text": "...", "scores": {...}}, ...]}

The reference's admissibility decision is attached later from the human-anchor
file (``preferences --human-anchors``). Each predictor sees only the preceding
client utterance and one candidate response.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from aria_esc.train.valence_predictor import predict_probabilities
from aria_esc.utils.chat import validate_messages
from aria_esc.utils.io import read_jsonl, write_jsonl


def with_system_prompt(messages: list[dict], system_prompt: str) -> list[dict]:
    messages = validate_messages(messages)
    if messages[-1]["role"] != "user":
        raise ValueError("messages must end with the client turn")
    return messages if messages[0]["role"] == "system" else [{"role": "system", "content": system_prompt}, *messages]


def sample_candidates(model, tokenizer, prompt: str, *, num_candidates: int, temperature: float,
                      top_p: float, max_new_tokens: int) -> list[str]:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=True, temperature=temperature, top_p=top_p,
                                max_new_tokens=max_new_tokens, num_return_sequences=num_candidates,
                                pad_token_id=tokenizer.pad_token_id)
    start = encoded["input_ids"].shape[-1]
    return [tokenizer.decode(row[start:], skip_special_tokens=True).strip() for row in output]


def parse_member(value: str) -> tuple[str, str, str]:
    name, modality, path = value.split(":", 2)
    if modality not in {"audio", "visual"}:
        raise argparse.ArgumentTypeError("member modality must be audio or visual")
    return name, modality, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and score candidate responses for preference mining.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", required=True, help="SFT checkpoint or LoRA adapter directory")
    parser.add_argument("--member", action="append", type=parse_member, required=True,
                        help="NAME:audio|visual:PREDICTOR_DIR; repeat for every ensemble member")
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--system-prompt-file",
                        default=str(Path(__file__).resolve().parents[3] / "prompts" / "support_provider.txt"))
    args = parser.parse_args()
    if not any(modality == "audio" for _, modality, _ in args.member):
        parser.error("at least one audio predictor member is required")

    from aria_esc.backends import TransformersBackend

    model, tokenizer = TransformersBackend(args.policy)._load()
    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    samples = []
    for row in read_jsonl(args.input):
        messages = with_system_prompt(row["messages"], system_prompt)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts = [row["reference"], *sample_candidates(
            model, tokenizer, prompt, num_candidates=args.num_candidates, temperature=args.temperature,
            top_p=args.top_p, max_new_tokens=args.max_new_tokens)]
        samples.append({
            "sample_id": row["sample_id"], "prompt": prompt, "client_utterance": messages[-1]["content"],
            "candidates": [{"candidate_id": "reference" if i == 0 else f"generated-{i - 1}",
                            "kind": "reference" if i == 0 else "generated", "text": text, "scores": {}}
                           for i, text in enumerate(texts)],
        })
    del model

    members = {"audio": [], "visual": []}
    for name, modality, path in args.member:
        members[modality].append(name)
        pairs = [(sample, candidate) for sample in samples for candidate in sample["candidates"]]
        contexts = [[{"speaker": "Client", "utterance": sample["client_utterance"]},
                     {"speaker": "Support provider", "utterance": candidate["text"]}] for sample, candidate in pairs]
        for (_, candidate), probability in zip(pairs, predict_probabilities(path, contexts, device=args.device)):
            candidate["scores"][name] = probability
    write_jsonl(args.output, [{**{k: v for k, v in sample.items() if k != "client_utterance"}, "members": members}
                              for sample in samples])
    print(f"Scored {len(samples)} contexts with {len(args.member)} predictors")


if __name__ == "__main__":
    main()
