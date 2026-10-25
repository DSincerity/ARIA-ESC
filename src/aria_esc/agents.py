from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from aria_esc.backends import OpenAIBackend
from aria_esc.utils.chat import validate_messages


class DialogueAgent:
    """Prompt-owning dialogue agent with pluggable text-generation backend."""

    agent_name: ClassVar[str] = "dialogue_agent"
    prompt_filename: ClassVar[str]
    required_fields: ClassVar[tuple[str, ...]] = ()
    default_temperature: ClassVar[float] = 0.0
    default_max_tokens: ClassVar[int] = 256
    system_message: ClassVar[str] = "You output strict JSON only."

    def __init__(
        self,
        *,
        model: str,
        prompt_path: str | Path,
        base_url: str | None = None,
        served_model_name: str | None = None,
        api_key: str = "local",
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 3,
        retry_sleep_seconds: float = 0.5,
        timeout_seconds: float = 180.0,
        client: Any | None = None,
        backend: Any | None = None,
    ) -> None:
        self.model = model
        self.request_model = served_model_name or model
        self.prompt_path = Path(prompt_path)
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8").strip()
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = self.default_temperature if temperature is None else temperature
        self.max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
        self.retries = retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.timeout_seconds = timeout_seconds
        self.backend = backend or OpenAIBackend(
            self.request_model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            client=client,
        )

    @classmethod
    def from_prompt_dir(cls, prompt_dir: str | Path, **kwargs: Any) -> "DialogueAgent":
        return cls(prompt_path=Path(prompt_dir) / cls.prompt_filename, **kwargs)

    def render_prompt(self, **values: Any) -> str:
        missing = [field for field in self.required_fields if field not in values]
        if missing:
            raise ValueError(f"{self.agent_name} missing prompt fields: {missing}")
        rendered = self.prompt_template
        for field in self.required_fields:
            rendered = rendered.replace("{" + field + "}", str(values[field]))
        unresolved = [field for field in self.required_fields if "{" + field + "}" in rendered]
        if unresolved:
            raise ValueError(f"{self.agent_name} unresolved prompt fields: {unresolved}")
        return rendered

    def build_messages(self, **values: Any) -> list[dict[str, str]]:
        return validate_messages([
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.render_prompt(**values)},
        ])

    @staticmethod
    def parse_json_response(content: str) -> dict[str, Any]:
        text = str(content or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            if start < 0:
                raise
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        if not isinstance(parsed, dict):
            raise ValueError("Agent response must be a JSON object")
        return parsed

    def _invoke(self, messages: list[dict[str, str]], parser) -> Any:
        clean_messages = validate_messages(messages)
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                retry_max_tokens = self.max_tokens
                if attempt > 0 and self.base_url and self.max_tokens:
                    retry_max_tokens = min(self.max_tokens * (attempt + 1), 2048)
                content = self.backend.generate(
                    clean_messages,
                    temperature=self.temperature,
                    max_tokens=retry_max_tokens,
                )
                if not content.strip():
                    raise ValueError("Model server returned an empty completion")
                result = parser(content)
                return result
            except Exception as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(self.retry_sleep_seconds * (attempt + 1))
        raise RuntimeError(f"{self.agent_name} failed after {self.retries} attempts: {last_error}")

    def invoke_messages(self, messages: list[dict[str, str]], validate=None) -> dict[str, Any]:
        def parse(content: str) -> dict[str, Any]:
            result = self.parse_json_response(content)
            if validate is not None:
                validate(result)
            return result

        return self._invoke(messages, parse)

    def run(self, **values: Any) -> dict[str, Any]:
        return self.invoke_messages(self.build_messages(**values))

class SupportProviderAgent(DialogueAgent):
    """Generate a support-provider turn from dialogue history."""

    agent_name = "support_provider"
    prompt_filename = "support_provider.txt"
    default_temperature = 0.7
    default_max_tokens = 128

    def build_messages(self, *, recent_turns: list[dict[str, Any]]) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.prompt_template}]
        for turn in recent_turns:
            speaker = str(turn.get("speaker", turn.get("role", ""))).lower()
            role = "user" if speaker in {"client", "seeker", "user"} else "assistant"
            if speaker not in {"client", "seeker", "user", "support_provider", "provider", "assistant"}:
                raise ValueError(f"Unsupported rollout speaker: {speaker!r}")
            messages.append({"role": role, "content": str(turn.get("text", turn.get("content", "")))})
        if messages[-1]["role"] != "user":
            raise ValueError("Support-provider generation requires a final client turn")
        return validate_messages(messages)

    def respond(self, *, recent_turns: list[dict[str, Any]]) -> str:
        return self._invoke(self.build_messages(recent_turns=recent_turns), lambda text: text.strip())


class SeekerAgent(DialogueAgent):
    agent_name = "seeker"
    prompt_filename = "seeker.txt"
    required_fields = ("case_summary", "recent_turns", "latest_support_provider_response")
    default_temperature = 0.2
    default_max_tokens = 180

    @staticmethod
    def format_case_summary(case_summary: Any) -> str:
        if isinstance(case_summary, str):
            return case_summary
        return json.dumps(case_summary, ensure_ascii=False, indent=2)

    @staticmethod
    def format_recent_turns(recent_turns: str | list[dict[str, Any]], limit: int = 8) -> str:
        if isinstance(recent_turns, str):
            return recent_turns
        rows = []
        dialogue = [
            turn for turn in recent_turns
            if str(turn.get("speaker", turn.get("role", ""))).lower() != "system"
        ]
        for turn in dialogue[-limit:]:
            role = str(turn.get("speaker", turn.get("role", ""))).strip().lower()
            content = str(turn.get("text", turn.get("content", ""))).strip()
            if not content:
                continue
            if role in {"assistant", "support_provider", "provider"}:
                label = "Support provider"
            elif role in {"user", "client", "seeker"}:
                label = "Client"
            else:
                label = role.title() or "Unknown"
            rows.append(f"[{label}] {content}")
        return "\n".join(rows)

    def build_messages(
        self,
        *,
        case_summary: Any,
        recent_turns: str | list[dict[str, Any]],
        latest_support_provider_response: str,
    ) -> list[dict[str, str]]:
        return super().build_messages(
            case_summary=self.format_case_summary(case_summary),
            recent_turns=self.format_recent_turns(recent_turns),
            latest_support_provider_response=latest_support_provider_response,
        )

    def respond(
        self,
        *,
        case_summary: Any,
        recent_turns: str | list[dict[str, Any]],
        latest_support_provider_response: str,
    ) -> dict[str, Any]:
        return self.run(
            case_summary=case_summary,
            recent_turns=recent_turns,
            latest_support_provider_response=latest_support_provider_response,
        )


class InterventionComponentEvaluator(DialogueAgent):
    """Annotate the intervention components present in one provider turn."""

    agent_name = "intervention_component"
    prompt_filename = "judge_intervention_component.txt"
    required_fields = ("previous_client_utterance", "support_provider_response")
    default_temperature = 0.2  # same decoding as the seeker
    default_max_tokens = 400

    def evaluate(self, *, previous_client_utterance: str, support_provider_response: str) -> dict[str, Any]:
        return self.run(
            previous_client_utterance=previous_client_utterance,
            support_provider_response=support_provider_response,
        )


class GenuineProgressEvaluator(DialogueAgent):
    """Score client-owned progress and surface compliance for a transition."""

    agent_name = "genuine_progress"
    prompt_filename = "judge_genuine_progress.txt"
    required_fields = (
        "flow_context_before_baseline",
        "previous_client_utterance",
        "intervening_support_provider_response",
        "current_client_utterance",
    )
    default_temperature = 0.2  # same decoding as the seeker
    default_max_tokens = 500

    def evaluate(
        self,
        *,
        flow_context_before_baseline: str,
        previous_client_utterance: str,
        intervening_support_provider_response: str,
        current_client_utterance: str,
    ) -> dict[str, Any]:
        return self.run(
            flow_context_before_baseline=flow_context_before_baseline,
            previous_client_utterance=previous_client_utterance,
            intervening_support_provider_response=intervening_support_provider_response,
            current_client_utterance=current_client_utterance,
        )


class OpennessTransitionEvaluator(DialogueAgent):
    """Detect whether a client turn signals openness to change or help."""

    agent_name = "openness_transition"
    prompt_filename = "judge_openness_transition.txt"
    required_fields = (
        "previous_support_provider_utterance",
        "previous_client_utterance",
        "current_support_provider_utterance",
        "current_client_utterance",
    )
    default_temperature = 0.2  # same decoding as the seeker
    default_max_tokens = 350

    def evaluate(
        self,
        *,
        previous_support_provider_utterance: str,
        previous_client_utterance: str,
        current_support_provider_utterance: str,
        current_client_utterance: str,
    ) -> dict[str, Any]:
        return self.run(
            previous_support_provider_utterance=previous_support_provider_utterance,
            previous_client_utterance=previous_client_utterance,
            current_support_provider_utterance=current_support_provider_utterance,
            current_client_utterance=current_client_utterance,
        )


@dataclass
class DialogueAgentSuite:
    seeker: SeekerAgent
    intervention_component: InterventionComponentEvaluator
    genuine_progress: GenuineProgressEvaluator
    openness_transition: OpennessTransitionEvaluator

    @classmethod
    def from_prompt_dir(
        cls,
        *,
        prompt_dir: str | Path,
        model: str,
        base_url: str | None = None,
        served_model_name: str | None = None,
        api_key: str = "local",
        client: Any | None = None,
        seeker_temperature: float = 0.2,
        seeker_max_tokens: int = 180,
        intervention_component_max_tokens: int = 400,
        genuine_progress_max_tokens: int = 500,
        openness_transition_max_tokens: int = 350,
        retries: int = 3,
        retry_sleep_seconds: float = 0.5,
        timeout_seconds: float = 180.0,
    ) -> "DialogueAgentSuite":
        common = {
            "model": model,
            "base_url": base_url,
            "served_model_name": served_model_name,
            "api_key": api_key,
            "client": client,
            "retries": retries,
            "retry_sleep_seconds": retry_sleep_seconds,
            "timeout_seconds": timeout_seconds,
        }
        return cls(
            seeker=SeekerAgent.from_prompt_dir(
                prompt_dir,
                temperature=seeker_temperature,
                max_tokens=seeker_max_tokens,
                **common,
            ),
            intervention_component=InterventionComponentEvaluator.from_prompt_dir(
                prompt_dir, max_tokens=intervention_component_max_tokens, **common
            ),
            genuine_progress=GenuineProgressEvaluator.from_prompt_dir(
                prompt_dir, max_tokens=genuine_progress_max_tokens, **common
            ),
            openness_transition=OpennessTransitionEvaluator.from_prompt_dir(
                prompt_dir, max_tokens=openness_transition_max_tokens, **common
            ),
        )

    def evaluators(self) -> dict[str, DialogueAgent]:
        return {
            "intervention_component": self.intervention_component,
            "genuine_progress": self.genuine_progress,
            "openness_transition": self.openness_transition,
        }
