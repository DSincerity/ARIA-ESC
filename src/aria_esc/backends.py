from __future__ import annotations

from pathlib import Path
from typing import Any

from aria_esc.utils.chat import validate_messages


def _openai_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(getattr(item, "text", None), str):
                parts.append(item.text)
        if parts:
            return "".join(parts).strip()
    reasoning = getattr(message, "reasoning_content", None)
    return reasoning if isinstance(reasoning, str) else ""


class OpenAIBackend:
    """ChatGPT or an OpenAI-compatible model server such as vLLM."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str = "local",
        timeout_seconds: float = 180.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client = client

    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                max_retries=0,
            )
        return self._client

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        optional = {} if temperature is None else {"temperature": temperature}
        if max_tokens is not None:
            optional["max_tokens"] = max_tokens
        response = self.client().chat.completions.create(
            model=self.model,
            messages=validate_messages(messages),
            **optional,
        )
        if not response.choices:
            raise ValueError("Model server returned no completion choices")
        return _openai_message_text(response.choices[0].message)


class AnthropicBackend:
    """Claude through the Anthropic Messages API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 180.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._client = client

    def client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            kwargs = {"api_key": self.api_key, "timeout": self.timeout_seconds}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = Anthropic(**kwargs)
        return self._client

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        clean = validate_messages(messages)
        system = "\n\n".join(row["content"] for row in clean if row["role"] == "system")
        dialogue = [row for row in clean if row["role"] != "system"]
        optional = {} if temperature is None else {"temperature": temperature}
        response = self.client().messages.create(
            model=self.model,
            system=system,
            messages=dialogue,
            # The Messages API requires max_tokens.
            max_tokens=4096 if max_tokens is None else max_tokens,
            **optional,
        )
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


class TransformersBackend:
    """Local Hugging Face full checkpoint or PEFT/LoRA adapter."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        base_model: str | None = None,
        device_map: str = "auto",
        top_p: float | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.top_p = top_p
        self.base_model = base_model
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._tokenizer
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_config = self.checkpoint / "adapter_config.json"
        if adapter_config.exists():
            from peft import PeftConfig, PeftModel

            config = PeftConfig.from_pretrained(str(self.checkpoint))
            base_name = self.base_model or config.base_model_name_or_path
            model = AutoModelForCausalLM.from_pretrained(
                base_name, torch_dtype="auto", device_map=self.device_map
            )
            model = PeftModel.from_pretrained(model, str(self.checkpoint))
            tokenizer_source = (
                self.checkpoint
                if (self.checkpoint / "tokenizer_config.json").exists()
                else base_name
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(self.checkpoint), torch_dtype="auto", device_map=self.device_map
            )
            tokenizer_source = self.checkpoint

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.eval()
        self._model, self._tokenizer = model, tokenizer
        return model, tokenizer

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None,
        max_tokens: int,
    ) -> str:
        import torch

        model, tokenizer = self._load()
        encoded = tokenizer.apply_chat_template(
            validate_messages(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        device = model.get_input_embeddings().weight.device
        encoded = {name: value.to(device) for name, value in encoded.items()}
        input_length = encoded["input_ids"].shape[-1]
        settings = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            settings["temperature"] = temperature
            if self.top_p is not None:
                settings["top_p"] = self.top_p
        with torch.inference_mode():
            output = model.generate(**encoded, **settings)
        return tokenizer.decode(output[0, input_length:], skip_special_tokens=True).strip()
