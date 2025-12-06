from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class LLMModel(ABC):
    """Abstract interface for a text-generation model used for RAG/inference."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        """Generate text for the given prompt.

        Args:
            prompt: The input prompt to condition on.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """


class OpenAILLM(LLMModel):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        try:
            import openai

            self._openai = openai
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("openai package is required for OpenAILLM") from exc

        if api_key:
            self._openai.api_key = api_key

        self.model = model

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        resp = self._openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()


class LocalLLM(LLMModel):
    """Local text-generation using Hugging Face `transformers` pipeline.

    This attempts to load a small model by default (distilgpt2). For instruction
    tuned models, point `model_name` to an appropriate checkpoint.
    """

    def __init__(self, model_name: str = "distilgpt2", device: Optional[int] = None, **kwargs):
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers is required for LocalLLM") from exc

        self.model_name = model_name
        self.device = device
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device=device if device is not None else -1,
            **kwargs,
        )

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        out = self._pipe(prompt, max_new_tokens=max_tokens, do_sample=temperature > 0.0, temperature=temperature)
        # pipeline returns list of dicts with 'generated_text'
        return out[0]["generated_text"][len(prompt) :].strip()


class DeepseekLLM(LLMModel):
    """Adapter for the Deepseek API.

    This is a lightweight wrapper assuming a JSON API endpoint that accepts
    `prompt`, `max_tokens`, and `temperature`. Adjust headers/URL to your
    Deepseek plan. Provide `api_key` via constructor.
    """

    DEFAULT_ENDPOINT = "https://api.deepseek.ai/v1/generate"

    def __init__(self, api_key: str, endpoint: Optional[str] = None):
        import requests

        self._requests = requests
        self.api_key = api_key
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: Dict = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        resp = self._requests.post(self.endpoint, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Try common keys used by model APIs
        if isinstance(data, dict):
            for key in ("text", "generated_text", "output", "response"):
                if key in data:
                    return data[key]
            # Some APIs return choices list
            choices = data.get("choices") or data.get("outputs")
            if choices and isinstance(choices, list) and len(choices) > 0:
                c0 = choices[0]
                if isinstance(c0, dict):
                    return c0.get("text") or c0.get("output") or str(c0)
                return str(c0)

        return str(data)
