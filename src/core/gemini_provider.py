import os
import time
import google.generativeai as genai
from typing import Dict, Any, Optional, Generator
from src.core.llm_provider import LLMProvider

class GeminiProvider(LLMProvider):
    """
    LLM Provider for Google Gemini models.
    Uses system_instruction at model init time (the correct Gemini API pattern).
    """
    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None,
                 system_prompt: Optional[str] = None):
        super().__init__(model_name, api_key)
        genai.configure(api_key=self.api_key)
        # Pass system_instruction during model init — the correct Gemini pattern
        self._default_system = system_prompt
        init_kwargs = {}
        if system_prompt:
            init_kwargs["system_instruction"] = system_prompt
        self.model = genai.GenerativeModel(model_name, **init_kwargs)

    def _get_model_with_system(self, system_prompt: Optional[str]):
        """Return model configured with the given system_instruction."""
        if system_prompt and system_prompt != self._default_system:
            return genai.GenerativeModel(
                self.model_name,
                system_instruction=system_prompt
            )
        return self.model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()

        model = self._get_model_with_system(system_prompt)
        response = model.generate_content(prompt)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        content = response.text

        # usage_metadata may be absent on older SDK versions — handle gracefully
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            usage = {
                "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0),
                "completion_tokens": getattr(usage_meta, "candidates_token_count", 0),
                "total_tokens": getattr(usage_meta, "total_token_count", 0),
            }
        else:
            words = len(content.split())
            usage = {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": words,
                "total_tokens": len(prompt.split()) + words,
            }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "google",
            "model": self.model_name,
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        model = self._get_model_with_system(system_prompt)
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
