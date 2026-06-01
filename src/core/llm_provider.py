import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Generator

class LLMProvider(ABC):
    """
    Abstract Base Class for LLM Providers.
    Supports OpenAI, Gemini, and Local models.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Produce a non-streaming completion.
        Returns:
            Dict containing:
            - content: The response text
            - usage: { 'prompt_tokens', 'completion_tokens' }
            - latency_ms: Response time
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """Produce a streaming completion."""
        pass

    @staticmethod
    def extract_json(response_text: str) -> str:
        """
        Helper method to extract JSON from markdown code blocks in LLM responses.
        Useful for Dev C when parsing Actions.
        """
        import re
        # Look for json blocks
        match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback to the original text (might already be raw json)
        return response_text.strip()
