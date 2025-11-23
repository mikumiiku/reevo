import logging
from typing import Optional
from .openai import OpenAIClient
import os

logger = logging.getLogger(__name__)

class LMStudioClient(OpenAIClient):
    """
    LM Studio client that uses OpenAI-compatible API.
    LM Studio runs locally and provides an OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        # LM Studio doesn't require a real API key, but the OpenAI client expects one
        if api_key is None:
            api_key = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
        
        # Default LM Studio local server URL
        base_url = base_url or "http://127.0.0.1:1234/v1"

        super().__init__(model, temperature, base_url, api_key)
    
    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        assert n == 1, "LM Studio currently only supports n=1"
        response = self.client.chat.completions.create(
            model=self.model, 
            messages=messages, 
            temperature=temperature, 
            stream=False,
            max_tokens=4096,
            timeout=300,
        )
        return response.choices
