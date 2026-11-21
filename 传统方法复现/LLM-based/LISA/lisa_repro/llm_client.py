from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class ChatResult:
    content: str
    raw: Dict[str, Any]


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: float = 120.0,
        retry_limit: int = 5,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_limit = retry_limit
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> ChatResult:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        last_error: Optional[Exception] = None
        for attempt in range(self.retry_limit):
            try:
                resp = self._client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return ChatResult(content=data["choices"][0]["message"]["content"], raw=data)
            except Exception as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    def chat_json(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> ChatResult:
        last_error: Optional[Exception] = None
        cur_messages = list(messages)
        for attempt in range(self.retry_limit):
            result = self.chat(cur_messages, model=model)
            try:
                parse_json_object(result.content)
                return result
            except Exception as exc:
                last_error = exc
                cur_messages = cur_messages + [
                    {"role": "user", "content": "The previous reply was invalid JSON. Return JSON only."}
                ]
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"JSON parsing failed after retries: {last_error}")


def parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Cannot parse JSON from response: {text[:500]}")
    return json.loads(text[start : end + 1])

