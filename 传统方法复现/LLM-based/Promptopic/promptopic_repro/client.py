from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class LlmResult:
    text: str
    raw: Dict[str, Any]


class ChatApi:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        retry_limit: int,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_limit = retry_limit
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def ask(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> LlmResult:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error = None
        for attempt in range(self.retry_limit):
            try:
                resp = self.client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()
                return LlmResult(text=body["choices"][0]["message"]["content"], raw=body)
            except Exception as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"LLM call failed after retries: {last_error}")


def parse_topic_list(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1].replace("'", '"'))
            return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    for sep in ["\n", ",", ";"]:
        if sep in cleaned:
            return [item.strip(" -'\"") for item in cleaned.split(sep) if item.strip(" -'\"")]
    return [cleaned.strip(" -'\"")]

