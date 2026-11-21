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
        max_tokens: int = 300,
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
                content = data["choices"][0]["message"]["content"]
                return ChatResult(content=content, raw=data)
            except Exception as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    def chat_json(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> ChatResult:
        last_error: Optional[Exception] = None
        for attempt in range(self.retry_limit):
            result = self.chat(messages, model=model)
            try:
                parse_json_object(result.content)
                return result
            except Exception as exc:
                last_error = exc
                messages = list(messages) + [
                    {
                        "role": "user",
                        "content": "The previous response was not valid JSON. Return JSON only.",
                    }
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
        raise ValueError(f"Cannot parse JSON from text: {text[:500]}")
    payload = text[start : end + 1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return json.loads(_repair_invalid_json_escapes(payload))


def _repair_invalid_json_escapes(payload: str) -> str:
    """
    Repair malformed escape sequences inside JSON string literals.

    Some LLMs emit single backslashes in text fields, which breaks strict JSON
    parsing with "Invalid \\escape". We preserve already-valid escapes and only
    double backslashes that start invalid escape sequences.
    """
    valid_simple = {'"', "\\", "/", "b", "f", "n", "r", "t"}
    result: List[str] = []
    in_string = False
    escaped = False
    i = 0
    while i < len(payload):
        ch = payload[i]
        if escaped:
            result.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\" and in_string:
            nxt = payload[i + 1] if i + 1 < len(payload) else ""
            if nxt == "u" and i + 5 < len(payload):
                hex_part = payload[i + 2 : i + 6]
                if all(c in "0123456789abcdefABCDEF" for c in hex_part):
                    result.append(ch)
                    i += 1
                    continue
            if nxt in valid_simple:
                result.append(ch)
                i += 1
                continue
            result.append("\\\\")
            i += 1
            continue
        result.append(ch)
        if ch == '"' and not escaped:
            backslashes = 0
            j = i - 1
            while j >= 0 and payload[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                in_string = not in_string
        i += 1
    return "".join(result)
