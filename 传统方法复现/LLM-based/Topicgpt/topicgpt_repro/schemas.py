from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Topic:
    topic_id: str
    label: str
    description: str
    source_doc_ids: List[str] = field(default_factory=list)
    frequency: int = 0

    def text(self) -> str:
        return f"{self.label}: {self.description}".strip()


@dataclass
class TopicAssignment:
    topic_id: str
    label: str
    description: str
    quote: str = ""
    reason: str = ""


@dataclass
class GenerationResult:
    topics: List[Topic]
    trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AssignmentResult:
    assignments: List[TopicAssignment]
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopicGPTRecord:
    assigned_topics: List[TopicAssignment]
    topic_pool_size: int
    run_meta: Dict[str, Any] = field(default_factory=dict)

