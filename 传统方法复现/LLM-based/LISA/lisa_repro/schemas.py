from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GeneratedTopicWord:
    doc_id: str
    topic_word: str
    description: str
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterTopic:
    cluster_id: int
    topic_word: str
    description: str
    source_topic_ids: List[str] = field(default_factory=list)
    frequency: int = 0


@dataclass
class AssignmentDecision:
    topic_id: int
    topic_word: str
    description: str
    mode: str
    quote: str = ""
    reason: str = ""


@dataclass
class PipelineArtifact:
    topics: List[ClusterTopic]
    topic_words: List[GeneratedTopicWord]
    doc_cluster_ids: List[int]
    topic_cluster_ids: List[int]
    doc_cluster_map: Dict[int, int]
    topic_cluster_map: Dict[int, int]

