from __future__ import annotations

import json
from typing import Iterable, List


def generation_prompt(document: str) -> str:
    return f"""Given the following document, generate a topic word for the article, followed by a short description of the meaning of the topic.

Return JSON only with keys:
{{
  "topic_word": "...",
  "description": "..."
}}

Rules:
- The topic should be a single word or a short phrase of 2-3 words.
- The description should be no more than two sentences.

Document:
{document}
"""


def summarize_prompt(topic_list: Iterable[str]) -> str:
    return f"""Given a group of topic words, your task is to generate one general topic word that can describe as many topic words from this group as possible.
Just tell me the topic word without any explanation or context.
Topic words: {", ".join(list(topic_list))}
Your response:"""


def assignment_prompt(document: str, candidate_topics: List[dict]) -> str:
    topic_list = ", ".join(str(item.get("topic_word", "")) for item in candidate_topics)
    return f"""Given a document and several candidate topics, you should decide which topic the document belongs to. If the document is not related to any of the candidate topics, you should respond "None".
Return the topic that best describes the document.
Do not provide any explanation or context.
Document: {document}
Candidatetopics: {topic_list}
Your chosen topic:"""
