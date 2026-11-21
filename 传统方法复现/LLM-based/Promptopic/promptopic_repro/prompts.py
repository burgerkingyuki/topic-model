from __future__ import annotations

from typing import Dict, List


def topic_generation_messages(
    article: str,
    demonstrations: List[Dict],
    max_topics_per_document: int = 0,
) -> List[Dict[str, str]]:
    topic_instruction = (
        "Return exactly one high-level topic for this article."
        if max_topics_per_document == 1
        else "List the high-level topics in the following article."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are designated as an assistant that identifies and extracts "
                "high-level topics from articles. Avoid specific details and return "
                "unique topic names only. "
                + (
                    "Return exactly one topic in a one-element list."
                    if max_topics_per_document == 1
                    else ""
                )
            ),
        }
    ]
    for demo in demonstrations:
        messages.append(
            {
                "role": "user",
                "content": f"{topic_instruction}\nArticle: {demo['article']}\nTopics:",
            }
        )
        demo_topics = list(demo.get("topics") or [])
        if max_topics_per_document > 0:
            demo_topics = demo_topics[:max_topics_per_document]
        messages.append({"role": "assistant", "content": str(demo_topics)})
    messages.append(
        {
            "role": "user",
            "content": f"{topic_instruction}\nArticle: {article}\nTopics:",
        }
    )
    return messages


def pbm_merge_messages(topic: str, candidates: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You merge overlapping high-level topic names. Return only one existing candidate topic name, or Miscellaneous if no candidate matches.",
        },
        {
            "role": "user",
            "content": f"Topic to merge: {topic}\nCandidate topics: {', '.join(candidates)}\nChosen topic:",
        },
    ]


def word_filter_messages(topic: str, candidate_words: List[str], top_n: int) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You select representative topic words. Return only a comma-separated word list.",
        },
        {
            "role": "user",
            "content": f"Topic: {topic}\nCandidate words: {', '.join(candidate_words)}\nSelect the top {top_n} representative words:",
        },
    ]
