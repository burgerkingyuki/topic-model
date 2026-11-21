from __future__ import annotations

import json
from typing import Iterable, List

from .schemas import Topic


def format_topics(topics: Iterable[Topic]) -> str:
    lines = []
    for topic in topics:
        lines.append(
            f"- {topic.topic_id} | {topic.label}: {topic.description}"
        )
    return "\n".join(lines)


def generation_prompt(
    document: str,
    demo_topics: List[Topic],
    topics: List[Topic],
) -> str:
    demo_block = format_topics(demo_topics) if demo_topics else "- None"
    topic_block = format_topics(topics) if topics else "- None"
    return f"""You are reproducing TopicGPT topic generation.

Given a document, a small set of demonstration topics, and the current topic pool, decide whether the document fits an existing topic or introduces a new generalizable topic.

Demonstration topics:
{demo_block}

Current topic pool:
{topic_block}

Granularity examples:
- Good broad topic: "Agriculture" — policies relating to agricultural practices and products.
- Avoid document-specific topic: "Repeal of agricultural export requirements in the SEAS Act".
- Good broad topic: "Trade" — the exchange of capital, goods, and services.
- Avoid document-specific topic: "Temporary duty suspension on Fluopyram mixtures".
- Good broad topic: "Employment Taxes" — taxation and reporting requirements related to employers.
- Avoid overly narrow topic: "Employer tax reporting requirement in one specific bill".

Document:
{document}

Return JSON only with this schema:
{{
  "action": "existing_topic" | "new_topic" | "none",
  "existing_topic_id": "T001" or null,
  "topic": {{"label": "...", "description": "..."}} or null,
  "support_quote": "...",
  "reason": "..."
}}

Rules:
- The current topic pool is authoritative and contains every topic available for this decision.
- First compare the document against every topic in the current topic pool.
- Prefer an existing topic whenever it reasonably covers the document.
- Create a new topic only when no current topic is sufficiently general and semantically appropriate.
- If using an existing topic, return its exact existing_topic_id and set topic to null.
- If creating a topic, create exactly one concise, generalizable topic and set existing_topic_id to null.
- Never invent an existing_topic_id and never create a duplicate of a current topic.
- A new topic must be broad enough to cover multiple documents and possible future subtopics.
- Do not use a bill name, act name, product name, person name, location, date, or one-off event as the topic label.
- Do not copy a distinctive phrase from the document as the topic label.
- Prefer the highest useful level of abstraction that still describes the document's main subject.
- Use exactly one main topic decision.
- support_quote must be copied from the document if possible.
"""


def refinement_prompt(pair_payload: List[dict]) -> str:
    return f"""You are reproducing TopicGPT topic refinement.

You will receive candidate topic pairs from the same level of the generated topic tree.
Merge topics only when they are paraphrases or near duplicates with substantially
overlapping meaning. Do not merge topics merely because they are related, co-occur,
belong to the same broad domain, or describe different aspects of the same document.
Do not merge parent-child topics or distinct subtopics.

Return JSON only:
{{
  "decisions": [
    {{
      "pair_id": 1,
      "merge": true,
      "merged_label": "...",
      "merged_description": "...",
      "reason": "..."
    }}
  ]
}}

Rules:
- Evaluate every supplied pair independently.
- Set merge to false when the pair is not a near duplicate.
- When merge is true, the merged label and description must remain generalizable.
- Only use the supplied pair_id values.
- If no pair should be merged, return {{"decisions": []}}.

Candidate pairs:
{json.dumps(pair_payload, ensure_ascii=False, indent=2)}
"""


def assignment_prompt(
    document: str,
    topics: List[Topic],
    max_assignments: int = 1,
    demonstrations: List[dict] | None = None,
) -> str:
    topic_block = format_topics(topics)
    demo_blocks = []
    for index, demo in enumerate(demonstrations or [], start=1):
        demo_document = str(demo.get("document", "")).strip()
        demo_assignments = demo.get("assignments") or []
        if demo_document and demo_assignments:
            demo_blocks.append(
                f"Example {index}:\n"
                f"Document:\n{demo_document}\n"
                f"Assignments:\n{json.dumps(demo_assignments, ensure_ascii=False)}"
            )
    examples_block = "\n\n".join(demo_blocks) if demo_blocks else "None provided."
    return f"""You are reproducing TopicGPT topic assignment.

Assign the document to the most relevant topic(s) from the provided refined topic tree.
The topic tree is fixed during this stage.

Refined topic tree:
{topic_block}

Demonstrations:
{examples_block}

Document:
{document}

Return JSON only:
{{
  "assignments": [
    {{
      "topic_id": "T001",
      "label": "...",
      "description": "...",
      "quote": "...",
      "reason": "..."
    }}
  ]
}}

Rules:
- Only use topic_id values that exist in the refined topic tree.
- Never create, rename, paraphrase, or invent a topic.
- Never output a topic label that is not represented by the selected topic_id.
- The description may explain why the topic applies to this document, but must remain grounded in the document.
- quote must be copied verbatim from the document; never fabricate a quote.
- Return at most {max_assignments} assignment(s).
- If the response is invalid, return an empty assignments list rather than inventing a result.
"""


def assignment_correction_prompt(
    document: str,
    topics: List[Topic],
    max_assignments: int,
    invalid_response: dict,
    errors: List[str],
) -> str:
    return f"""Self-correction for TopicGPT topic assignment.

The previous assignment was invalid. Reassign the document using only the
fixed refined topic tree below.

Refined topic tree:
{format_topics(topics)}

Document:
{document}

Previous response:
{json.dumps(invalid_response, ensure_ascii=False, indent=2)}

Detected errors:
{json.dumps(errors, ensure_ascii=False)}

Return JSON only:
{{
  "assignments": [
    {{
      "topic_id": "an ID copied exactly from the refined topic tree",
      "label": "the exact label for that topic",
      "description": "why the topic applies to this document",
      "quote": "a verbatim substring copied from the document",
      "reason": "brief evidence-based reason"
    }}
  ]
}}

Rules:
- Correct every detected error.
- Never create, rename, or paraphrase a topic.
- Use at most {max_assignments} assignment(s).
- The quote must be a verbatim substring of the document.
- If no valid assignment can be made, return an empty assignments list.
"""
