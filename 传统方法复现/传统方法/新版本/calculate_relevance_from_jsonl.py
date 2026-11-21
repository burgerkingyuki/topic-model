"""Calculate topic-document relevance for XLSX rows matched to source texts in JSONL.

Edit the CONFIG section, then run this file directly.

Expected XLSX columns by default:
- Paper_Title
- Second_Topics
- Author_Keywords
- Original_Representations

The JSONL file should contain one JSON object per line with a tagged `text` field, e.g.
{"id": "...", "text": "<title>Paper title<abstract>...", "label": "..."}
"""

import ast
import html
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================
# CONFIG: edit these values
# =========================
INPUT_XLSX = r"E:\主题建模\LSTFM主题建模\传统方法\新版本\your_input.xlsx"
SOURCE_JSONL = r"E:\主题建模\LSTFM主题建模\传统方法\新版本\dataset\文章数据\ROBOT.jsonl"
OUTPUT_XLSX = r""  # Empty means: <input_name>_relevance.xlsx
SUMMARY_JSON = r""  # Empty means: <input_name>_relevance_summary.json
MODEL_PATH = r"E:/主题建模/LSTFM主题建模/Model/all-MiniLM-L12-v2"

TITLE_COLUMN = "Paper_Title"
TOPIC_COLUMN = "Second_Topics"
# Use "Second_Topics" for topic labels, or "Original_Representations" for topic word lists.
# If TOPIC_COLUMN contains nested lists, each inner list is joined into one topic text.
TEXT_FIELD_IN_JSONL = "text"
ID_FIELD_IN_JSONL = "id"

SECTIONS = ["title", "abstract"]
MIN_SENTENCE_CHARS = 20
MAX_SENTENCES_PER_DOC = 12
BATCH_SIZE = 128


# Metric definitions for one paper:
# Let S_d be source sentences from JSONL text, Z_d be generated topics from XLSX.
# sim_matrix[i, j] = cos(E(sentence_i), E(topic_j))
#
# relevance_mean_best_topic    = mean_i max_j sim_matrix[i, j]
# relevance_max                = max_{i,j} sim_matrix[i, j]
# relevance_mean_all_pairs     = mean_{i,j} sim_matrix[i, j]
# relevance_mean_best_sentence = mean_j max_i sim_matrix[i, j]


def normalize_title(title):
    title = html.unescape(str(title or ""))
    title = re.sub(r"<[^>]+>", " ", title)
    title = unicodedata.normalize("NFKC", title).casefold()
    title = re.sub(r"[^0-9a-z]+", " ", title)
    return " ".join(title.split())


def parse_literal(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except (ValueError, SyntaxError):
        pass
    return [item.strip() for item in re.split(r"[;,|]", text) if item.strip()]


def topic_texts_from_value(value):
    parsed = parse_literal(value)
    topics = []
    for item in parsed:
        if isinstance(item, (list, tuple)):
            words = [str(word).strip() for word in item if str(word).strip()]
            if words:
                topics.append("; ".join(words))
        else:
            text = str(item).strip()
            if text:
                topics.append(text)
    return topics


def extract_tagged_section(text, tag):
    text = html.unescape(str(text or ""))
    pattern = rf"<{re.escape(tag)}[^>]*>(.*?)(?=<[A-Za-z][^>]*>|$)"
    match = re.search(pattern, text, re.I | re.S)
    if not match:
        return ""
    return re.sub(r"<[^>]+>", " ", match.group(1)).strip()


def extract_title_from_text(text):
    return extract_tagged_section(text, "title")


def split_sentences(text, min_chars):
    cleaned = html.unescape(str(text or ""))
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) >= min_chars]


def document_sentences(text, sections, min_chars, max_sentences):
    parts = []
    for section in sections:
        value = extract_tagged_section(text, section)
        if value:
            parts.append(value)
    if not parts:
        parts.append(re.sub(r"<[^>]+>", " ", html.unescape(str(text or ""))))
    return split_sentences(" ".join(parts), min_chars)[:max_sentences]


def load_jsonl_texts(jsonl_path):
    title_to_record = {}
    duplicate_titles = set()
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get(TEXT_FIELD_IN_JSONL, "")
            title = extract_title_from_text(text)
            normalized = normalize_title(title)
            if not normalized:
                continue
            record = {
                "jsonl_line": line_number,
                "jsonl_id": item.get(ID_FIELD_IN_JSONL, ""),
                "jsonl_title": title,
                "jsonl_text": text,
            }
            if normalized in title_to_record:
                duplicate_titles.add(normalized)
            else:
                title_to_record[normalized] = record
    return title_to_record, duplicate_titles


def encode_texts(model, texts, batch_size):
    if not texts:
        return np.empty((0, 0))
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def calculate_relevance(frame, title_to_record):
    matched_texts = []
    matched_titles = []
    matched_ids = []
    matched_lines = []
    match_status = []
    topic_groups = []
    sentence_groups = []

    for _, row in frame.iterrows():
        title = row.get(TITLE_COLUMN, "")
        record = title_to_record.get(normalize_title(title))
        topics = topic_texts_from_value(row.get(TOPIC_COLUMN, ""))
        topic_groups.append(topics)

        if record is None:
            matched_texts.append("")
            matched_titles.append("")
            matched_ids.append("")
            matched_lines.append("")
            match_status.append(False)
            sentence_groups.append([])
            continue

        matched_texts.append(record["jsonl_text"])
        matched_titles.append(record["jsonl_title"])
        matched_ids.append(record["jsonl_id"])
        matched_lines.append(record["jsonl_line"])
        match_status.append(True)
        sentence_groups.append(
            document_sentences(
                record["jsonl_text"],
                sections=SECTIONS,
                min_chars=MIN_SENTENCE_CHARS,
                max_sentences=MAX_SENTENCES_PER_DOC,
            )
        )

    model = SentenceTransformer(str(MODEL_PATH))
    all_topics = [topic for topics in topic_groups for topic in topics]
    all_sentences = [sentence for sentences in sentence_groups for sentence in sentences]
    topic_embeddings = encode_texts(model, all_topics, BATCH_SIZE)
    sentence_embeddings = encode_texts(model, all_sentences, BATCH_SIZE)

    topic_cursor = 0
    sentence_cursor = 0
    mean_best_topic_scores = []
    max_scores = []
    mean_all_pair_scores = []
    mean_best_sentence_scores = []
    topic_counts = []
    sentence_counts = []

    for topics, sentences in zip(topic_groups, sentence_groups):
        topic_count = len(topics)
        sentence_count = len(sentences)
        topic_counts.append(topic_count)
        sentence_counts.append(sentence_count)

        if topic_count == 0 or sentence_count == 0:
            mean_best_topic_scores.append(0.0)
            max_scores.append(0.0)
            mean_all_pair_scores.append(0.0)
            mean_best_sentence_scores.append(0.0)
            topic_cursor += topic_count
            sentence_cursor += sentence_count
            continue

        doc_topic_embeddings = topic_embeddings[topic_cursor : topic_cursor + topic_count]
        doc_sentence_embeddings = sentence_embeddings[sentence_cursor : sentence_cursor + sentence_count]
        sim_matrix = doc_sentence_embeddings @ doc_topic_embeddings.T

        mean_best_topic_scores.append(float(np.mean(np.max(sim_matrix, axis=1))))
        max_scores.append(float(np.max(sim_matrix)))
        mean_all_pair_scores.append(float(np.mean(sim_matrix)))
        mean_best_sentence_scores.append(float(np.mean(np.max(sim_matrix, axis=0))))

        topic_cursor += topic_count
        sentence_cursor += sentence_count

    result = frame.copy()
    result["matched"] = match_status
    result["matched_jsonl_title"] = matched_titles
    result["matched_jsonl_id"] = matched_ids
    result["matched_jsonl_line"] = matched_lines
    result["relevance_topic_count"] = topic_counts
    result["relevance_sentence_count"] = sentence_counts
    result["relevance_mean_best_topic"] = mean_best_topic_scores
    result["relevance_max"] = max_scores
    result["relevance_mean_all_pairs"] = mean_all_pair_scores
    result["relevance_mean_best_sentence"] = mean_best_sentence_scores
    return result


def main():
    input_path = Path(INPUT_XLSX)
    jsonl_path = Path(SOURCE_JSONL)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input XLSX does not exist: {input_path}")
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Source JSONL does not exist: {jsonl_path}")

    output_path = Path(OUTPUT_XLSX) if OUTPUT_XLSX else input_path.with_name(f"{input_path.stem}_relevance.xlsx")
    summary_path = Path(SUMMARY_JSON) if SUMMARY_JSON else input_path.with_name(f"{input_path.stem}_relevance_summary.json")

    frame = pd.read_excel(input_path)
    for column in (TITLE_COLUMN, TOPIC_COLUMN):
        if column not in frame.columns:
            raise ValueError(f"Input file has no required column: {column}")

    title_to_record, duplicate_titles = load_jsonl_texts(jsonl_path)
    result = calculate_relevance(frame, title_to_record)
    result.to_excel(output_path, index=False)

    matched = result["matched"].astype(bool)
    valid = matched & result["relevance_topic_count"].gt(0) & result["relevance_sentence_count"].gt(0)
    summary = {
        "input_xlsx": str(input_path),
        "source_jsonl": str(jsonl_path),
        "output_xlsx": str(output_path),
        "rows": int(len(result)),
        "jsonl_unique_titles": int(len(title_to_record)),
        "jsonl_duplicate_title_count": int(len(duplicate_titles)),
        "matched_count": int(matched.sum()),
        "unmatched_count": int((~matched).sum()),
        "match_accuracy": float(matched.mean()) if len(result) else 0.0,
        "valid_metric_rows": int(valid.sum()),
        "relevance_mean_best_topic": float(result.loc[valid, "relevance_mean_best_topic"].mean()) if valid.any() else 0.0,
        "relevance_max": float(result.loc[valid, "relevance_max"].mean()) if valid.any() else 0.0,
        "relevance_mean_all_pairs": float(result.loc[valid, "relevance_mean_all_pairs"].mean()) if valid.any() else 0.0,
        "relevance_mean_best_sentence": float(result.loc[valid, "relevance_mean_best_sentence"].mean()) if valid.any() else 0.0,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
