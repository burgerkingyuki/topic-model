"""Standalone topic-document relevance calculator for XLSX inputs.

Edit the CONFIG section below, then run this file directly.

Input columns expected:
- refined_responses: generated topic list, e.g. ['Quantum Information', 'Photonics']
- text: tagged paper text containing <title> and <abstract> sections. HTML-escaped tags are supported.

Optional columns are preserved in the output, including Author_Keywords, extracted_title,
and matched_xlsx_title.
"""
# relevance_mean_best_topic：每个句子找最相关主题，偏“论文内容是否被主题覆盖”
# relevance_mean_best_sentence：每个主题找最相关句子，偏“每个主题是否有文本支撑”
# relevance_max：整篇论文最强匹配，偏“是否存在强相关点”
# relevance_mean_all_pairs：所有句子和主题直接平均，最保守

import ast
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================
# CONFIG: edit these values
# =========================
INPUT_XLSX = r"E:\主题建模\LSTFM主题建模\一级主题生成(正式版)\第三步表格\Q_500_1108\Q_500_topic_1_1108.xlsx"
OUTPUT_XLSX = r"E:\主题建模\LSTFM主题建模\一级主题生成(正式版)\第四步计算指标\relevance\Q_500_topic_1_1108.xlsx"  # Empty means: <input_name>_relevance.xlsx
MODEL_PATH = r"E:/主题建模/LSTFM主题建模/Model/all-MiniLM-L12-v2"

TOPIC_COLUMN = "refined_responses"
TEXT_COLUMN = "text"
SECTIONS = ["title", "abstract"]
MIN_SENTENCE_CHARS = 20
MAX_SENTENCES_PER_DOC = 12
BATCH_SIZE = 128


# Calculation example for one paper:
# Sentences S = [s1, s2], generated topics Z = [z1, z2, z3]
# Suppose the cosine similarity matrix is:
#
#              z1     z2     z3
#       s1    0.80   0.30   0.55
#       s2    0.40   0.70   0.20
#
# 1) relevance_mean_best_topic:
#    For each sentence, select the most relevant topic, then average over sentences.
#    s1 best = max(0.80, 0.30, 0.55) = 0.80
#    s2 best = max(0.40, 0.70, 0.20) = 0.70
#    relevance_mean_best_topic = (0.80 + 0.70) / 2 = 0.75
#
# 2) relevance_max:
#    Select the strongest sentence-topic match in the paper.
#    relevance_max = max(all six values) = 0.80
#
# 3) relevance_mean_all_pairs:
#    Average every sentence-topic pair.
#    relevance_mean_all_pairs = (0.80 + 0.30 + 0.55 + 0.40 + 0.70 + 0.20) / 6 = 0.4917
#
# 4) relevance_mean_best_sentence:
#    For each generated topic, select the most relevant paper sentence, then average over topics.
#    z1 best = max(0.80, 0.40) = 0.80
#    z2 best = max(0.30, 0.70) = 0.70
#    z3 best = max(0.55, 0.20) = 0.55
#    relevance_mean_best_sentence = (0.80 + 0.70 + 0.55) / 3 = 0.6833
#
# Recommended main metric for multiple topics: relevance_mean_best_topic.
# It avoids diluting a valid multi-topic output just because each topic naturally
# matches only part of the abstract.


def parse_list(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [item.strip() for item in re.split(r"[;,|]", text) if item.strip()]


def extract_tagged_section(text, tag):
    text = html.unescape(str(text or ""))
    pattern = rf"<{re.escape(tag)}[^>]*>(.*?)(?=<[A-Za-z][^>]*>|$)"
    match = re.search(pattern, text, re.I | re.S)
    if not match:
        return ""
    return re.sub(r"<[^>]+>", " ", match.group(1)).strip()


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


def calculate_relevance(frame, model_path, topic_column, text_column, sections, min_chars, max_sentences, batch_size):
    topics_per_doc = [parse_list(value) for value in frame[topic_column]]
    sentences_per_doc = [
        document_sentences(value, sections=sections, min_chars=min_chars, max_sentences=max_sentences)
        for value in frame[text_column]
    ]

    model = SentenceTransformer(str(model_path))

    all_topics = [topic for topics in topics_per_doc for topic in topics]
    all_sentences = [sentence for sentences in sentences_per_doc for sentence in sentences]

    topic_embeddings = encode_texts(model, all_topics, batch_size)
    sentence_embeddings = encode_texts(model, all_sentences, batch_size)

    topic_cursor = 0
    sentence_cursor = 0
    mean_best_topic_scores = []
    max_scores = []
    mean_all_pair_scores = []
    mean_best_sentence_scores = []
    topic_count_column = []
    sentence_count_column = []

    for topics, sentences in zip(topics_per_doc, sentences_per_doc):
        topic_count = len(topics)
        sentence_count = len(sentences)
        topic_count_column.append(topic_count)
        sentence_count_column.append(sentence_count)

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
    result["relevance_mean_best_topic"] = mean_best_topic_scores
    result["relevance_max"] = max_scores
    result["relevance_mean_all_pairs"] = mean_all_pair_scores
    result["relevance_mean_best_sentence"] = mean_best_sentence_scores
    result["relevance_topic_count"] = topic_count_column
    result["relevance_sentence_count"] = sentence_count_column
    return result


def main():
    input_path = Path(INPUT_XLSX)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path = Path(OUTPUT_XLSX) if OUTPUT_XLSX else input_path.with_name(f"{input_path.stem}_relevance.xlsx")

    frame = pd.read_excel(input_path)
    for column in (TOPIC_COLUMN, TEXT_COLUMN):
        if column not in frame.columns:
            raise ValueError(f"Input file has no required column: {column}")

    result = calculate_relevance(
        frame=frame,
        model_path=MODEL_PATH,
        topic_column=TOPIC_COLUMN,
        text_column=TEXT_COLUMN,
        sections=SECTIONS,
        min_chars=MIN_SENTENCE_CHARS,
        max_sentences=MAX_SENTENCES_PER_DOC,
        batch_size=BATCH_SIZE,
    )
    result.to_excel(output_path, index=False)

    valid = result["relevance_topic_count"].gt(0) & result["relevance_sentence_count"].gt(0)
    summary = {
        "rows": int(len(result)),
        "valid_rows": int(valid.sum()),
        "relevance_mean_best_topic": float(result.loc[valid, "relevance_mean_best_topic"].mean()) if valid.any() else 0.0,
        "relevance_max": float(result.loc[valid, "relevance_max"].mean()) if valid.any() else 0.0,
        "relevance_mean_all_pairs": float(result.loc[valid, "relevance_mean_all_pairs"].mean()) if valid.any() else 0.0,
        "relevance_mean_best_sentence": float(result.loc[valid, "relevance_mean_best_sentence"].mean()) if valid.any() else 0.0,
        "output": str(output_path),
    }
    print(summary)


if __name__ == "__main__":
    main()
