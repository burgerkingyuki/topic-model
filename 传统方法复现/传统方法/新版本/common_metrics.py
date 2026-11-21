"""Metrics integrated from the four scripts in 重新计算指标."""

import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


# Matches CV_TP(bertopic_nmf_lda专用).py.
DOMAIN_COMMON_WORDS = set()
TOP_WORDS = 10
REFERENCE_CV_WINDOW_SIZE = 110
NULL_TOPIC_KEY = "__null_topic__"


def normalize_title(title):
    title = re.sub(r"<[^>]+>", " ", str(title or ""))
    title = title.lower()
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    return " ".join(title.split())


def extract_title(text):
    match = re.search(r"<title[^>]*>(.*?)(?:<abstract|<introduction|<|$)", text, re.I | re.S)
    return re.sub(r"<[^>]+>", " ", match.group(1)).strip() if match else ""


def extract_title_from_fulltext(text):
    if "<title>" in text:
        return text.split("<title>", 1)[1].split("<", 1)[0].strip()
    return ""

# Relevance helpers are adapted from NovBench-style source-to-output matching.
def extract_tagged_section(text, tag):
    pattern = rf"<{re.escape(tag)}[^>]*>(.*?)(?=<[A-Za-z][^>]*>|$)"
    match = re.search(pattern, str(text or ""), re.I | re.S)
    if not match:
        return ""
    return re.sub(r"<[^>]+>", " ", match.group(1)).strip()


def split_metric_sentences(text, min_chars=20):
    cleaned = re.sub(r"\[[^\]]*\]", " ", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) >= min_chars]


def document_relevance_sentences(text, sections=("title", "abstract"), min_chars=20, max_sentences=12):
    parts = []
    for section in sections:
        value = extract_tagged_section(text, section)
        if value:
            parts.append(value)
    if not parts:
        parts.append(re.sub(r"<[^>]+>", " ", str(text or "")))
    sentences = split_metric_sentences(" ".join(parts), min_chars=min_chars)
    return sentences[:max_sentences]


def preprocess_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
    tokens = [word.lower() for word in text.split() if len(word) > 2]
    return " ".join(tokens)


def safe_literal_eval(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def model_topics_from_jsonl(value):
    """Keep one JSONL topic string as one topic-list item, as in old XLSX inputs."""
    invalid = {"", "无主题关键词", "异常值主题", "(empty)", "文本过短，无法提取主题", "主题提取失败"}
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if text in invalid:
        return []
    if text.startswith("["):
        parsed = safe_literal_eval(text)
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


def record_native_topic_words(record):
    if "topic_words" in record:
        return split_keywords(record.get("topic_words"))
    return split_keywords(record.get("topics", ""))


def split_keywords(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in re.split(r"[;,|]", str(value)) if item.strip()]


def detect_column(columns, requested, candidates, label):
    if requested:
        if requested not in columns:
            raise ValueError(f"Metadata has no '{requested}' column for {label}.")
        return requested
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ValueError(f"Cannot detect the {label} column. Set it in config.json.")


def create_evaluation_table(model_name, model_jsonl, metadata_path, title_column=None, keywords_column=None):
    metadata = pd.read_excel(metadata_path)
    title_col = detect_column(metadata.columns, title_column, ["Article Title", "title"], "title")
    keyword_col = detect_column(
        metadata.columns,
        keywords_column,
        ["Author Keywords", "author_keywords", "keywords"],
        "author keyword",
    )
    title_to_keywords = {}
    for _, row in metadata.iterrows():
        normalized = normalize_title(row[title_col])
        if normalized:
            title_to_keywords[normalized] = split_keywords(row[keyword_col])

    rows = []
    unmatched = 0
    with open(model_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            title = extract_title(record.get("text", ""))
            normalized = normalize_title(title)
            if not normalized or normalized not in title_to_keywords:
                unmatched += 1
                continue
            rows.append(
                {
                    "model": model_name,
                    "title": title,
                    "true_keywords": title_to_keywords[normalized],
                    "topic_id": record.get("topic_id"),
                    "model_topics": model_topics_from_jsonl(record.get("topics", "")),
                    "native_topic_words": record_native_topic_words(record),
                }
            )
    return pd.DataFrame(rows), unmatched


def attach_full_text(records, input_file):
    title_to_fulltext = {}
    with open(input_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                full_text = item.get("text", "")
                title = extract_title_from_fulltext(full_text)
                if title:
                    title_to_fulltext[title.lower().strip()] = full_text
            except (json.JSONDecodeError, AttributeError):
                continue

    result = records.copy()
    result["metric_text"] = [
        title_to_fulltext.get(str(title).strip().lower(), title)
        for title in result["title"]
    ]
    return result


def build_corpus(records):
    corpus = []
    for _, row in records.iterrows():
        processed = preprocess_text(row["metric_text"])
        corpus.append(processed)
    return corpus


# CV_TP(bertopic_nmf_lda专用).py: sorted list, duplicates retained.
def cv_topic_key(topics_list):
    if not topics_list or not isinstance(topics_list, list):
        return "unknown_topic"
    cleaned = [str(word).strip().lower() for word in topics_list if isinstance(word, str) and word.strip()]
    return tuple(sorted(cleaned))


# kl_background(传统方法适用款).py: sorted set.
def kl_topic_key(topics_list):
    if not topics_list or not isinstance(topics_list, list):
        return ("unknown_topic",)
    cleaned = [str(word).strip().lower() for word in topics_list if str(word).strip()]
    return tuple(sorted(set(cleaned))) if cleaned else ("unknown_topic",)


# sotf_jaccard(bertopic_lda_nmf).py: lower, strip, discard one-character words, sorted set.
def clean_word_list(words):
    if not isinstance(words, list):
        return []
    return [str(word).strip().lower() for word in words if str(word).strip() and len(str(word).strip()) > 1]


def soft_topic_key(topics_list):
    if not topics_list or not isinstance(topics_list, list):
        return ("unknown_topic",)
    cleaned = clean_word_list(topics_list)
    return tuple(sorted(set(cleaned))) if cleaned else ("unknown_topic",)


def build_topic_groups(records, key_builder):
    groups = defaultdict(list)
    for _, row in records.iterrows():
        groups[key_builder(row["model_topics"])].append(preprocess_text(row["metric_text"]))
    return groups


def topic_assignment_key(row):
    topic_id = row.get("topic_id")
    if topic_id is not None and not pd.isna(topic_id):
        try:
            topic_id = int(topic_id)
        except (TypeError, ValueError):
            topic_id = None
        if topic_id is not None:
            return ("topic_id", topic_id)

    topic_words = row.get("model_topics", [])
    return cv_topic_key(topic_words) if topic_words else None


def build_assignment_groups(records):
    groups = defaultdict(list)
    for _, row in records.iterrows():
        topic_key = topic_assignment_key(row)
        if topic_key is not None:
            groups[topic_key].append(preprocess_text(row["metric_text"]))
    return groups


def extract_top_words(groups, corpus, top_k=TOP_WORDS):
    if not corpus:
        return {}
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
    )
    vectorizer.fit(corpus)
    feature_names = vectorizer.get_feature_names_out()
    top_words = {}
    for topic_key, docs in groups.items():
        if not docs:
            top_words[topic_key] = []
            continue
        scores = np.asarray(vectorizer.transform(docs).mean(axis=0)).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        top_words[topic_key] = [feature_names[i] for i in top_indices if scores[i] > 0][:top_k]
    return top_words


def calculate_cv_td(groups, corpus, top_k=TOP_WORDS):
    """Original CV/TD calculation retained for backward-compatible results."""
    topic_words = extract_top_words(groups, corpus, top_k)
    topics_words_list = [words for words in topic_words.values() if words]
    tokenized_corpus = [doc.split() for doc in corpus]
    cv_score = 0.0
    cv_npmi_score = 0.0
    if topics_words_list and tokenized_corpus:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel

        dictionary = Dictionary(tokenized_corpus)
        cv_score = CoherenceModel(
            topics=topics_words_list,
            texts=tokenized_corpus,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        ).get_coherence()
        cv_npmi_score = CoherenceModel(
            topics=topics_words_list,
            texts=tokenized_corpus,
            dictionary=dictionary,
            coherence="c_npmi",
            topn=10,
            processes=1,
        ).get_coherence()

    filtered_top_words = [
        word
        for words in topics_words_list
        for word in words[:top_k]
        if word.lower() not in DOMAIN_COMMON_WORDS
    ]
    td_score = len(set(filtered_top_words)) / len(filtered_top_words) if filtered_top_words else 0.0
    return float(cv_score), float(cv_npmi_score), float(td_score), topic_words


def normalize_metric_words(words, top_k):
    normalized = []
    for word in words:
        token = str(word).strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", token):
            continue
        if len(token) <= 2 or token in normalized:
            continue
        normalized.append(token)
        if len(normalized) == top_k:
            break
    return normalized


def extract_document_tfidf_words(groups, corpus, top_k=TOP_WORDS):
    if not corpus:
        return {}
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 1),
        token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9-]*\b",
    )
    vectorizer.fit(corpus)
    feature_names = vectorizer.get_feature_names_out()
    topic_words = {}
    for topic_key, docs in groups.items():
        if not docs:
            topic_words[topic_key] = []
            continue
        scores = np.asarray(vectorizer.transform(docs).mean(axis=0)).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        topic_words[topic_key] = [
            feature_names[index]
            for index in top_indices
            if scores[index] > 0
        ][:top_k]
    return topic_words


def native_topic_words(records, top_k=TOP_WORDS):
    topic_words = {}
    for _, row in records.iterrows():
        topic_key = topic_assignment_key(row)
        if topic_key is None or topic_key in topic_words:
            continue
        words = row.get("native_topic_words", row["model_topics"])
        topic_words[topic_key] = normalize_metric_words(words, top_k)
    return topic_words


def calculate_cv_td_from_topic_words(topic_words, corpus, top_k=TOP_WORDS):
    normalized_topic_words = {
        topic_key: normalize_metric_words(words, top_k)
        for topic_key, words in topic_words.items()
    }
    topics_words_list = [words for words in normalized_topic_words.values() if words]
    tokenized_corpus = [doc.split() for doc in corpus]
    cv_score = 0.0
    cv_npmi_score = 0.0
    if topics_words_list and tokenized_corpus:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel

        dictionary = Dictionary(tokenized_corpus)
        cv_score = CoherenceModel(
            topics=topics_words_list,
            texts=tokenized_corpus,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        ).get_coherence()
        cv_npmi_score = CoherenceModel(
            topics=topics_words_list,
            texts=tokenized_corpus,
            dictionary=dictionary,
            coherence="c_npmi",
            topn=top_k,
            processes=1,
        ).get_coherence()

    all_top_words = [
        word
        for words in topics_words_list
        for word in words[:top_k]
    ]
    td_score = len(set(all_top_words)) / len(all_top_words) if all_top_words else 0.0
    return float(cv_score), float(cv_npmi_score), float(td_score), normalized_topic_words


def calculate_kl_background(groups, corpus, top_k=TOP_WORDS):
    all_tokens = [word for doc in corpus for word in doc.split()]
    if not all_tokens:
        return pd.DataFrame(columns=["Topic_Key", "Coverage", "KL_background", "Top_Words"])
    background_prob = {
        word: count / len(all_tokens)
        for word, count in Counter(all_tokens).items()
    }
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
    ).fit(corpus)
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for topic_key, docs in groups.items():
        coverage = len(docs)
        if coverage == 0:
            rows.append(
                {
                    "Topic_Key": str(topic_key),
                    "Coverage": 0,
                    "KL_background": 0.0,
                    "Top_Words": "[]",
                }
            )
            continue
        scores = np.asarray(vectorizer.transform(docs).sum(axis=0)).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        top_words = [feature_names[i] for i in top_indices if scores[i] > 0][:top_k]
        top_scores = [scores[i] for i in top_indices if scores[i] > 0][:top_k]
        if not top_words or sum(top_scores) == 0:
            kl_score = 0.0
        else:
            total_score = sum(top_scores)
            kl_score = 0.0
            for word, score in zip(top_words, top_scores):
                p_w_given_t = score / total_score
                p_w = background_prob.get(word, 1e-10)
                kl_score += p_w_given_t * math.log(p_w_given_t / (p_w + 1e-10) + 1e-10)
            kl_score = max(kl_score, 0.0)
        rows.append(
            {
                "Topic_Key": str(topic_key),
                "Coverage": coverage,
                "KL_background": round(float(kl_score), 6),
                "Top_Words": str(top_words),
            }
        )
    return pd.DataFrame(rows).sort_values("KL_background", ascending=False).reset_index(drop=True)


def calculate_p_npmi(records, embedding_model_path):
    """P_NPMI(yzg).py: NPMI_Avg/Max plus two semantic-similarity values."""
    co_occur = defaultdict(int)
    word_freq = defaultdict(int)
    total_docs = len(records)
    for _, row in records.iterrows():
        words1 = set(row["model_topics"])
        words2 = set(row["true_keywords"])
        for word in words1:
            word_freq[word] += 1
        for word in words2:
            word_freq[word] += 1
        for word1 in words1:
            for word2 in words2:
                co_occur[tuple(sorted((word1, word2)))] += 1

    epsilon = 1e-8
    npmi_dict = {}
    for (word1, word2), frequency in co_occur.items():
        p_xy = (frequency + epsilon) / (total_docs + 2 * epsilon)
        p_x = (word_freq[word1] + epsilon) / (total_docs + 2 * epsilon)
        p_y = (word_freq[word2] + epsilon) / (total_docs + 2 * epsilon)
        pmi = np.log(p_xy / (p_x * p_y))
        npmi_dict[(word1, word2)] = pmi / (-np.log(p_xy) + epsilon)

    npmi_rows = []
    for _, row in records.iterrows():
        scores = []
        for word1 in set(row["model_topics"]):
            for word2 in set(row["true_keywords"]):
                key = tuple(sorted((word1, word2)))
                if key in npmi_dict:
                    scores.append(npmi_dict[key])
        npmi_rows.append(
            {
                "NPMI_Avg": float(np.mean(scores)) if scores else 0.0,
                "NPMI_Max": float(np.max(scores)) if scores else 0.0,
                "NPMI_Num_Pairs": len(scores),
            }
        )

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(str(embedding_model_path))
    all_targets = [word for sublist in records["true_keywords"] for word in sublist]
    all_topics = [topic for sublist in records["model_topics"] for topic in sublist]
    if all_targets:
        target_embs = model.encode(all_targets, batch_size=128, convert_to_numpy=True)
    else:
        target_embs = np.empty((0, 0))
    if all_topics:
        topic_embs = model.encode(all_topics, batch_size=128, convert_to_numpy=True)
    else:
        topic_embs = np.empty((0, 0))

    target_idx_map = []
    curr = 0
    for targets in records["true_keywords"]:
        target_idx_map.append((curr, curr + len(targets)))
        curr += len(targets)
    topic_idx_map = []
    curr = 0
    for topics in records["model_topics"]:
        topic_idx_map.append((curr, curr + len(topics)))
        curr += len(topics)

    mean_sims = []
    max_sims = []
    for i in range(len(records)):
        t_start, t_end = topic_idx_map[i]
        g_start, g_end = target_idx_map[i]
        if t_end <= t_start or g_end <= g_start:
            mean_sims.append(0.0)
            max_sims.append(0.0)
            continue
        sim_matrix = cosine_similarity(topic_embs[t_start:t_end], target_embs[g_start:g_end])
        mean_sims.append(float(np.mean(np.max(sim_matrix, axis=1))))
        max_sims.append(float(np.max(sim_matrix)))

    result = records.copy()
    for column in ("NPMI_Avg", "NPMI_Max", "NPMI_Num_Pairs"):
        result[column] = [row[column] for row in npmi_rows]
    result["Semantic_Similarity_Mean"] = mean_sims
    result["Semantic_Similarity_Max"] = max_sims
    return result


def calculate_soft_jaccard(records, groups, corpus, embedding_model_path, top_k=TOP_WORDS):
    """sotf_jaccard(bertopic_lda_nmf).py."""
    topic_key_to_top_words = extract_top_words(groups, corpus, top_k)
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    semantic_model = SentenceTransformer(str(embedding_model_path))
    scores = []
    model_words_column = []
    for _, row in records.iterrows():
        topic_key = soft_topic_key(row["model_topics"])
        author_kws = clean_word_list(row["true_keywords"])
        model_words = topic_key_to_top_words.get(topic_key, [])
        model_words_column.append(model_words)
        if not author_kws or not model_words:
            scores.append(0.0)
            continue
        try:
            embs_a = semantic_model.encode(author_kws, convert_to_numpy=True)
            embs_b = semantic_model.encode(model_words, convert_to_numpy=True)
            sim_matrix = cosine_similarity(embs_a, embs_b)
            recall = np.mean(np.max(sim_matrix, axis=1))
            precision = np.mean(np.max(sim_matrix, axis=0))
            scores.append(float((recall + precision) / 2.0))
        except Exception as exc:
            print(f"Soft Jaccard encoding error: {exc}")
            scores.append(0.0)

    result = records.copy()
    result["Model_Top_Words"] = model_words_column
    result["Soft_Jaccard_Similarity"] = scores
    return result, topic_key_to_top_words


def calculate_topic_document_relevance(records, embedding_model_path, config):
    """Unified relevance metrics; traditional models provide one concatenated topic per document."""
    sections = config.get("relevance_sections", ["title", "abstract"])
    if isinstance(sections, str):
        sections = [item.strip() for item in sections.split(",") if item.strip()]
    min_chars = int(config.get("relevance_min_sentence_chars", 20))
    max_sentences = int(config.get("relevance_max_sentences_per_doc", 12))

    sentence_groups = []
    topic_groups = []
    topic_texts = []
    for _, row in records.iterrows():
        words = row.get("native_topic_words", row.get("model_topics", []))
        if not words:
            words = row.get("model_topics", [])
        topic_words = normalize_metric_words(words, int(config.get("top_words_for_metrics", TOP_WORDS)))
        topic_text = "; ".join(topic_words)
        topic_texts.append(topic_text)
        topic_groups.append([topic_text] if topic_text else [])
        sentence_groups.append(
            document_relevance_sentences(
                row.get("metric_text", ""),
                sections=sections,
                min_chars=min_chars,
                max_sentences=max_sentences,
            )
        )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(embedding_model_path))
    all_topics = [topic for topics in topic_groups for topic in topics]
    topic_embeddings = np.empty((0, 0))
    if all_topics:
        topic_embeddings = model.encode(
            all_topics,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    all_sentences = [sentence for sentences in sentence_groups for sentence in sentences]
    sentence_embeddings = np.empty((0, 0))
    if all_sentences:
        sentence_embeddings = model.encode(
            all_sentences,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    mean_best_topic_scores = []
    max_scores = []
    mean_all_pair_scores = []
    mean_best_sentence_scores = []
    topic_counts = []
    sentence_counts = []
    topic_cursor = 0
    cursor = 0
    for topics, sentences in zip(topic_groups, sentence_groups):
        topic_count = len(topics)
        count = len(sentences)
        topic_counts.append(topic_count)
        sentence_counts.append(count)
        if topic_count == 0 or count == 0:
            mean_best_topic_scores.append(0.0)
            max_scores.append(0.0)
            mean_all_pair_scores.append(0.0)
            mean_best_sentence_scores.append(0.0)
            topic_cursor += topic_count
            cursor += count
            continue
        doc_topic_embeddings = topic_embeddings[topic_cursor : topic_cursor + topic_count]
        doc_sentence_embeddings = sentence_embeddings[cursor : cursor + count]
        similarities = doc_sentence_embeddings @ doc_topic_embeddings.T
        mean_best_topic_scores.append(float(np.mean(np.max(similarities, axis=1))))
        max_scores.append(float(np.max(similarities)))
        mean_all_pair_scores.append(float(np.mean(similarities)))
        mean_best_sentence_scores.append(float(np.mean(np.max(similarities, axis=0))))
        topic_cursor += topic_count
        cursor += count

    result = records.copy()
    result["relevance_mean_best_topic"] = mean_best_topic_scores
    result["relevance_max"] = max_scores
    result["relevance_mean_all_pairs"] = mean_all_pair_scores
    result["relevance_mean_best_sentence"] = mean_best_sentence_scores
    result["relevance_topic_count"] = topic_counts
    result["relevance_sentence_count"] = sentence_counts
    result["relevance_topic_text"] = topic_texts
    return result


def reference_document_text(row):
    """Use abstract text for the migrated document-level metrics."""
    text = str(row.get("metric_text", "") or "")
    abstract = extract_tagged_section(text, "abstract")
    if abstract:
        return abstract
    cleaned = re.sub(r"<[^>]+>", " ", text)
    return " ".join(cleaned.split())


def reference_document_sentences(row, min_chars=20, max_sentences=12):
    text = reference_document_text(row)
    sentences = re.sub(r"\[[^\]]*\]", " ", text)
    sentences = re.sub(r"\s+", " ", sentences).strip()
    if not sentences:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", sentences)
    return [part.strip() for part in parts if len(part.strip()) >= min_chars][:max_sentences]


def build_reference_topic_assignments(records, include_null=True):
    """Use topic_id as the document cluster key for traditional model outputs."""
    topic_to_doc_indices = defaultdict(list)
    doc_topic_keys = []
    for index, (_, row) in enumerate(records.iterrows()):
        topic_id = row.get("topic_id")
        if topic_id is None or (isinstance(topic_id, float) and pd.isna(topic_id)):
            topic_key = topic_assignment_key(row)
            keys = [str(topic_key)] if topic_key is not None else ([NULL_TOPIC_KEY] if include_null else [])
        else:
            try:
                topic_id = int(topic_id)
            except (TypeError, ValueError):
                topic_id = str(topic_id).strip()
            keys = [str(topic_id)]
        for key in keys:
            topic_to_doc_indices[key].append(index)
        doc_topic_keys.append(keys)
    return topic_to_doc_indices, doc_topic_keys


def build_reference_representative_words(records, topic_to_doc_indices, corpus, top_n):
    """Re-extract representative words from documents assigned to each topic."""
    if not corpus or not any(corpus) or not topic_to_doc_indices:
        return {key: [] for key in topic_to_doc_indices}
    try:
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w\-]+\b",
        )
        vectorizer.fit(corpus)
    except ValueError:
        return {key: [] for key in topic_to_doc_indices}

    feature_names = vectorizer.get_feature_names_out()
    representative = {}
    for topic_key, indices in topic_to_doc_indices.items():
        docs = [corpus[index] for index in indices if corpus[index]]
        if not docs:
            representative[topic_key] = []
            continue
        scores = np.asarray(vectorizer.transform(docs).mean(axis=0)).flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        representative[topic_key] = [
            {
                "term": str(feature_names[index]),
                "score": float(scores[index]),
            }
            for index in top_indices
            if scores[index] > 0
        ][:top_n]
    return representative


def load_reference_embedder(embedding_model_path):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(embedding_model_path))


class ReferenceEmbedder:
    def __init__(self, model):
        self.model = model
        self.cache = {}

    def encode(self, values):
        normalized = [str(value).strip() for value in values if str(value).strip()]
        missing = [value for value in normalized if value not in self.cache]
        if missing:
            embeddings = self.model.encode(
                missing,
                batch_size=128,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings = np.atleast_2d(np.asarray(embeddings, dtype="float32"))
            for value, embedding in zip(missing, embeddings):
                self.cache[value] = embedding
        if not normalized:
            return np.empty((0, 0), dtype="float32")
        return np.asarray([self.cache[value] for value in normalized], dtype="float32")


def calculate_reference_tka(topic_words, keywords, embedder):
    if not topic_words or not keywords:
        return {
            "valid": False,
            "topic_count": len(topic_words),
            "keyword_count": len(keywords),
            "semantic_similarity_max": 0.0,
            "TKA": 0.0,
        }
    similarity = embedder.encode(topic_words) @ embedder.encode(keywords).T
    score = float(np.max(similarity))
    return {
        "valid": True,
        "topic_count": len(topic_words),
        "keyword_count": len(keywords),
        "semantic_similarity_max": score,
        "TKA": score,
    }


def calculate_reference_tcr(topic_words, sentences, embedder):
    if not topic_words or not sentences:
        return {
            "valid": False,
            "topic_count": len(topic_words),
            "sentence_count": len(sentences),
            "relevance_mean_best_sentence": 0.0,
            "TCR": 0.0,
        }
    similarity = embedder.encode(sentences) @ embedder.encode(topic_words).T
    score = float(np.mean(np.max(similarity, axis=0)))
    return {
        "valid": True,
        "topic_count": len(topic_words),
        "sentence_count": len(sentences),
        "relevance_mean_best_sentence": score,
        "TCR": score,
    }


def calculate_reference_soft_jaccard(author_keywords, model_words, embedder):
    if not author_keywords or not model_words:
        return {
            "valid": False,
            "Soft_Jaccard": 0.0,
            "recall": 0.0,
            "precision": 0.0,
        }
    similarity = embedder.encode(author_keywords) @ embedder.encode(model_words).T
    recall = float(np.mean(np.max(similarity, axis=1)))
    precision = float(np.mean(np.max(similarity, axis=0)))
    return {
        "valid": True,
        "Soft_Jaccard": float((recall + precision) / 2.0),
        "recall": recall,
        "precision": precision,
    }


def calculate_reference_cv(topic_words, corpus, window_size=REFERENCE_CV_WINDOW_SIZE):
    valid_topics = [words for words in topic_words if words]
    tokenized_corpus = [document.split() for document in corpus if document]
    if not valid_topics or not tokenized_corpus:
        return {"C_V": 0.0, "available": False, "error": ""}
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel

        dictionary = Dictionary(tokenized_corpus)
        model = CoherenceModel(
            topics=valid_topics,
            texts=tokenized_corpus,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
            window_size=window_size,
        )
        return {"C_V": float(model.get_coherence()), "available": True, "error": ""}
    except Exception as exc:
        return {"C_V": 0.0, "available": False, "error": str(exc)}


def calculate_reference_td(topic_words, top_n):
    filtered = [
        word
        for words in topic_words
        for word in words[:top_n]
        if word.lower() not in DOMAIN_COMMON_WORDS
    ]
    return float(len(set(filtered)) / len(filtered)) if filtered else 0.0


def build_reference_term_doc_sets(topic_words, corpus):
    terms = sorted({word for words in topic_words for word in words if word})
    token_sets = [set(document.split()) for document in corpus]
    return {
        term: {
            index
            for index, document in enumerate(corpus)
            if document and (
                term in token_sets[index]
                if " " not in term
                else f" {term} " in f" {document} "
            )
        }
        for term in terms
    }


def calculate_reference_mimno_coherence(words, term_doc_sets):
    if len(words) < 2:
        return 0.0
    value = 0.0
    for k in range(1, len(words)):
        docs_k = term_doc_sets.get(words[k], set())
        for j in range(k):
            docs_j = term_doc_sets.get(words[j], set())
            df_j = len(docs_j)
            if df_j:
                value += math.log((len(docs_k & docs_j) + 1.0) / df_j)
    return float(value)


def calculate_reference_kl(words, corpus):
    token_counts = Counter(token for document in corpus for token in document.split())
    total_tokens = sum(token_counts.values())
    if not words or not total_tokens:
        return 0.0
    total_score = sum(item["score"] for item in words)
    if total_score <= 0:
        return 0.0
    value = 0.0
    for item in words:
        p_w_given_t = item["score"] / total_score
        p_w = token_counts.get(item["term"], 0) / total_tokens
        value += p_w_given_t * math.log(p_w_given_t / (p_w + 1e-10) + 1e-10)
    return float(max(value, 0.0))


def calculate_reference_cluster_metrics(
    topic_to_doc_indices,
    doc_embeddings,
    embedding_valid_mask,
):
    topic_counts = [len(indices) for indices in topic_to_doc_indices.values() if indices]
    if len(topic_counts) <= 1:
        shannon_entropy = 0.0
        pielou_evenness = 0.0
    else:
        total_documents = sum(topic_counts)
        proportions = [count / total_documents for count in topic_counts]
        shannon_entropy = float(
            -sum(p * math.log(p) for p in proportions if p > 0)
        )
        pielou_evenness = float(shannon_entropy / math.log(len(topic_counts)))

    effective_assignments = {
        key: [index for index in indices if embedding_valid_mask[index]]
        for key, indices in topic_to_doc_indices.items()
    }
    effective_assignments = {
        key: indices for key, indices in effective_assignments.items() if indices
    }
    topic_keys = list(effective_assignments)
    empty_result = {
        "topic_count": len(topic_to_doc_indices),
        "non_empty_topic_count": len(topic_keys),
        "Compactness_mean": 0.0,
        "Separation_mean": 0.0,
        "Silhouette": 0.0,
        "Silhouette_available": False,
        "Davies_Bouldin": 0.0,
        "Davies_Bouldin_available": False,
        "Calinski_Harabasz": 0.0,
        "Calinski_Harabasz_available": False,
        "Pielou_evenness": pielou_evenness,
        "Shannon_entropy": shannon_entropy,
        "topics": [],
    }
    if not topic_keys or doc_embeddings.size == 0:
        return empty_result

    centroids = {}
    compactness = {}
    for key, indices in effective_assignments.items():
        vectors = doc_embeddings[indices]
        centroid = np.asarray(vectors.mean(axis=0), dtype="float32")
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[key] = centroid
        compactness[key] = 1.0 if len(indices) == 1 else float(np.mean(vectors @ centroid))

    centroid_keys = list(centroids)
    centroid_matrix = np.asarray([centroids[key] for key in centroid_keys], dtype="float32")
    centroid_similarity = centroid_matrix @ centroid_matrix.T
    separation = {}
    nearest = {}
    for i, key in enumerate(centroid_keys):
        others = [
            (float(centroid_similarity[i, j]), centroid_keys[j])
            for j in range(len(centroid_keys))
            if j != i
        ]
        if others:
            nearest_similarity, nearest_key = max(others, key=lambda item: item[0])
        else:
            nearest_similarity, nearest_key = 0.0, ""
        nearest[key] = (nearest_key, nearest_similarity)
        separation[key] = float(1.0 - nearest_similarity)

    labels = []
    vectors = []
    label_index = {key: index for index, key in enumerate(centroid_keys)}
    for key, indices in effective_assignments.items():
        for index in indices:
            labels.append(label_index[key])
            vectors.append(doc_embeddings[index])

    silhouette = 0.0
    silhouette_available = False
    davies_bouldin = 0.0
    davies_bouldin_available = False
    calinski_harabasz = 0.0
    calinski_harabasz_available = False
    if len(set(labels)) >= 2 and len(vectors) >= 3 and len(labels) > len(set(labels)):
        matrix = np.asarray(vectors, dtype="float32")
        try:
            distance_matrix = 1.0 - np.clip(matrix @ matrix.T, -1.0, 1.0)
            np.fill_diagonal(distance_matrix, 0.0)
            silhouette = float(silhouette_score(distance_matrix, labels, metric="precomputed"))
            silhouette_available = True
        except Exception:
            pass
        try:
            davies_bouldin = float(davies_bouldin_score(matrix, labels))
            davies_bouldin_available = True
        except Exception:
            pass
        try:
            calinski_harabasz = float(calinski_harabasz_score(matrix, labels))
            calinski_harabasz_available = True
        except Exception:
            pass

    topic_details = []
    for key in sorted(effective_assignments, key=str.casefold):
        nearest_key, nearest_similarity = nearest.get(key, ("", 0.0))
        topic_details.append(
            {
                "topic_key": key,
                "document_count": len(effective_assignments[key]),
                "compactness": float(compactness.get(key, 0.0)),
                "nearest_topic_key": nearest_key,
                "nearest_topic_similarity": float(nearest_similarity),
                "separation": float(separation.get(key, 0.0)),
            }
        )

    compactness_values = list(compactness.values())
    separation_values = list(separation.values())
    return {
        "topic_count": len(topic_to_doc_indices),
        "non_empty_topic_count": len(topic_keys),
        "Compactness_mean": float(np.mean(compactness_values)) if compactness_values else 0.0,
        "Separation_mean": float(np.mean(separation_values)) if separation_values else 0.0,
        "Silhouette": silhouette,
        "Silhouette_available": silhouette_available,
        "Davies_Bouldin": davies_bouldin,
        "Davies_Bouldin_available": davies_bouldin_available,
        "Calinski_Harabasz": calinski_harabasz,
        "Calinski_Harabasz_available": calinski_harabasz_available,
        "Pielou_evenness": pielou_evenness,
        "Shannon_entropy": shannon_entropy,
        "topics": topic_details,
    }


def calculate_reference_topic_metrics(records, corpus, config):
    """Migrate the reference metrics to traditional topic-word-set outputs."""
    top_n = int(config.get("reference_metrics_top_n", config.get("top_words_for_metrics", TOP_WORDS)))
    min_chars = int(config.get("reference_metrics_min_sentence_chars", 20))
    max_sentences = int(config.get("reference_metrics_max_sentences", 12))
    include_null = bool(config.get("reference_metrics_include_null_topic", True))

    topic_to_doc_indices, doc_topic_keys = build_reference_topic_assignments(
        records,
        include_null=include_null,
    )
    representative_words = build_reference_representative_words(
        records,
        topic_to_doc_indices,
        corpus,
        top_n,
    )
    topic_words_by_key = {
        key: [item["term"] for item in words[:top_n]]
        for key, words in representative_words.items()
    }
    topic_words = [topic_words_by_key.get(key, []) for key in topic_to_doc_indices]
    term_doc_sets = build_reference_term_doc_sets(topic_words, corpus)
    cv_result = calculate_reference_cv(
        topic_words,
        corpus,
        int(config.get("reference_metrics_cv_window_size", REFERENCE_CV_WINDOW_SIZE)),
    )

    semantic_model = load_reference_embedder(config["embedding_model_path"])
    embedder = ReferenceEmbedder(semantic_model)
    document_texts = [reference_document_text(row) for _, row in records.iterrows()]
    valid_mask = np.asarray([bool(text) for text in document_texts], dtype=bool)
    valid_texts = [text for text in document_texts if text]
    if valid_texts:
        valid_embeddings = embedder.encode(valid_texts)
        document_embeddings = np.zeros(
            (len(records), valid_embeddings.shape[1]),
            dtype="float32",
        )
        document_embeddings[valid_mask] = valid_embeddings
    else:
        document_embeddings = np.empty((len(records), 0), dtype="float32")

    document_rows = []
    for index, (_, record) in enumerate(records.iterrows()):
        keywords = record["true_keywords"]
        topic_keys = doc_topic_keys[index]
        model_words = []
        for key in topic_keys:
            model_words.extend(topic_words_by_key.get(key, []))
        model_words = list(dict.fromkeys(model_words))
        sentences = reference_document_sentences(record, min_chars, max_sentences)
        tka = calculate_reference_tka(model_words, keywords, embedder)
        tcr = calculate_reference_tcr(model_words, sentences, embedder)
        soft_jaccard = calculate_reference_soft_jaccard(keywords, model_words, embedder)
        document_rows.append(
            {
                "row_index": index,
                "title": record["title"],
                "topic_key": "; ".join(topic_keys),
                "model_topic_words": model_words,
                "true_keywords": keywords,
                "reference_sentences": sentences,
                "TKA": tka["TKA"],
                "TKA_valid": tka["valid"],
                "TKA_topic_count": tka["topic_count"],
                "TKA_keyword_count": tka["keyword_count"],
                "TCR": tcr["TCR"],
                "TCR_valid": tcr["valid"],
                "TCR_sentence_count": tcr["sentence_count"],
                "Reference_Soft_Jaccard": soft_jaccard["Soft_Jaccard"],
                "Reference_Soft_Jaccard_valid": soft_jaccard["valid"],
                "Reference_Soft_Jaccard_recall": soft_jaccard["recall"],
                "Reference_Soft_Jaccard_precision": soft_jaccard["precision"],
            }
        )

    cluster_metrics = calculate_reference_cluster_metrics(
        topic_to_doc_indices,
        document_embeddings,
        valid_mask,
    )
    topic_cluster_by_key = {
        item["topic_key"]: item
        for item in cluster_metrics["topics"]
    }
    topic_rows = []
    mimno_values = []
    kl_values = []
    for key in sorted(topic_to_doc_indices, key=str.casefold):
        words = representative_words.get(key, [])[:top_n]
        word_terms = [item["term"] for item in words]
        mimno = calculate_reference_mimno_coherence(word_terms, term_doc_sets)
        kl = calculate_reference_kl(words, corpus)
        mimno_values.append(mimno)
        kl_values.append(kl)
        cluster = topic_cluster_by_key.get(key, {})
        topic_rows.append(
            {
                "topic_key": key,
                "document_count": len(topic_to_doc_indices[key]),
                "document_indices": topic_to_doc_indices[key],
                "representative_words": word_terms,
                "representative_words_with_scores": words,
                "Topic_Coherence_Mimno": mimno,
                "KL_background": kl,
                "compactness": cluster.get("compactness"),
                "nearest_topic_key": cluster.get("nearest_topic_key"),
                "nearest_topic_similarity": cluster.get("nearest_topic_similarity"),
                "separation": cluster.get("separation"),
            }
        )

    valid_tka = [row["TKA"] for row in document_rows if row["TKA_valid"]]
    valid_tcr = [row["TCR"] for row in document_rows if row["TCR_valid"]]
    valid_soft = [
        row["Reference_Soft_Jaccard"]
        for row in document_rows
        if row["Reference_Soft_Jaccard_valid"]
    ]
    summary = {
        "TKA": float(np.mean(valid_tka)) if valid_tka else 0.0,
        "TKA_valid_documents": len(valid_tka),
        "TCR": float(np.mean(valid_tcr)) if valid_tcr else 0.0,
        "TCR_valid_documents": len(valid_tcr),
        "Reference_Soft_Jaccard": float(np.mean(valid_soft)) if valid_soft else 0.0,
        "Reference_Soft_Jaccard_valid_documents": len(valid_soft),
        "C_V_reference": cv_result["C_V"],
        "C_V_reference_available": cv_result["available"],
        "C_V_reference_error": cv_result["error"],
        "TD_reference": calculate_reference_td(topic_words, top_n),
        "Topic_Coherence_Mimno_mean": float(np.mean(mimno_values)) if mimno_values else 0.0,
        "KL_background_reference_mean": float(np.mean(kl_values)) if kl_values else 0.0,
        "Compactness_mean": cluster_metrics["Compactness_mean"],
        "Separation_mean": cluster_metrics["Separation_mean"],
        "Silhouette": cluster_metrics["Silhouette"],
        "Silhouette_available": cluster_metrics["Silhouette_available"],
        "Davies_Bouldin": cluster_metrics["Davies_Bouldin"],
        "Davies_Bouldin_available": cluster_metrics["Davies_Bouldin_available"],
        "Calinski_Harabasz": cluster_metrics["Calinski_Harabasz"],
        "Calinski_Harabasz_available": cluster_metrics["Calinski_Harabasz_available"],
        "Pielou_evenness": cluster_metrics["Pielou_evenness"],
        "Shannon_entropy": cluster_metrics["Shannon_entropy"],
        "unique_topic_count": len(topic_to_doc_indices),
        "documents_with_topics": int(sum(1 for keys in doc_topic_keys if keys and keys != [NULL_TOPIC_KEY])),
        "mean_topics_per_document": float(
            np.mean([0 if keys == [NULL_TOPIC_KEY] else len(keys) for keys in doc_topic_keys])
        )
        if doc_topic_keys
        else 0.0,
        "top_n": top_n,
        "document_embedding_source": "abstract",
        "document_embedding_valid_documents": int(np.sum(valid_mask)),
        "document_embedding_missing_documents": int(len(valid_mask) - np.sum(valid_mask)),
        "topic_word_source": "re_extracted_tfidf_words_from_topic_documents",
    }
    return {
        "summary": summary,
        "document_details": pd.DataFrame(document_rows),
        "topic_details": pd.DataFrame(topic_rows),
        "topic_details_json": {
            "topic_count": len(topic_to_doc_indices),
            "top_n": top_n,
            "topics": topic_rows,
            "cluster_metrics": cluster_metrics,
        },
    }


def evaluate_model(model_name, model_jsonl, config):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    records, unmatched = create_evaluation_table(
        model_name,
        model_jsonl,
        config["metadata_file"],
        config.get("title_column"),
        config.get("keywords_column"),
    )
    if records.empty:
        raise ValueError("No records matched metadata titles; cannot calculate metrics.")
    records = attach_full_text(records, config["input_file"])
    records.to_excel(output_dir / "evaluation_input.xlsx", index=False)

    top_k = int(config.get("top_words_for_metrics", TOP_WORDS))
    corpus = build_corpus(records)

    original_cv_groups = build_topic_groups(records, cv_topic_key)
    original_cv_score, original_cv_npmi_score, original_td_score, original_topic_words = calculate_cv_td(
        original_cv_groups,
        corpus,
        top_k,
    )
    cv_groups = build_assignment_groups(records)
    native_words = native_topic_words(records, top_k)
    native_cv_score, native_cv_npmi_score, native_td_score, native_words = calculate_cv_td_from_topic_words(
        native_words,
        corpus,
        top_k,
    )
    document_tfidf_words = extract_document_tfidf_words(cv_groups, corpus, top_k)
    cv_score, cv_npmi_score, td_score, document_tfidf_words = calculate_cv_td_from_topic_words(
        document_tfidf_words,
        corpus,
        top_k,
    )
    with open(metrics_dir / "cv_td.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "native": {
                    "CV": native_cv_score,
                    "CV_NPMI": native_cv_npmi_score,
                    "TD": native_td_score,
                    "topic_count": len(native_words),
                    "topic_words": {str(key): value for key, value in native_words.items()},
                },
                "document_tfidf": {
                    "CV": cv_score,
                    "CV_NPMI": cv_npmi_score,
                    "TD": td_score,
                    "topic_count": len(document_tfidf_words),
                    "topic_words": {str(key): value for key, value in document_tfidf_words.items()},
                },
                # Original calculation retained unchanged for backward compatibility.
                "CV": original_cv_score,
                "CV_NPMI": original_cv_npmi_score,
                "TD": original_td_score,
                "topic_count": len(original_topic_words),
                "topic_words": {str(key): value for key, value in original_topic_words.items()},
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    kl_groups = build_topic_groups(records, kl_topic_key)
    kl_df = calculate_kl_background(kl_groups, corpus, top_k)
    kl_df.to_excel(metrics_dir / "kl_background.xlsx", index=False)

    p_npmi_df = calculate_p_npmi(records, config["embedding_model_path"])
    p_npmi_df.to_excel(metrics_dir / "p_npmi.xlsx", index=False)

    soft_groups = build_topic_groups(records, soft_topic_key)
    soft_df, _ = calculate_soft_jaccard(
        records,
        soft_groups,
        corpus,
        config["embedding_model_path"],
        top_k,
    )
    soft_df.to_excel(metrics_dir / "soft_jaccard.xlsx", index=False)

    relevance_df = calculate_topic_document_relevance(
        records,
        config["embedding_model_path"],
        config,
    )
    relevance_df.to_excel(metrics_dir / "topic_document_relevance.xlsx", index=False)

    reference_metrics = calculate_reference_topic_metrics(records, corpus, config)
    reference_metrics["document_details"].to_excel(
        metrics_dir / "reference_metric_details.xlsx",
        index=False,
    )
    reference_metrics["topic_details"].to_excel(
        metrics_dir / "reference_topic_details.xlsx",
        index=False,
    )
    pd.DataFrame([reference_metrics["summary"]]).to_csv(
        metrics_dir / "reference_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(metrics_dir / "reference_metrics_summary.json", "w", encoding="utf-8") as handle:
        json.dump(reference_metrics["summary"], handle, ensure_ascii=False, indent=2)
    with open(metrics_dir / "reference_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(reference_metrics["topic_details_json"], handle, ensure_ascii=False, indent=2)

    detailed = p_npmi_df.copy()
    detailed["Model_Top_Words"] = soft_df["Model_Top_Words"]
    detailed["Soft_Jaccard_Similarity"] = soft_df["Soft_Jaccard_Similarity"]
    detailed["relevance_mean_best_topic"] = relevance_df["relevance_mean_best_topic"]
    detailed["relevance_max"] = relevance_df["relevance_max"]
    detailed["relevance_mean_all_pairs"] = relevance_df["relevance_mean_all_pairs"]
    detailed["relevance_mean_best_sentence"] = relevance_df["relevance_mean_best_sentence"]
    detailed["relevance_topic_count"] = relevance_df["relevance_topic_count"]
    detailed["relevance_sentence_count"] = relevance_df["relevance_sentence_count"]
    detailed["relevance_topic_text"] = relevance_df["relevance_topic_text"]
    detailed.to_excel(metrics_dir / "all_metric_details.xlsx", index=False)

    has_keywords = detailed["true_keywords"].apply(lambda value: len(value) > 0)
    has_topics = detailed["model_topics"].apply(lambda value: len(value) > 0)
    strict_valid = has_keywords & has_topics
    has_model_words = detailed["Model_Top_Words"].apply(lambda value: len(value) > 0)
    soft_strict_valid = has_keywords & has_model_words
    total_records = len(records) + unmatched

    summary = {
        "模型": model_name,
        "匹配成功数": int(len(records)),
        "未匹配数": int(unmatched),
        "匹配成功率": float(len(records) / total_records) if total_records else 0.0,
        "NPMI_Avg": float(detailed.loc[strict_valid, "NPMI_Avg"].mean()) if strict_valid.any() else 0.0,
        "NPMI_Max": float(detailed.loc[strict_valid, "NPMI_Max"].mean()) if strict_valid.any() else 0.0,
        "Semantic_Similarity_Mean": float(detailed.loc[strict_valid, "Semantic_Similarity_Mean"].mean()) if strict_valid.any() else 0.0,
        "Semantic_Similarity_Max": float(detailed.loc[strict_valid, "Semantic_Similarity_Max"].mean()) if strict_valid.any() else 0.0,
        "CV": original_cv_score,
        "CV_NPMI": original_cv_npmi_score,
        "TD": original_td_score,
        "CV_native": native_cv_score,
        "CV_NPMI_native": native_cv_npmi_score,
        "TD_native": native_td_score,
        "CV_document_tfidf": cv_score,
        "CV_NPMI_document_tfidf": cv_npmi_score,
        "TD_document_tfidf": td_score,
        "Soft Jaccard": float(detailed.loc[soft_strict_valid, "Soft_Jaccard_Similarity"].mean()) if soft_strict_valid.any() else 0.0,
        "relevance_mean_best_topic": float(detailed.loc[has_topics, "relevance_mean_best_topic"].mean()) if has_topics.any() else 0.0,
        "relevance_max": float(detailed.loc[has_topics, "relevance_max"].mean()) if has_topics.any() else 0.0,
        "relevance_mean_all_pairs": float(detailed.loc[has_topics, "relevance_mean_all_pairs"].mean()) if has_topics.any() else 0.0,
        "relevance_mean_best_sentence": float(detailed.loc[has_topics, "relevance_mean_best_sentence"].mean()) if has_topics.any() else 0.0,
        "KL_mean": float(kl_df["KL_background"].mean()) if not kl_df.empty else 0.0,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    return summary
