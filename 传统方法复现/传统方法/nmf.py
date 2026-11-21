import json
import re
from html import unescape
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path

# 首次运行自动下载
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)

# === 配置 ===
INPUT_FILE = r"E:\主题建模\LSTFM主题建模\数据集\测试数据\ieee_robot_human_1000.jsonl"
OUTPUT_FILE = r"E:\主题建模\LSTFM主题建模\传统方法\传统方法结果汇总_1109之后\ieee_human_robot_1109\nmf\jsonl\nmf_80_topics_1_1109.jsonl"
N_TOPICS = 80
TOP_WORDS_PER_TOPIC = 5
RANDOM_STATE = 42

# 基础英文停用词
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

# 🔑 仅添加真正无意义的噪声词（绝不包含 quantum, algorithm 等！）
domain_noise = {
    # 'et', 'al', 'fig', 'figure', 'table', 'section', 'equation', 'eq',
    # 'author', 'authors', 'paper', 'study', 'results', 'method', 'approach',
    # 'based', 'using', 'proposed', 'presented', 'show', 'discuss', 'analysis',
    # 'one', 'two', 'first', 'new', 'novel', 'recent', 'general', 'different',
    # 'wileyonlinelibrary', 'doi', 'com', 'www', 'copyright', 'rights', 'reserved',
    # 'abstract', 'introduction', 'conclusion', 'theory', 'results', 'discussion'
}
stop_words = stop_words.union(domain_noise)


def extract_title_abstract(text: str) -> str:
    """从 <title>...<abstract>... 中提取干净文本"""
    title = ""
    abstract = ""
    # 提取 <title> 内容（不区分大小写）
    title_match = re.search(r'<title[^>]*>(.*?)(?:<abstract|<Introduction|</?text|<|$)', text, re.DOTALL | re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
    # 提取 <abstract> 内容
    abs_match = re.search(r'<abstract[^>]*>(.*?)(?:<Introduction|<Theory|<Conclusion|</?text|<|$)', text, re.DOTALL | re.IGNORECASE)
    if abs_match:
        abstract = abs_match.group(1).strip()
    return (title + " " + abstract).strip()


def preprocess_text_for_nmf(text: str) -> str:
    if not text:
        return ""
    # 移除所有标签（包括自定义标签）
    text = re.sub(r'<[^>]+>', ' ', text)
    text = unescape(text)
    text = text.lower()

    # 移除非 ASCII 字符（保留英文）
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    # 移除数字、科学计数、引用如 [1], [12,13]
    text = re.sub(r'\b\d+(\.\d+)?([eE][+-]?\d+)?\b', ' ', text)
    text = re.sub(r'\[[\d,\s]+\]', ' ', text)
    # 只保留字母和空格
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 分词
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()

    # 过滤：停用词 + 长度>2 + 纯字母
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2 and t.isalpha()]
    return ' '.join(tokens)


def run_nmf_topic_modeling(
    input_file=INPUT_FILE,
    output_file=OUTPUT_FILE,
    n_topics=N_TOPICS,
    top_words_per_topic=TOP_WORDS_PER_TOPIC,
    metric_top_words=10,
    random_state=RANDOM_STATE,
    max_features=5000,
    min_df=2,
    max_df=0.90,
    alpha_w=0.0,
    l1_ratio=0.01,
    init='nndsvd',
    max_iter=1000,
    tol=1e-4,
):
    print("🚀 开始 NMF 主题建模...")

    raw_texts = []
    processed_texts = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取文档"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                full_text = data.get("text", "")
                # ✅ 关键：只用标题+摘要
                clean_text = extract_title_abstract(full_text)
                raw_texts.append(full_text)
                proc_text = preprocess_text_for_nmf(clean_text)
                processed_texts.append(proc_text)
            except Exception as e:
                print(f"⚠️ 跳过无效行: {e}")
                raw_texts.append("")
                processed_texts.append("")

    print(f"✅ 共加载 {len(raw_texts)} 个文档")

    # 过滤空处理文本
    non_empty = [(r, p) for r, p in zip(raw_texts, processed_texts) if p.strip()]
    if not non_empty:
        raise ValueError("预处理后无有效文本！请检查 extract_title_abstract 和 preprocess_text_for_nmf。")
    raw_texts, processed_texts = zip(*non_empty)
    raw_texts, processed_texts = list(raw_texts), list(processed_texts)
    print(f"✅ 过滤后剩余 {len(processed_texts)} 个有效文档")

    # === 调试：打印样例 ===
    print("\n🔍 预处理后样例（前3篇）:")
    for i in range(min(3, len(processed_texts))):
        print(f"  Doc {i}: {processed_texts[i][:120]}...")

    # Step 2: TF-IDF
    print("\nVectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        # stop_words='english'
    )
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    print(f"✅ TF-IDF 矩阵 shape: {tfidf_matrix.shape}")
    print(f"✅ TF-IDF 非零元素比例: {tfidf_matrix.nnz / tfidf_matrix.size:.4%}")

    if tfidf_matrix.nnz == 0:
        raise ValueError("TF-IDF 矩阵全零！预处理可能过度。")

    # Step 3: 训练 NMF（增强版）
    if n_topics > min(tfidf_matrix.shape):
        raise ValueError(
            f"主题数 {n_topics} 不能超过有效文档数/特征数的较小值 {min(tfidf_matrix.shape)}。"
        )

    print(f"训练 NMF (n_components={n_topics})...")
    nmf_model = NMF(
        n_components=n_topics,
        random_state=random_state,
        alpha_W=alpha_w,
        l1_ratio=l1_ratio,
        init=init,
        max_iter=max_iter,
        tol=tol
    )
    W = nmf_model.fit_transform(tfidf_matrix)
    H = nmf_model.components_

    # 🔥 关键：检查 H 是否全零
    print(f"✅ H 矩阵最小值: {H.min():.2e}, 最大值: {H.max():.2e}")
    if H.max() < 1e-10:
        raise RuntimeError("NMF 失败：H 矩阵几乎全零！请检查输入数据或尝试其他初始化。")

    # Step 4: 提取主题关键词
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = []
    topic_metric_words = []
    for topic_idx, topic in enumerate(H):
        word_limit = max(top_words_per_topic, metric_top_words)
        top_indices = topic.argsort()[-word_limit:][::-1]
        top_words = [feature_names[i] for i in top_indices if topic[i] > 0]
        if not top_words:
            top_words = ["(empty)"]
        topic_keywords.append(top_words[:top_words_per_topic])
        topic_metric_words.append(top_words[:metric_top_words])
        print(f"Topic {topic_idx:2d}: {', '.join(topic_keywords[-1])}")
    feature_names = vectorizer.get_feature_names_out()
    print("词表示例（前10个）:", feature_names[:10])
    print("是否包含 'quantum'?", 'quantum' in feature_names)

    # Step 5: 分配主题
    print("\n为文档分配主题...")
    results = []
    for i, raw_text in enumerate(raw_texts):
        topic_idx = np.argmax(W[i])
        results.append({
            "text": raw_text,
            "topic_id": int(topic_idx),
            "topics": ", ".join(topic_keywords[topic_idx]),
            "topic_words": topic_metric_words[topic_idx],
        })

    # Step 6: 保存
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for res in results:
            out_f.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"\n✅ 完成！结果保存至: {output_file}")


if __name__ == "__main__":
    run_nmf_topic_modeling()
