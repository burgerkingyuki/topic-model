import json
import re
from pathlib import Path
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric, \
    strip_multiple_whitespaces, remove_stopwords


def preprocess_text(text):
    """预处理文本"""
    custom_filters = [
        strip_punctuation,
        strip_numeric,
        strip_multiple_whitespaces,
        remove_stopwords
    ]
    processed = preprocess_string(text, custom_filters)
    return [word for word in processed if len(word) > 2]


def lda_topic_mining(
    input_file,
    output_file,
    num_topics=50,
    top_words_per_topic=10,
    top_words_per_doc=5,
    random_state=42,
    passes=10,
    alpha='auto',
    no_below=2,
    no_above=0.8,
):
    """使用LDA进行主题挖掘"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # 读取数据
    texts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                text_content = data.get('text', '')
                texts.append(text_content)

    print(f"成功读取 {len(texts)} 个文档")

    # 预处理所有文本
    processed_texts = [preprocess_text(text) for text in texts]

    # 创建字典和语料库
    dictionary = corpora.Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # 训练LDA模型
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        passes=passes,
        alpha=alpha,
        per_word_topics=True
    )

    # 提取主题
    topic_words_by_id = {}
    for topic_id in range(num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=top_words_per_topic)
        topic_words_by_id[topic_id] = [word for word, _ in topic_words]

    # 为每个文档分配主题
    with open(output_file, 'w', encoding='utf-8') as out_f:
        with open(input_file, 'r', encoding='utf-8') as in_f:
            for i, line in enumerate(in_f):
                if line.strip():
                    data = json.loads(line)
                    text_content = data.get('text', '')

                    # 获取文档的主题分布
                    doc_bow = dictionary.doc2bow(processed_texts[i])
                    doc_topics = lda_model.get_document_topics(doc_bow)

                    # 找到主要主题
                    main_topic = max(doc_topics, key=lambda x: x[1]) if doc_topics else (0, 0)
                    topic_id = main_topic[0]
                    topic_words = topic_words_by_id[topic_id]
                    topic_keywords = lda_model.show_topic(topic_id, topn=top_words_per_doc)
                    keywords_str = ", ".join([word for word, _ in topic_keywords])

                    # 保存简化格式结果
                    result = {
                        "text": text_content,
                        "topic_id": int(topic_id),
                        "topics": keywords_str,
                        "topic_words": topic_words,
                    }
                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"LDA主题挖掘完成！共挖掘 {num_topics} 个主题")


if __name__ == "__main__":
    input_file = r"E:\主题建模\LSTFM主题建模\数据集\测试数据\ieee_robot_human_1000.jsonl"
    output_file = r"E:\主题建模\LSTFM主题建模\传统方法\传统方法结果汇总_1109之后\ieee_human_robot_1109\lda\jsonl\lda_80_topics_1_1109.jsonl"
    lda_topic_mining(input_file, output_file, num_topics=80)
