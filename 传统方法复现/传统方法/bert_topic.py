import json
import re
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from pathlib import Path


def preprocess_for_bertopic(text):
    """为BERTopic预处理文本"""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def bertopic_topic_mining(
    input_file,
    output_file,
    num_topics=None,
    embedding_model_path=r"E:\主题建模\LSTFM主题建模\Model\all-MiniLM-L12-v2",
    umap_n_neighbors=15,
    umap_n_components=5,
    umap_min_dist=0.0,
    hdbscan_min_cluster_size=5,
    hdbscan_min_samples=1,
    hdbscan_cluster_selection_epsilon=0.1,
    vectorizer_min_df=1,
    vectorizer_max_df=0.95,
    vectorizer_max_features=5000,
    top_n_words=10,
):
    """
    使用BERTopic进行主题挖掘 (Grootendorst, 2022)
    num_topics: None表示自动确定，或指定具体数量
    """

    print("初始化BERTopic模型...")

    # 读取数据
    documents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                text_content = data.get('text', '')
                processed_text = preprocess_for_bertopic(text_content)
                if len(processed_text) > 50:  # 过滤掉太短的文本
                    documents.append({"text": text_content, "processed_text": processed_text})

    print(f"已加载 {len(documents)} 个文档")

    if len(documents) == 0:
        print("错误：没有有效的文本数据！")
        return None

    texts = [document["processed_text"] for document in documents]

    # 配置BERTopic组件 - 优化参数
    sentence_model = SentenceTransformer(embedding_model_path)

    # UMAP配置 - 更平衡的参数
    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric='cosine',
        random_state=42,
        low_memory=False
    )

    # HDBSCAN配置 - 更宽松的聚类参数
    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    # CountVectorizer配置 - 更宽松的文本处理
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=vectorizer_min_df,
        max_df=vectorizer_max_df,
        max_features=vectorizer_max_features
    )

    # 初始化BERTopic模型 - 移除不兼容参数
    try:
        topic_model = BERTopic(
            embedding_model=sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=num_topics,  # 可选的强制主题数量
            calculate_probabilities=True,
            verbose=True,
            top_n_words=top_n_words
        )
    except TypeError as e:
        print(f"检测到参数兼容性问题: {e}")
        print("使用简化参数重新初始化...")
        topic_model = BERTopic(
            embedding_model=sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=num_topics,
            verbose=True
        )

    # 训练模型
    print("正在进行文档嵌入和聚类...")
    try:
        topics, probabilities = topic_model.fit_transform(texts)
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("尝试使用默认参数...")
        # 使用默认参数作为备选
        topic_model_default = BERTopic(verbose=True)
        topics, probabilities = topic_model_default.fit_transform(texts)
        topic_model = topic_model_default

    # 获取主题信息
    topic_info = topic_model.get_topic_info()
    print("主题分布:")
    print(topic_info)

    # 统计主题分布
    unique_topics, counts = np.unique(topics, return_counts=True)
    print("\n主题统计:")
    for topic, count in zip(unique_topics, counts):
        if topic != -1:
            print(f"主题 {topic}: {count} 个文档")
        else:
            print(f"异常值: {count} 个文档")

    # 如果自动生成的主题太少，尝试调整策略
    if len(topic_info) <= 3 and num_topics is None:
        print("\n检测到主题数量过少，尝试使用更敏感的聚类参数...")

        # 重新配置HDBSCAN为更敏感的参数
        hdbscan_model_sensitive = HDBSCAN(
            min_cluster_size=3,
            min_samples=1,
            cluster_selection_epsilon=0.05,
            metric='euclidean',
            cluster_selection_method='leaf',
            prediction_data=True
        )

        topic_model_sensitive = BERTopic(
            embedding_model=sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model_sensitive,
            vectorizer_model=vectorizer_model,
            nr_topics=num_topics,
            verbose=True
        )

        try:
            topics, probabilities = topic_model_sensitive.fit_transform(texts)
            topic_model = topic_model_sensitive
            topic_info = topic_model.get_topic_info()
            print("调整后的主题分布:")
            print(topic_info)
        except Exception as e:
            print(f"调整参数后仍然失败: {e}")

    # 为每个文档生成主题描述
    successful_docs = 0
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for document, topic_id in zip(documents, topics):
            keywords = []
            if topic_id != -1:
                topic_words = topic_model.get_topic(topic_id)
                if topic_words:
                    keywords = [word for word, prob in topic_words[:top_n_words]]
                    topics_str = ", ".join(keywords[:8])
                else:
                    topics_str = "无主题关键词"
            else:
                topics_str = "异常值主题"

            result = {
                "text": document["text"],
                "topic_id": int(topic_id),
                "topics": topics_str,
                "topic_words": keywords,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
            successful_docs += 1

    print(f"成功处理 {successful_docs} 个文档")
    print(f"BERTopic主题挖掘完成！结果已保存到: {output_file}")

    # 保存详细主题信息
    try:
        topic_info.to_csv(output_file.replace('.jsonl', '_topic_info.csv'), index=False)
        print(f"详细主题信息已保存到: {output_file.replace('.jsonl', '_topic_info.csv')}")

        # 保存每个主题的详细关键词
        topic_details = {}
        for topic_id in topic_info['Topic']:
            if topic_id != -1:
                words = topic_model.get_topic(topic_id)
                if words:
                    topic_details[f"topic_{topic_id}"] = [word for word, prob in words[:10]]

        with open(output_file.replace('.jsonl', '_topic_details.json'), 'w', encoding='utf-8') as f:
            json.dump(topic_details, f, ensure_ascii=False, indent=2)
        print(f"主题详细信息已保存到: {output_file.replace('.jsonl', '_topic_details.json')}")

    except Exception as e:
        print(f"保存详细主题信息时出错: {e}")

    return topic_model


# 使用示例
if __name__ == "__main__":
    input_file = r"E:\主题建模\LSTFM主题建模\数据集\测试数据\ieee_robot_human_1000.jsonl"
    output_file = r"E:\主题建模\LSTFM主题建模\传统方法\传统方法结果汇总_1109之后\ieee_human_robot_1109\bertopic\jsonl\bertopic_1109.jsonl"

    # 第一次尝试：自动确定主题数量
    # print("=== 第一次运行：自动确定主题数量 ===")
    # model1 = bertopic_topic_mining(input_file, output_file.replace('.jsonl', '_auto.jsonl'), num_topics=None)

    # 第二次尝试：指定主题数量
    print("\n=== 第二次运行：指定50个主题 ===")
    model2 = bertopic_topic_mining(input_file, output_file.replace('.jsonl', '_80topics.jsonl'), num_topics=80)

    # 第三次尝试：使用更少主题
    # print("\n=== 第三次运行：指定8个主题 ===")
    # model3 = bertopic_topic_mining(input_file, output_file.replace('.jsonl', '_8topics.jsonl'), num_topics=8)
