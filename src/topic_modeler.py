import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from src.config import Config


class TopicAnalyzer:
    def __init__(self, embedding_model_name="BAAI/bge-large-zh-v1.5", device="cuda"):
        """
        初始化分析器
        :param embedding_model_name: 使用 BGE-Large-zh
        :param device: "cuda" 使用显卡加速, "cpu" 使用CPU
        """
        print(f"正在加载 Embedding 模型: {embedding_model_name} (Device: {device})...")
        # 1. 文本向量化 (Embedding)
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)

        # 2. 向量降维 (Dimensionality Reduction)
        # n_neighbors: 控制局部结构与全局结构的平衡，15是默认值，越大越关注全局
        # n_components: 降维后的维度，BERTopic默认5，但为了可视化和聚类效果，通常设为5
        self.umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42  # 固定随机种子，保证结果可复现
        )

        # 3. 聚类算法 (Clustering)
        # min_cluster_size: 核心参数！决定了最小的一个流派有多少人。
        # 设为 10 表示至少有10条相似评论才能算一个观点，否则视为噪音(-1)
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=10,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # 4. 关键词提取 (c-TF-IDF)
        # 专门针对中文优化：因为中文不用空格分词，但BERTopic默认按空格切分。技巧：我们在外部把分词结果用空格连起来。
        # 加载配置里的停用词
        conf = Config()
        stop_words_list = list(conf.stopwords)

        self.vectorizer_model = CountVectorizer(
            stop_words=stop_words_list,  # 在计算关键词时，再次强制过滤掉“小米”等词
            ngram_range=(1, 2),  # === 关键修改：允许 2-gram ===
            # 这意味着 "卢" "总" 分开了没事，它会把 "卢 总" 作为一个词组统计
            min_df=2  # 忽略只出现过1次的生僻词，减少噪音
        )

        # 整合 BERTopic
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            calculate_probabilities=True,  # 计算属于某个话题的概率
            verbose=True,
            language="multilingual"  # 或 "chinese"
        )

    def fit_transform(self, docs_tokens_string, timestamps=None):
        """
        执行聚类分析
        :param docs_tokens_string: list of str, 这里的字符串应该是已分词并用空格连接的，如 ["黑神话 真 好玩", "显卡 太 贵"]
        :param timestamps: list of str/datetime, 可选，用于动态主题分析
        """
        print(f"开始分析 {len(docs_tokens_string)} 条数据...")

        # 核心训练步骤
        topics, probs = self.topic_model.fit_transform(docs_tokens_string)

        # 获取话题信息表
        topic_info = self.topic_model.get_topic_info()
        print("\n=== 分析完成，Top 5 话题流派 ===")
        print(topic_info.head(6))  # 打印前6行 (Row 0 通常是 -1 噪音)

        return topics, probs, topic_info

    def save_visualizations(self, output_dir="data/results"):
        """
        生成并保存可视化图表 (HTML格式)
        增加了容错处理，防止因话题过少导致程序崩溃
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 获取话题数量（不包含噪音 -1）
        topic_info = self.topic_model.get_topic_info()
        # 减去1是因为通常第一行是 Topic -1 (Outlier)
        # 也就是计算真实发现的流派数量
        real_topics_count = len(topic_info) - 1

        print(f"当前发现有效话题数量: {real_topics_count}")

        # 1. 话题距离图 (需要至少 2 个以上的话题才能画出距离)
        try:
            if real_topics_count > 1:
                fig1 = self.topic_model.visualize_topics()
                fig1.write_html(os.path.join(output_dir, "topic_clusters.html"))
                print("-> 已生成话题距离图 topic_clusters.html")
            else:
                print("-> 跳过生成话题距离图 (有效话题少于2个，无法构建坐标系)")
        except Exception as e:
            print(f"-> 话题距离图生成失败: {e}")

        # 2. 条形图 (只要有1个话题就能画)
        try:
            if real_topics_count > 0:
                fig2 = self.topic_model.visualize_barchart(top_n_topics=10)
                fig2.write_html(os.path.join(output_dir, "topic_keywords.html"))
                print("-> 已生成关键词条形图 topic_keywords.html")
            else:
                print("-> 跳过生成条形图 (没有发现有效话题)")
        except Exception as e:
            print(f"-> 条形图生成失败: {e}")

        # 3. 层次聚类图 (需要至少 2 个以上话题)
        try:
            if real_topics_count > 1:
                fig3 = self.topic_model.visualize_hierarchy()
                fig3.write_html(os.path.join(output_dir, "topic_hierarchy.html"))
                print("-> 已生成层次聚类图 topic_hierarchy.html")
        except Exception as e:
            print(f"-> 层次聚类图生成失败: {e}")

        print(f"可视化流程结束，文件保存在 {output_dir}")

    def get_topic_details(self, token_to_raw_map=None):
        """
        获取结构化的话题数据（供后续存入 MongoDB 使用）
        :param token_to_raw_map: (dict) 一个字典，key是分词后的字符串，value是原始评论
        """
        # 获取所有话题的详细信息
        info_df = self.topic_model.get_topic_info()

        results = []
        for index, row in info_df.iterrows():
            topic_id = row['Topic']
            if topic_id == -1: continue  # 跳过噪音类

            # 获取该话题的代表性关键词
            keywords = [word for word, score in self.topic_model.get_topic(topic_id)]

            # 获取代表性文档 (这里默认拿到的是分词后的 string)
            rep_docs_tokenized = self.topic_model.get_representative_docs(topic_id)

            # === 关键修改：尝试还原为原始评论 ===
            rep_docs_raw = []
            if token_to_raw_map:
                for doc in rep_docs_tokenized:
                    # 在字典里找，找不到就还是用分词版的兜底
                    raw_text = token_to_raw_map.get(doc, doc)
                    rep_docs_raw.append(raw_text)
            else:
                rep_docs_raw = rep_docs_tokenized
            # ==================================

            results.append({
                "topic_id": int(topic_id),
                "count": int(row['Count']),
                "name": row['Name'],
                "keywords": keywords,
                "representative_docs": rep_docs_raw  # 这里存的是还原后的人话
            })
        return results