import os
import json
import torch
from src.preprocessor import OpinionPreprocessor
from src.topic_modeler import TopicAnalyzer

def main():
    # 1. 准备数据路径
    # 请确保 data/raw 目录下有你的 json 文件

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    raw_file_path = "data/raw/search_comments_2025-12-25.json"

    if not os.path.exists(raw_file_path):
        print(f"错误：找不到数据文件 {raw_file_path}")
        # 为了演示，创建一个假数据
        print("正在生成模拟数据以供测试...")
        mock_data = [
                        {"content": "黑神话悟空优化太差了，4060都掉帧", "create_time": 1764163704},
                        {"content": "剧情确实有点魔改，但是美术无敌", "create_time": 1764163705},
                        {"content": "退钱！这根本玩不了", "create_time": 1764163706},
                        {"content": "年度最佳游戏，实至名归", "create_time": 1764163707},
                        {"content": "我显卡炸了，什么破优化", "create_time": 1764163708}
                    ] * 20  # 复制多一点以满足 HDBSCAN 最小聚类要求
        raw_data = mock_data
    else:
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

    # 2. 数据预处理
    print("-" * 30)
    print("Step 1: 数据清洗与分词")
    preprocessor = OpinionPreprocessor()
    # process 返回的是 list of dict
    processed_data = preprocessor.process(raw_data)

    print(f"预处理完成，剩余有效数据: {len(processed_data)} 条")

    # 3. 准备 BERTopic 需要的输入格式
    print("正在构建数据映射...")
    docs_for_clustering = []

    # === 新增：构建映射字典 ===
    # Key: 分词后的字符串 (BERTopic理解的) -> Value: 原始字符串 (人类理解的)
    token_to_raw_mapping = {}

    for item in processed_data:
        # 确保 tokens 存在
        if item.get('tokens'):
            # 1. 生成分词后的字符串 (给模型训练用)
            joined_text = " ".join(item['tokens'])
            docs_for_clustering.append(joined_text)

            # 2. 记录映射关系 (给最后展示用)
            # item['content'] 是完全原始的文本
            # 或者是 item['clean_content'] (清洗过表情和@，但没分词)
            # 建议使用 item['content'] 以展示最原汁原味的内容
            token_to_raw_mapping[joined_text] = item.get('content', '')

    # 检查数据量
    if len(docs_for_clustering) < 15:
        print("数据量太少，无法进行有效聚类 (至少需要 15 条以上)")
        return

    # 4. 初始化并运行聚类分析
    print("-" * 30)
    print("Step 2: 话题聚类与流派挖掘")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"运行设备: {device}")

    analyzer = TopicAnalyzer(device=device)

    # 训练模型 (这里依然传入分词后的数据)
    topics, probs, topic_info = analyzer.fit_transform(docs_for_clustering)

    # 5. 保存结果与可视化
    print("-" * 30)
    print("Step 3: 生成结果与可视化")
    analyzer.save_visualizations(output_dir="data/results")

    # 6. 获取结构化数据
    structured_topics = analyzer.get_topic_details(token_to_raw_map=token_to_raw_mapping)

    # 打印一下看看效果 (修改了这里)
    print(f"\n捕获到 {len(structured_topics)} 个有效意见流派:")
    for t in structured_topics:
        print(f"Topic {t['topic_id']} ({t['name']}): 热度 {t['count']}")
        print(f"  -> 关键词: {t['keywords'][:5]}")

        # === 修改开始：展示 3 条代表性评论 ===
        print(f"  -> 代表性评论 (Top 3):")
        # BERTopic 默认通常返回 3 条，我们遍历显示
        # 使用 enumerate 加序号，并用 replace 去掉换行符让显示更整洁
        for idx, doc in enumerate(t['representative_docs'][:3]):
            clean_doc = str(doc).replace('\n', ' ').strip()
            print(f"     {idx + 1}. {clean_doc[:100]}...")  # 每条限制显示前100个字
        print("-" * 20)  # 加个分隔线更好看
        # === 修改结束 ===

    # 保存 JSON
    with open("data/results/topics_output.json", "w", encoding='utf-8') as f:
        json.dump(structured_topics, f, ensure_ascii=False, indent=2)
    print("\n结果已保存至 data/results/topics_output.json")


if __name__ == "__main__":
    main()