import os
import json
import torch
import asyncio
from src.preprocessor import OpinionPreprocessor
from src.topic_modeler import TopicAnalyzer
from src.media_crawler_client import crawl_once
from pathlib import Path
from src.database import db

async def fetech_data(platform,login_type,keywords):
    result = await crawl_once(
        platform=platform,          # xhs / dy / ks / bili / wb / tieba / zhihu
        login_type=login_type,     # qrcode / phone / cookie
        keywords=keywords,
    )
    contents = result["contents"]
    comments = result["comments"]
    # 在这里直接把 contents/comments 丢给你的后续预处理逻辑即可
    return contents, comments

def main():
    # 1. 爬取数据
    # 请确保 data/raw 目录下有你的 json 文件

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    _,raw_data=asyncio.run(fetech_data(platform='bili',login_type='cookie',keywords="小米su7"))
    print(f"数据收集完成，有效数据: {len(raw_data)} 条")

    # 2. 数据预处理
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
    THIS_DIR=Path(__file__).resolve().parent
    analyzer.save_visualizations(output_dir=THIS_DIR/"data/results")

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
    with open(THIS_DIR/"data/results/topics_output.json", "w", encoding='utf-8') as f:
        json.dump(structured_topics, f, ensure_ascii=False, indent=2)
    print("\n结果已保存至 data/results/topics_output.json")

    # 7. 结果保存数据库
    print("\n[Phase 6] 生成结果...")

    # A. 存原始数据 (可选：如果你想存清洗后的数据)
    # processed_data 里面包含了 content, clean_content, tokens 等
    # 我们可以把它们存进去，作为“数据资产”
    db.save_raw_data(processed_data)
    # B. 存分析结果
    structured_topics = analyzer.get_topic_details(token_to_raw_map=token_to_raw_mapping)

    # 存入 MongoDB
    batch_id = db.save_analysis_result(structured_topics)

    print(f"\n✅ 全流程执行完毕！数据已入库，批次ID: {batch_id}")


if __name__ == "__main__":
    main()