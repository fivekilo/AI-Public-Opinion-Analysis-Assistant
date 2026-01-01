import os
os.environ["EXECJS_RUNTIME"] = "Node"
import json
import torch
import asyncio
from src.preprocessor import OpinionPreprocessor
from src.topic_modeler import TopicAnalyzer
from src.media_crawler_client import crawl_once
from pathlib import Path
from src.database import db

THIS_DIR=Path(__file__).resolve().parent
# 设置 HuggingFace 镜像加速下载嵌入模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

async def fetech_data(platform, login_type, keywords):
    try:
        print(f"开始爬取平台: {platform}")
        result = await crawl_once(
            platform=platform,
            login_type=login_type,
            keywords=keywords,
        )
        contents = result.get("contents", [])
        comments = result.get("comments", [])
        print(f"平台 {platform} 爬取完成，获取到 {len(contents)} 条内容，{len(contents)+len(comments)} 条评论")
        return contents, comments
    except Exception as e:
        print(f"平台 {platform} 爬取失败: {e}")
        return [], []  # 发生异常时返回空列表，确保程序不崩溃

    
async def run_analysis_pipeline(keywords: str, login_type: str = "cookie"):
    """
    【核心流水线】接收关键词和平台，执行全套分析，返回结果字典。
    """
    print(f"\n🚀 [Pipeline Start] 启动实时分析: {keywords} ")

    # --- 1. 爬取数据 ---
    print(f"🕷️ [Step 1] 正在爬取数据...")
    login_type = "cookie"
    target_platforms = ['xhs', 'dy', 'ks', 'bili', 'wb', 'tieba', 'zhihu']
    contents=[]
    comments=[]
    for plat in target_platforms:
        content, comment = await(
            fetech_data(plat, login_type, keywords)
        )
        contents.extend(content)
        comments.extend(comment)

    total_raw_count = len(contents) + len(comments)
    if total_raw_count == 0:
        return {"error": "爬虫未获取到任何数据，请检查关键词或Cookie是否有效。"}

    print(f"✅ 数据收集完成，有效数据: {total_raw_count} 条")

    # --- 2. 数据预处理 ---
    print(f"🧹 [Step 2] 数据预处理...")
    preprocessor = OpinionPreprocessor()

    # 根据关键词更新领域停用词
    if hasattr(preprocessor, 'update_domain_stopwords_from_keyword'):
        preprocessor.update_domain_stopwords_from_keyword(keyword=keywords)

    # 处理评论和内容
    processed_comments = preprocessor.process(comments)
    processed_contents = preprocessor.process(contents)

    # 合并用于聚类
    all_processed_items = processed_comments + processed_contents

    print(f"✅ 预处理完成，剩余有效数据: {len(all_processed_items)} 条")

    # --- 3. 构建映射与 Embedding 输入 ---
    print("🔄 [Step 3] 构建数据映射...")
    docs_for_clustering = []
    token_to_raw_mapping = {}

    for item in all_processed_items:
        if item.get('tokens'):
            # 生成分词后的字符串 (给模型训练用)
            joined_text = " ".join(item['tokens'])
            docs_for_clustering.append(joined_text)

            # 记录映射关系 (给最后展示用)
            token_to_raw_mapping[joined_text] = item.get('content', '')

    # 检查数据量
    if len(docs_for_clustering) < 10:  # 可按需调整
        return {"error": f"有效数据过少 ({len(docs_for_clustering)}条)，无法进行聚类分析。"}

    # --- 4. 聚类分析 ---
    print("🧠 [Step 4] BERTopic 聚类分析...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   运行设备: {device}")

    try:
        analyzer = TopicAnalyzer(device=device)
        # 训练模型
        analyzer.fit_transform(docs_for_clustering)

        # 保存可视化图表
        output_dir = THIS_DIR / "data/results"
        analyzer.save_visualizations(output_dir=str(output_dir))

        # 获取结构化数据
        structured_topics = analyzer.get_topic_details(token_to_raw_map=token_to_raw_mapping)

        # 保存 JSON 文件备份
        with open(output_dir / "topics_output.json", "w", encoding='utf-8') as f:
            json.dump(structured_topics, f, ensure_ascii=False, indent=2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"聚类分析过程出错: {str(e)}"}

    # --- 5. 数据库存储 ---
    print("💾 [Step 5] 存入 MongoDB...")

    # A. 存原始数据
    db.save_raw_data(all_processed_items)

    # B. 存分析结果 (已经交由database.py异步控制)
    # batch_id = db.save_analysis_result(structured_topics, keyword=keywords)

    print(f"✅ [Pipeline End] 全流程结束")

    # --- 6. 构造返回给 MCP/大模型的数据 ---
    return {
        "status": "success",
        # "batch_id": batch_id,#(已经交由database.py异步控制)
        "keyword": keywords,
        "total_analyzed": len(all_processed_items),
        "topic_count": len(structured_topics),
        "topics": structured_topics,  # 这里包含了关键词和代表性评论
        "visualization_path": str(output_dir)
    }

# main 用于本地测试
if __name__ == "__main__":
    asyncio.run(run_analysis_pipeline(keywords="台湾地震"))