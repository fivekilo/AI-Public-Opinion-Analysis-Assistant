import sys
import json
import asyncio
import uvicorn
from pathlib import Path
from mcp.server.fastmcp import FastMCP

#修复路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.database import db

mcp = FastMCP("OpinionMonitor", dependencies=["uvicorn"])

# === 后台工作线程 ===
async def background_analysis_worker(batch_id: str, keyword: str):
    """
    在后台默默工作的分析进程，它不会阻塞 MCP 的回复。
    """
    print(f"⚙️ [后台任务] 开始执行批次 {batch_id} - {keyword}")
    try:
        # 延迟导入 main，避免启动加载慢
        from pipeline import run_analysis_pipeline

        # 运行全网爬虫+聚类 (耗时 20分钟)
        # 修改：main.py 返回 structured_topics 即可，由这里统一更新分析的报告存入 DB
        result = await run_analysis_pipeline(keywords=keyword)

        if "error" in result:
            print(f"❌ [后台任务] 失败: {result['error']}")
            db.update_task_failed(batch_id, result['error'])
        else:
            print(f"✅ [后台任务] 成功完成！")
            # result['topics'] 是结构化的流派数据
            db.update_task_success(batch_id, result['topics'])

    except Exception as e:
        print(f"❌ [后台任务] 崩溃: {e}")
        db.update_task_failed(batch_id, str(e))


# === 工具 1: 启动任务 (立即返回) ===
@mcp.tool()
async def start_global_analysis(keyword: str) -> str:
    """
    【第一步】启动全网舆情分析任务。
    由于爬取7大平台数据量巨大，此工具会立即返回一个“任务ID”。
    请告知用户任务已后台启动，并让用户稍后使用 'check_analysis_status' 查询结果。
    """
    # 1. 在数据库创建一条 "running" 状态的记录
    batch_id = db.init_analysis_task(keyword)

    # 2. 创建后台非阻塞任务 (Fire-and-Forget)
    asyncio.create_task(background_analysis_worker(batch_id, keyword))

    return f"✅ 任务已成功启动！\n批次号: [{batch_id}]\n分析对象: {keyword}\n状态: 全网爬取中(预计耗时20分钟)...\n\n请告诉用户：'您的分析任务已在后台运行，请稍后(约20分钟后)让我帮您查询结果。'"


# === 工具 2: 查询状态 (轮询) ===
@mcp.tool()
async def check_analysis_status(keyword: str) -> str:
    """
    【第二步】查询指定话题的分析进度或结果。
    【LLM 输出规范 (IMPORTANT)】
    如果任务还在跑，会提示继续等待。
    如果任务完成，当工具返回 'success' 状态且包含 JSON 数据时，请务必严格按照以下格式生成回答：
    1. **舆情可视化 (Mermaid)**
       - 请根据 JSON 中 `topics` 列表的前 5 个话题的 `count` (热度) 数据，绘制一个 **Mermaid 饼图 (Pie Chart)**。
       - 格式示例：
         ```mermaid
         pie title 舆情热度分布 (Top 5)
             "话题关键词摘要" : 123
             "另一个话题" : 88
         ```
    2. **核心舆论阵地 (Top 3)**
       - 从 `topics` 中提取热度最高的前 3 个聚类。
       - 对每个聚类展示：
         - **主题名称**：根据 `keywords` 总结成一个简短的 4-6 字标题。
         - **代表性原声**：从 `representative_docs` 中展示 所有3 条评论，**必须保留原汁原味**，不要改写。
    3. **深度总结**
       - 结合所有数据，给出2-3段落的整体舆情研判。

    Args:
        keyword: 要查询的话题关键词
    """
    task = db.get_task_status(keyword)

    if not task:
        return "未找到该话题的分析记录，请先调用 'start_global_analysis' 启动任务。"

    status = task.get("status")
    batch_id = task.get("batch_id")

    if status == "running":
        return f"⏳ 任务 [{batch_id}] 正在火速运行中...\n目前状态: 爬虫/聚类计算中\n请再耐心等待一会儿。"

    elif status == "failed":
        return f"❌ 任务 [{batch_id}] 执行失败。\n错误原因: {task.get('error_msg')}"

    elif status == "success":
        # 任务成功，返回数据给大模型写报告
        result_json = {
            "status": "finished",
            "batch_id": batch_id,
            "keyword": keyword,
            "analyze_time": task.get("finished_at"),
            "topics": task.get("topics")
        }
        return json.dumps(result_json, default=str, ensure_ascii=False)

    return "未知状态"


if __name__ == "__main__":
    print("MCP Server (Async Mode) starting...", file=sys.stderr)
    # 模式 1: Web
    # 启动方式: 在源代码根目录下：python src/mcp_server.py --web
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        # 但更推荐直接用 mcp.run() 的 server 参数
        mcp.settings.port = 8000
        mcp.run(transport="sse")  # 这会在 localhost:8000 启动服务
    # 模式 2: 标准输入输出模式 (Stdio)
    # 启动方式: 在源代码根目录下：python src/mcp_server.py
    else:
        mcp.run(transport="stdio")