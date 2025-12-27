import sys
import json
import asyncio
import uvicorn
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# 1. 修复路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.database import db

mcp = FastMCP("OpinionMonitor", dependencies=["uvicorn"])

# === 后台工作线程 ===
async def background_analysis_worker(batch_id: str, keyword: str):
    """
    这是在后台默默工作的“苦力”，它不会阻塞 MCP 的回复。
    """
    print(f"⚙️ [后台任务] 开始执行批次 {batch_id} - {keyword}")
    try:
        # 延迟导入 main，避免启动加载慢
        from pipeline import run_analysis_pipeline

        # 运行全网爬虫+聚类 (耗时 10-20分钟)
        # 注意：main.py 的 run_analysis_pipeline 里原本有 db.save_analysis_result
        # 修改：main.py 返回 structured_topics 即可，由这里统一更新 DB
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
    # 这行代码会瞬间执行完，不会等爬虫
    asyncio.create_task(background_analysis_worker(batch_id, keyword))

    return f"✅ 任务已成功启动！\n批次号: [{batch_id}]\n分析对象: {keyword}\n状态: 全网爬取中(预计耗时10-20分钟)...\n\n请告诉用户：'您的分析任务已在后台运行，请稍后(约20分钟后)让我帮您查询结果。'"


# === 工具 2: 查询状态 (轮询) ===
@mcp.tool()
async def check_analysis_status(keyword: str) -> str:
    """
    【第二步】查询指定话题的分析进度或结果。
    如果任务完成，会返回详细的舆情分析报告。
    如果任务还在跑，会提示继续等待。
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
    #模式1: Web
    # 启动方式: python src/mcp_server.py --web
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        # 这里的 mcp._sse_handler 是 FastMCP 内部处理 SSE 的逻辑
        # 但更推荐直接用 mcp.run() 的 server 参数
        # 为了简单，我们使用 uvicorn 挂载 FastMCP 的 API
        mcp.settings.port = 8000
        mcp.run(transport="sse")  # 这会在 localhost:8000 启动服务
    # 模式 2: 标准输入输出模式 (Stdio) - 默认
    # 启动方式: python src/mcp_server.py
    # 适用: Claude Desktop / Cherry Studio 本地配置
    else:
        mcp.run(transport="stdio")