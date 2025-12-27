import sys
import json
import asyncio
from mcp.server.fastmcp import FastMCP
from pathlib import Path

# 将根目录加入 sys.path，确保能导入 main.py
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# 导入刚才修改好的 pipeline 函数
from main import run_analysis_pipeline
from src.database import db

# 初始化 MCP
mcp = FastMCP("OpinionMonitor", dependencies=["uvicorn"])


@mcp.tool()
async def analyze_topic_realtime(keyword: str, platform: str = "tieba") -> str:
    """
    【主动触发】实时爬取并分析指定话题的舆情。

    Args:
        keyword: 要分析的话题关键词，例如 "小米17Ultra", "黑神话悟空"
        platform: 平台，可选 "tieba", "bilibili", "wb", "xhs", "dy"

    Returns:
        JSON string 包含分析后的流派、热度、代表性评论。
    """
    # 调用 main.py 里的流水线
    result = await run_analysis_pipeline(keywords=keyword, platform=platform)

    if "error" in result:
        return f"分析过程发生错误: {result['error']}"

    # 返回 JSON 给大模型
    return json.dumps(result, default=str, ensure_ascii=False)


@mcp.tool()
async def get_history_report(limit: int = 1) -> str:
    """
    查询数据库中最近的历史分析报告（不爬取新数据）。
    """
    #  database.py 里有 get_latest_analysis 方法
    try:
        res = db.get_latest_analysis()
        if not res:
            return "暂无历史记录"
        return json.dumps(res, default=str, ensure_ascii=False)
    except Exception as e:
        return f"查询数据库失败: {e}"


if __name__ == "__main__":
    # 根据命令行参数决定启动模式
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        print("🌍 启动 Web API 模式 (Port: 8000)...")
        # uvicorn 运行
        mcp.settings.port = 8000
        mcp.run(transport="sse")
    else:
        print("🔌 启动 Stdio 模式 (Claude/Cherry)...")
        mcp.run(transport="stdio")
