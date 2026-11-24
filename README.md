# AI 舆情分析助手

输入一个关键词，自动抓取 **7 个中文平台**的公开讨论，用 **BERTopic** 聚类出舆论流派，并把整个能力包装成 **MCP 工具**供大模型直接调用。

```text
关键词 → 7 平台爬取 → 清洗去重 → HanLP 分词 → BGE 向量化
      → UMAP 降维 → HDBSCAN 聚类 → c-TF-IDF 提词 → 舆论流派
```

## 支持平台

| 平台 | 平台 | 平台 |
|---|---|---|
| 小红书 `xhs` | B 站 `bili` | 贴吧 `tieba` |
| 抖音 `dy` | 微博 `wb` | 知乎 `zhihu` |
| 快手 `ks` | | |

## 核心设计

**① 中文 BERTopic 适配**
BERTopic 的 c-TF-IDF 默认按空格切词，对中文完全失效。做法是**先用 HanLP 分词、再以空格拼回**作为 `CountVectorizer` 的输入，同时开 `ngram_range=(1,2)` 让「卢 总」这类被切散的词组能被还原。Embedding 换用 `BAAI/bge-large-zh-v1.5`。

**② 动态领域停用词**
查询词本身会污染聚类关键词（搜「XX手机」，结果所有话题的 top word 都是「XX」）。预处理阶段会用查询词对停用词表做**动态扩展**，把查询词及其分词结果临时加入过滤，保证主题词落在真实观点上。

**③ 长耗时任务的 Agent 化封装**
全网爬取约需 20 分钟，无法在一次工具调用里同步返回。因此 MCP 层拆成两步：

```
start_global_analysis(keyword)  →  立即返回 batch_id，后台 fire-and-forget 跑流水线
check_analysis_status(keyword)  →  轮询 running / failed / success
```

任务状态机持久化在 MongoDB（`running → success | failed`），进程重启不丢任务。工具 docstring 内直接写入了 LLM 输出规范（热度 Mermaid 饼图 + Top3 舆论阵地 + 原始评论不改写），让模型无需额外提示即可产出结构化研判报告。

## 快速开始

```bash
# 依赖
pip install -r requirements.txt
npm install            # EXECJS_RUNTIME=Node，MediaCrawler 需要

# 启动 MongoDB
docker run -d -p 27017:27017 --name mongo mongo:7

# 单次跑通流水线（本地测试）
python pipeline.py     # 默认关键词「台湾地震」
```

`requirements.txt` 里的 `bertopic / sentence-transformers / hdbscan / umap-learn / hanlp` 是分析主干；`torch` 有 CUDA 时自动启用 GPU，无则回落 CPU。国内环境已内置 `HF_ENDPOINT=https://hf-mirror.com` 加速模型下载。

## 接入大模型（MCP）

```bash
python src/mcp_server.py          # stdio 模式，供本地 Agent 拉起
python src/mcp_server.py --web    # SSE 模式，监听 localhost:8000
```

MCP 客户端配置示例：

```json
{
  "mcpServers": {
    "opinion-monitor": {
      "command": "python",
      "args": ["src/mcp_server.py"],
      "cwd": "/path/to/AI-Public-Opinion-Analysis-Assistant"
    }
  }
}
```

## 目录结构

```text
.
├─ pipeline.py              # 主流水线：爬取 → 清洗 → 聚类 → 入库
├─ src/
│  ├─ media_crawler_client.py  # MediaCrawler 封装，统一 7 平台调用
│  ├─ preprocessor.py          # 正则清洗 / SimHash 去重 / HanLP 分词 / 动态停用词
│  ├─ topic_modeler.py         # BERTopic 配置与可视化
│  ├─ database.py              # MongoDB 连接池 + 异步任务状态机
│  ├─ mcp_server.py            # FastMCP 工具层
│  ├─ config.py                # 停用词表加载
│  └─ MediaCrawler/            # 爬虫子模块（第三方）
└─ data/
   ├─ raw/                     # 原始采集
   └─ results/                 # 话题可视化：topics / barchart / hierarchy
```

## 说明

- 需要各平台登录态 Cookie，爬虫层依赖 [MediaCrawler](https://github.com/NanmiCoder/MediaCrawler)，请遵守各平台服务条款与 robots 协议
- 仅用于公开数据的学术研究与技术验证，不做商业采集
- 聚类有效性依赖数据量：有效样本 < 10 条时流水线会主动返回错误而非硬跑
