# src/media_crawler_client.py
import asyncio
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Literal

# 1. 设置 MediaCrawler 的代码路径（改成你本地的实际路径）
THIS_DIR = Path(__file__).resolve().parent   
MEDIA_CRAWLER_ROOT = THIS_DIR / "MediaCrawler"        

if str(MEDIA_CRAWLER_ROOT) not in sys.path:
    # 把 MediaCrawler 项目根目录插入到 sys.path，方便直接 import 它的模块
    sys.path.insert(0, str(MEDIA_CRAWLER_ROOT))
os.chdir(MEDIA_CRAWLER_ROOT)

# 2. 导入 MediaCrawler 内部模块
import config  # type: ignore
from base.base_crawler import AbstractStore  # type: ignore
from main import CrawlerFactory  # type: ignore
import main as mc_main  # 用于调用 async_cleanup，避免冲突给它取别名

from store import xhs as xhs_store  # type: ignore
from store import douyin as dy_store  # type: ignore
from store import kuaishou as ks_store  # type: ignore
from store import bilibili as bili_store  # type: ignore
from store import weibo as wb_store  # type: ignore
from store import tieba as tieba_store  # type: ignore
from store import zhihu as zhihu_store  # type: ignore


Platform = Literal["xhs", "dy", "ks", "bili", "wb", "tieba", "zhihu"]
LoginType = Literal["qrcode", "phone", "cookie"]


class InMemoryStore(AbstractStore):
    """
    自定义的“内存存储”，替换 MediaCrawler 默认的 CSV/JSON/DB 存储。
    """

    def __init__(self) -> None:
        self.contents: List[Dict[str, Any]] = []
        self.comments: List[Dict[str, Any]] = []
        self.creators: List[Dict[str, Any]] = []

    async def store_content(self, content_item: Dict[str, Any]):
        self.contents.append(content_item)

    async def store_comment(self, comment_item: Dict[str, Any]):
        self.comments.append(comment_item)

    async def store_creator(self, creator: Dict[str, Any]):
        self.creators.append(creator)


# 平台 -> 对应的 StoreFactory，用来打补丁
PLATFORM_STORE_FACTORIES = {
    "xhs": xhs_store.XhsStoreFactory,
    "dy": dy_store.DouyinStoreFactory,
    "ks": ks_store.KuaishouStoreFactory,
    "bili": bili_store.BiliStoreFactory,
    "wb": wb_store.WeibostoreFactory,
    "tieba": tieba_store.TieBaStoreFactory,
    "zhihu": zhihu_store.ZhihuStoreFactory,
}


async def crawl_once(
    platform: Platform,
    login_type: LoginType,
    keywords: str,
    CRAWLER_MAX_NOTES_COUNT=20,
    CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES=30,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    调用 MediaCrawler 爬取一次，并把内容/评论/创作者信息作为变量返回。

    返回:
        {
            "contents": [...],
            "comments": [...],
            "creators": [...],
        }
    """
    if platform not in PLATFORM_STORE_FACTORIES:
        raise ValueError(f"不支持的平台: {platform}")

    if login_type not in ("qrcode", "phone", "cookie"):
        raise ValueError(f"不支持的登录方式: {login_type}")

    # 1. 配置 MediaCrawler 全局 config（只改我们关心的几个）
    config.PLATFORM = platform
    config.LOGIN_TYPE = login_type
    config.CRAWLER_TYPE = "search"
    config.KEYWORDS = keywords
    config.CRAWLER_MAX_NOTES_COUNT = CRAWLER_MAX_NOTES_COUNT
    config.CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES

    # 不使用 DB/Excel，避免多余开销；具体值无所谓，后面会用内存存储接管
    config.SAVE_DATA_OPTION = "json"

    # 2. 创建一个内存存储，并把对应平台的 *StoreFactory.create_store* 打补丁
    factory_cls = PLATFORM_STORE_FACTORIES[platform]
    memory_store = InMemoryStore()

    original_create_store = factory_cls.create_store

    def _create_store_override():
        # 所有平台的内容/评论都会进到同一个内存 store 实例里
        return memory_store

    # 注意：原来是 @staticmethod，这里保持静态方法语义
    factory_cls.create_store = staticmethod(_create_store_override)  # type: ignore[assignment]

    # 3. 创建并启动爬虫
    crawler = CrawlerFactory.create_crawler(platform=config.PLATFORM)
    # 让 MediaCrawler 自带的 async_cleanup() 能正确清理这个 crawler
    mc_main.crawler = crawler

    try:
        await crawler.start()
    finally:
        # 4. 清理浏览器 / 连接
        try:
            await mc_main.async_cleanup()
        except Exception:
            # 清理失败不影响结果返回
            pass

        # 5. 恢复 create_store，避免影响后续其他调用
        factory_cls.create_store = original_create_store  # type: ignore[assignment]

    # 6. 把内存中的结果返回给调用方
    return {
        "contents": memory_store.contents,
        "comments": memory_store.comments,
        "creators": memory_store.creators,
    }


# 如果你想在这个模块里直接测试：
if __name__ == "__main__":
    async def _demo():
        result = await crawl_once(
            platform="wb",
            login_type="cookie",
            keywords="高市早苗台湾",
        )
        print("contents:", len(result["contents"]))
        print("comments:", len(result["comments"]))
        print("creators:", len(result["creators"]))

    asyncio.run(_demo())
