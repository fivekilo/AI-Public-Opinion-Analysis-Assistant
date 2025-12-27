from pymongo import MongoClient, ASCENDING, DESCENDING
import datetime

class DatabaseManager:
    def __init__(self, db_name="opinion_analysis_db", uri="mongodb://localhost:27017/"):
        """
        初始化 MongoDB 连接
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

        # 定义两个集合(表)：
        # 1. raw_comments: 存爬虫下来的原始数据 (带上下文)
        # 2. topic_results: 存 BERTopic 分析后的结论
        self.col_raw = self.db["raw_comments"]
        self.col_topics = self.db["topic_results"]

        # 创建索引以加速查询 (可选，但推荐)
        # 按时间倒序查
        self.col_raw.create_index([("create_time", DESCENDING)])
        # 按批次ID查分析结果
        self.col_topics.create_index([("batch_id", DESCENDING)])

    def save_raw_data(self, data_list):
        """
        批量保存原始爬虫数据
        """
        if not data_list:
            return 0

        # 这里的 data_list 就是你 preprocessor 处理完的那些 dict
        # 使用 insert_many 批量插入，速度快
        # 注意：MongoDB 自动生成的 _id 有可能重复(极少)，
        # 严谨做法是用 update_one upsert，但数据量少可以直接 insert
        try:
            result = self.col_raw.insert_many(data_list)
            print(f"[DB] 成功存入 {len(result.inserted_ids)} 条原始数据")
            return len(result.inserted_ids)
        except Exception as e:
            print(f"[DB] 存储原始数据报错: {e}")
            return 0

    def save_analysis_result(self, topic_data, batch_id=None,keyword=None): # 新增 keyword 参数
        """
        保存 BERTopic 的分析结果
        """
        if not batch_id:
            batch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        document = {
            "batch_id": batch_id,
            "created_at": datetime.datetime.now(),
            "keyword": keyword,  # <--- 记录详细主题
            "topics": topic_data  # 这里直接存 list[dict]
        }

        self.col_topics.insert_one(document)
        print(f"[DB] 分析结果已保存，批次号: {batch_id}")
        return batch_id

    def get_latest_analysis(self):
        """
        供 MCP 调用：获取最新的一次分析结果
        """
        return self.col_topics.find_one(sort=[("created_at", -1)])

    def search_raw_comments(self, keyword, limit=5):
        """
        供 MCP 调用：查具体的评论证据
        """
        query = {"content": {"$regex": keyword}}
        cursor = self.col_raw.find(query).limit(limit)
        return list(cursor)

# 单例模式：在其他文件直接 from src.database import db 即可
db = DatabaseManager()