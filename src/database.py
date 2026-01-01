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

        # 创建索引
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

    def init_analysis_task(self, keyword):
        """
        【异步模式步骤1】初始化一个任务，标记为 'running'
        """
        batch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        document = {
            "batch_id": batch_id,
            "keyword": keyword,
            "status": "running",  # 关键状态：running, success, failed
            "created_at": datetime.datetime.now(),
            "topics": [],
            "error_msg": ""
        }
        self.col_topics.insert_one(document)
        return batch_id

    def update_task_success(self, batch_id, topic_data):
        """
        【异步模式步骤2】任务成功，更新状态和结果
        """
        self.col_topics.update_one(
            {"batch_id": batch_id},
            {"$set": {
                "status": "success",
                "topics": topic_data,
                "finished_at": datetime.datetime.now()
            }}
        )

    def update_task_failed(self, batch_id, error_msg):
        """
        【异步模式步骤3】任务失败，记录错误
        """
        self.col_topics.update_one(
            {"batch_id": batch_id},
            {"$set": {
                "status": "failed",
                "error_msg": error_msg,
                "finished_at": datetime.datetime.now()
            }}
        )

    def get_task_status(self, keyword):
        """
        【异步模式步骤4】根据关键词查询最新任务的状态
        """
        return self.col_topics.find_one(
            {"keyword": keyword},
            sort=[("created_at", -1)]
        )

# 单例模式供import
db = DatabaseManager()