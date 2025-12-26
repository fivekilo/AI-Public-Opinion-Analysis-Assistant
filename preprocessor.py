import json
import re
import datetime
import time
from simhash import Simhash
import hanlp
import config
import json

class OpinionPreprocessor:
    def __init__(self):
        # 1. 初始化 HanLP 分词模型 (使用轻量级模型以加快演示速度，生产环境可用大模型)
        print("正在加载 HanLP 模型...")
        try:
            # 这里使用的是粗颗粒度分词，适合舆情分析
            self.tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        except Exception as e:
            print(f"模型加载失败，请确保已安装hanlp并联网: {e}")
            self.tokenizer = lambda x: list(x) # fallback

        # 2. 定义表情包语义映射字典
        self.emoji_map = {
            r"\[doge\]": "<讽刺>",
            r"\[二哈\]": "<调侃>",
            r"\[微笑\]": "<敷衍>",
            r"\[嗑瓜子\]": "<围观>",
            r"\[吃瓜\]": "<围观>",
            r"\[流汗\]": "<无语>",
            r"\[大哭\]": "<悲伤>"
        }
        
        # 3. 定义无意义垃圾文本集合
        self.garbage_words = {"打卡", "哈哈", "哈哈哈", "沙发", "第一", "m", "dd", "转发"}
        
        # 4. 加载停用词表 (模拟加载，实际使用请读取txt文件)
        Config=config.Config()
        self.stop_words =Config.stopwords_list

    def _format_data(self, item):
        """
        (1) 基础格式化：时间戳转换、数值类型修正
        """
        try:
            # 修正数值类型
            if isinstance(item.get('sub_comment_count'), str):
                item['sub_comment_count'] = int(item['sub_comment_count'])
            
            # 修正时间戳 (Unix 10位或13位 转 标准字符串)
            ts = item.get('create_time')
            if ts:
                # 如果是毫秒级时间戳(13位)，转为秒
                if len(str(ts)) > 10:
                    ts = ts / 1000
                dt = datetime.datetime.fromtimestamp(ts)
                item['format_time'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                
            return item
        except Exception as e:
            print(f"格式化错误: {e}")
            return None

    def _clean_text(self, text):
        """
        (2) 文本清洗：去除@用户，表情包映射
        """
        if not text:
            return ""
            
        # 去除 @用户 (例如: @绵绵无绝期 )
        # 匹配逻辑：@开头，后面跟非空白字符
        text = re.sub(r"@\S+", "", text)
        
        # 表情包语义映射
        for pattern, meaning in self.emoji_map.items():
            text = re.sub(pattern, meaning, text)
            
        # 去除多余的空白字符
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def _is_garbage(self, text):
        """
        (3) 部分去垃圾逻辑：过短或无意义文本
        """
        # 过滤极短文本（清洗后长度小于2）
        if len(text) < 2:
            return True
        # 过滤纯数字或纯标点
        if re.match(r"^[\d\W_]+$", text):
            return True
        # 过滤特定垃圾词
        if text in self.garbage_words:
            return True
        return False

    def deduplicate_data(self, data_list):
        """
        (3) 去重逻辑：使用 SimHash
        返回：去重后的数据列表（包含相似合并统计）
        """
        unique_data = []
        seen_hashes = [] # 存储元组 (simhash_obj, raw_item_index)
        
        print(f"开始去重，原始数量: {len(data_list)}")
        
        for item in data_list:
            text = item['clean_content']
            if not text: 
                continue

            # 计算 SimHash
            item_hash = Simhash(text)
            is_duplicate = False
            
            # 与已存在的hash进行海明距离比较
            # 距离通常 <= 3 视为相似
            for existing_hash, index in seen_hashes:
                if item_hash.distance(existing_hash) <= 3:
                    # 发现重复/相似：增加计数，不添加新条目
                    unique_data[index]['similarity_count'] = unique_data[index].get('similarity_count', 1) + 1
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_hashes.append((item_hash, len(unique_data)))
                item['similarity_count'] = 1
                unique_data.append(item)
                
        print(f"去重完成，剩余数量: {len(unique_data)}")
        return unique_data

    def _tokenize(self, text):
        """
        (4) 分词：HanLP + 停用词过滤
        """
        # HanLP 分词
        tokens = self.tokenizer(text)
        
        # 过滤停用词
        filtered_tokens = [t for t in tokens if t not in self.stop_words and len(t.strip()) > 0]
        
        return filtered_tokens

    def process(self, raw_data_list):
        """
        主处理管道
        """
        step1_data = []
        
        # 第一阶段：格式化与清洗
        for item in raw_data_list:
            # 1. 基础格式化
            formatted_item = self._format_data(item)
            if not formatted_item: continue
            
            # 2. 文本清洗
            clean_text = self._clean_text(formatted_item.get('content', ''))
            formatted_item['clean_content'] = clean_text
            
            # 3. 垃圾过滤 (初步)
            if self._is_garbage(clean_text):
                continue
                
            step1_data.append(formatted_item)
            
        # 第二阶段：SimHash 去重
        deduplicated_data = self.deduplicate_data(step1_data)
        
        # 第三阶段：分词 (仅对去重后的数据操作，节省算力)
        for item in deduplicated_data:
            item['tokens'] = self._tokenize(item['clean_content'])
            
        return deduplicated_data

# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟你图片中的原始数据
    with open("data\小米17Ultra\\bili\search_comments_2025-12-25.json",'r',encoding='utf-8') as f:
        raw_data = json.load(f)

    processor = OpinionPreprocessor()
    result = processor.process(raw_data)

    # 打印结果展示
    print("\n--- 处理结果 ---")
    for res in result:
        print(f"原内容: {res['content']}")
        print(f"清洗后: {res['clean_content']}")
        print(f"时间: {res['format_time']}")
        print(f"相似聚合数: {res['similarity_count']}")
        print(f"分词结果: {res['tokens']}")
        print("-" * 50)