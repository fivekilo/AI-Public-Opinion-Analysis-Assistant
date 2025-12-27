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
        # 1. 初始化 HanLP 分词模型
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
            r"\[星星眼\]":"<激动>"
        }
        
        # 3. 定义无意义垃圾文本集合
        self.garbage_words = {"打卡", "哈哈", "哈哈哈", "沙发", "第一", "m", "dd", "转发"}

        # 3.1 抽奖 / 转发抽奖 相关垃圾关键词与模式
        # 只要命中这些基本可以判定为活动/营销类转发评论
        self.lottery_keywords = [
            "转发抽奖",
            "转发并@",
            "转发本条",
            "关注转发",
            "转赞评",
            "转评赞",
            "抽奖",
            "抽签",
            "抽一个",
            "抽几个",
            "抽手机",
            "送手机",
            "送你一部",
        ]
        self.lottery_patterns = [
            re.compile(r"转发.*(抽|送)"),
            re.compile(r"(关注|收藏).*(转发|评论)"),
            re.compile(r"(评论|留言).*(抽|送)"),
        ]

        # 3.2 URL 检测，用于过滤带外链的垃圾文本
        # 这里同时覆盖 http/https 以及常见短链 / 域名形式
        self.url_pattern = re.compile(
            r"(https?://\S+|"
            r"[a-zA-Z0-9_-]+\.(?:com|cn|net|org|io|tv|co|info)/\S*)"
        )

        # 4. 加载停用词表 (模拟加载，实际使用请读取txt文件)
        # 保留 Config 实例，便于后续动态更新领域停用词
        self.config = config.Config()
        self.stop_words = self.config.stopwords_list

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
        (2) 文本清洗：去除@用户，去除 URL 链接，表情包映射
        """
        if not text:
            return ""

        # === 去除 URL 链接 ===
        text = self.url_pattern.sub("", text)
            
        # 去除 @用户 (例如: @绵绵无绝期 )
        # 匹配逻辑：@开头，后面跟非空白字符
        text = re.sub(r"@\S+", "", text)
        
        # 表情包语义映射
        for pattern, meaning in self.emoji_map.items():
            text = re.sub(pattern, meaning, text)
            
        # 去除多余的空白字符
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def _has_url(self, text: str) -> bool:
        """判断文本中是否包含 URL / 外链"""
        if not text:
            return False
        return bool(self.url_pattern.search(text))

    def _is_lottery_spam(self, text: str) -> bool:
        """检测是否为转发抽奖 / 抽签类垃圾文本"""
        if not text:
            return False

        # 先做简单关键词命中
        for kw in self.lottery_keywords:
            if kw in text:
                return True

        # 再做模式匹配（转发 + 抽/送、关注 + 转发 等）
        for pat in self.lottery_patterns:
            if pat.search(text):
                return True

        return False

    def _is_garbage(self, text, has_url: bool = False, raw_text: str = None):
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

        # 过滤明显的转发抽奖 / 抽签类内容（优先用原始文本判断）
        if raw_text and self._is_lottery_spam(raw_text):
            return True
        if self._is_lottery_spam(text):
            return True

        # 过滤包含外部链接的内容（推广 / 引流评论通常都带链接）
        if has_url:
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

    def update_domain_stopwords_from_keyword(self, keyword: str):
        """
        根据传入关键词动态更新领域停用词。
        例如：传入 '小米su7' 时，会把 '小米su7'、'小米'、'su7' 等都加入停用词表。
        """
        if not keyword:
            return

        kw = keyword.strip()
        if not kw:
            return

        terms = set()
        # 原始关键词本身
        terms.add(kw)

        # 使用 HanLP 对关键词分词
        try:
            tokens = self.tokenizer(kw)
        except Exception:
            tokens = list(kw)

        for t in tokens:
            t = str(t).strip()
            if t:
                terms.add(t)

        # 额外用正则抽取连续中文和字母数字片段，增强对混合串的拆分
        for seg in re.findall(r"[\u4e00-\u9fff]+", kw):
            terms.add(seg)
        for seg in re.findall(r"[a-zA-Z0-9]+", kw):
            terms.add(seg)

        self.stop_words.update(terms)

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

            raw_text = formatted_item.get('content', '') or ''

            # 提前在原始文本上做 URL 检测，后续用于垃圾过滤
            has_url = self._has_url(raw_text)

            # 2. 文本清洗
            clean_text = self._clean_text(raw_text)
            formatted_item['clean_content'] = clean_text
            
            # 3. 垃圾过滤 (初步)
            if self._is_garbage(clean_text, has_url=has_url, raw_text=raw_text):
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

    processor = OpinionPreprocessor()
    processor.update_domain_stopwords_from_keyword("小米17Ultra")
    print('小米' in processor.stop_words)
