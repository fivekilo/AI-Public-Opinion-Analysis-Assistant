import os


class Config:
    def __init__(self, stopwords_path: str = None):
        # 默认停用词表路径为当前文件夹的 baidu_stopwords.txt
        if stopwords_path is None:
            stopwords_path = os.path.join(os.path.dirname(__file__), 'baidu_stopwords.txt')
        self.stopwords_path = stopwords_path
        self.stopwords = self._load_stopwords()

        # === 新增：领域停用词 (Domain Stopwords) ===
        # 这些词虽然有意义，但在当前话题下是“背景噪音”，必须剔除才能分出流派
        # 针对“小米17Ultra”这个话题：
        self.domain_stopwords = {
            "小米", "手机", "Ultra", "ultra", "17", "17u", "17U","17Ultra",# 相关的领域词汇
            "视频", "点击", "链接", "同款", "https", "b23", "tv"  # 广告常用词
        }
        # 将领域停用词合并到主停用词表中
        self.stopwords.update(self.domain_stopwords)

    def _load_stopwords(self):
        stopwords_set = set()
        if not os.path.exists(self.stopwords_path):
            raise FileNotFoundError(f"Stopwords file not found: {self.stopwords_path}")

        with open(self.stopwords_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:  # 非空词加入集合
                    stopwords_set.add(word)
        return stopwords_set

    def is_stopword(self, word: str) -> bool:
        """检查给定的词是否是停用词"""
        return word in self.stopwords

    @property
    def stopwords_list(self):
        """返回当前配置的停用词集合"""
        return self.stopwords
