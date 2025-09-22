import jieba
import logging
import dashscope
from typing import List, Tuple
from dashscope.api_entities.dashscope_response import Message
from keybert import KeyBERT
from keybert.llm import BaseLLM
from sklearn.feature_extraction.text import CountVectorizer
from ..domain.entity import FinancialReport
from ..extensions import db, es_service

_log = logging.getLogger(__name__)


class QwenTurboLLM(BaseLLM):
    def __init__(self, model_name: str = "qwen-turbo", api_key: str = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key or dashscope.api_key
        self.kwargs = kwargs

    def extract_keywords(self, documents: List[str],
                         candidate_keywords: List[List[str]] = None) -> List[List[Tuple[str, float]]]:
        keywords = []
        try:
            prompt = f"从以下文本中提取最相关的关键词，返回格式为'关键词1', '关键词2', ...，最多10个词：\n{documents}"
            response = dashscope.Generation.call(
                model=self.model_name,
                api_key=self.api_key,
                messages=[Message(role="user", content=prompt)],
                result_format="message"
            )
            if response.status_code == 200:
                keywords = response.output.choices[0].message.content.split(",")
        except Exception as e:
            print(f"Qwen-Turbo调用失败: {str(e)}")
        return keywords


class ESParsing:
    def __init__(self, _index: str):
        self.index = _index

    @staticmethod
    def get_keywords(text: List[str], top_n: int = 10):
        vectorizer = CountVectorizer(tokenizer=jieba.lcut)
        qwen_llm = QwenTurboLLM(temperature=0.3)
        kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

        # 混合关键词提取
        # 1. KeyBERT基础提取
        keybert_kws = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            top_n=20,
            vectorizer=vectorizer,
            use_maxsum=True,
            diversity=0.7
        )

        # 2. Qwen-Turbo增强提取
        llm_kws = qwen_llm.extract_keywords(text)

        # 第二阶段：融合处理
        results = {}

        # 1. KeyBERT关键词直接进入结果集
        for kw, weight in keybert_kws:
            results[kw] = weight * 0.7  # 基础系数

        # 2. LLM关键词处理
        llm_rank_weights = {kw: (len(llm_kws) - i) / len(llm_kws) for i, kw in enumerate(llm_kws)}

        for kw in llm_kws:
            if kw in results:
                # 共同关键词：KeyBERT权重 + LLM排名权重
                results[kw] += 0.3 * llm_rank_weights[kw]
            else:
                # LLM独有关键词：仅使用排名权重
                results[kw] = 0.5 * llm_rank_weights[kw]

        # 3. 标准化处理
        max_weight = max(results.values()) if results else 1
        normalized = {kw: w / max_weight for kw, w in results.items()}

        # 4. 排序处理
        combined_keywords = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [item[0] for item in combined_keywords]

    def store_text(self, item: FinancialReport):
        with open(item.markdown_path, 'rb') as f:
            pdf_text = f.read().decode('utf-8')
        keyword = self.get_keywords([pdf_text])
        doc = {
            'report_id': item.id,
            'stock_code': item.stock_code,
            'company': item.company_name,
            'report_name': item.report_name,
            'publish_date': item.publish_time.isoformat() if item.publish_time else None,
            'content': pdf_text,
            'keywords': keyword
        }
        result = es_service.es.index(index=self.index, id=item.id, body=doc)
        if result['result'] == 'created':
            FinancialReport.query.filter(FinancialReport.id == item.id).update({
                FinancialReport.es_flag: True,
            })
            db.session.commit()
        else:
            _log.error(f"ES存储失败: {result}")
