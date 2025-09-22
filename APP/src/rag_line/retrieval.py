import os
import json
import logging
import numpy as np
import dashscope
from pandas.io import pickle

from .reranking import LLMReranker
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings

_log = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        # 初始化BM25检索器，指定BM25索引和文档目录
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir

    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3,
                                 return_parent_pages: bool = False) -> List[Dict]:
        # 按公司名检索相关文本块，返回BM25分数最高的top_n个块
        document_path = None
        for path in self.documents_dir.glob("*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                if doc["metainfo"]["company_name"] == company_name:
                    document_path = path
                    document = doc
                    break

        if document_path is None:
            raise ValueError(f"No report found with '{company_name}' company name.")

        # 加载对应的BM25索引
        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)

        # 获取文档内容和BM25索引
        document = document
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]

        # 计算BM25分数
        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)

        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]

        retrieval_results = []
        seen_pages = set()

        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])

            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": score,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)

        return retrieval_results


class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, company_name: str,
                 embedding_provider: str = "dashscope"):
        # 初始化向量检索器，加载所有向量库和文档
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.company_name = company_name
        self.embedding_provider = embedding_provider.lower()
        self.all_dbs = self._load_dbs()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        # 根据 embedding_provider 初始化对应的 LLM 客户端
        load_dotenv()
        if self.embedding_provider == "openai":
            llm = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=None,
                max_retries=2
            )
            return llm
        elif self.embedding_provider == "dashscope":
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            return None  # dashscope 不需要 client 对象
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    def _get_embedding(self, text: str):
        # 根据 embedding_provider 获取文本的向量表示
        if self.embedding_provider == "openai":
            embedding = self.llm.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return embedding.data[0].embedding
        elif self.embedding_provider == "dashscope":
            rsp = dashscope.TextEmbedding.call(
                model="text-embedding-v3",
                input=[text]
            )
            # 兼容 dashscope 返回格式，不能用 resp.output，需用 resp['output']
            if 'output' in rsp and 'embeddings' in rsp['output']:
                # 多条输入（本处只有一条）
                emb = rsp['output']['embeddings'][0]
                if emb['embedding'] is None or len(emb['embedding']) == 0:
                    raise RuntimeError(f"DashScope返回的embedding为空，text_index={emb.get('text_index', None)}")
                return emb['embedding']
            elif 'output' in rsp and 'embedding' in rsp['output']:
                # 兼容单条输入格式
                if rsp['output']['embedding'] is None or len(rsp['output']['embedding']) == 0:
                    raise RuntimeError("DashScope返回的embedding为空")
                return rsp['output']['embedding']
            else:
                raise RuntimeError(f"DashScope embedding API返回格式异常: {rsp}")
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    @staticmethod
    def set_up_llm():
        # 静态方法，初始化OpenAI LLM
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm

    def _load_dbs(self):
        # 加载合并的向量库和对应文档，建立映射
        all_dbs = {}
        # 加载合并的FAISS索引
        try:
            # 使用LangChain的FAISS加载向量数据库
            embedding_model = DashScopeEmbeddings(
                model="text-embedding-v3",
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
            )
            faiss_dir = self.vector_db_dir / self.company_name
            merged_vector_db = FAISS.load_local(str(faiss_dir), embedding_model,
                                                index_name=self.company_name,
                                                allow_dangerous_deserialization=True)
            all_dbs['vector_db'] = merged_vector_db
            _log.info(f"Loaded merged FAISS index: {faiss_dir.name}")
        except Exception as e:
            _log.error(f"Error reading merged FAISS index: {e}")
            return all_dbs

        # 加载chunk元信息文件
        chunk_info_files = list(self.vector_db_dir.glob(f'{self.company_name}_chunk_info.json'))
        if not chunk_info_files:
            _log.error(f"No chunk info files found in {self.vector_db_dir}")
            return all_dbs

        try:
            with open(chunk_info_files[0], 'r', encoding='utf-8') as f:
                chunk_info = json.load(f)
                all_dbs['chunk_info'] = chunk_info
            _log.info(f"Loaded chunk info: {chunk_info_files[0].name}")
        except Exception as e:
            _log.error(f"Error reading chunk info: {e}")
            return all_dbs

        _doc_dir = self.documents_dir / self.company_name
        # 获取所有JSON文档路径
        all_documents_paths = list(_doc_dir.glob('*.json'))

        docs = []
        # 为每个文档创建报告对象
        for document_path in all_documents_paths:
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue

            # 校验文档结构
            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue

            docs.append({
                "name": document_path.stem,
                "document": document
            })

        all_dbs['doc'] = docs
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        # 计算两个字符串的余弦相似度（通过嵌入）
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None,
                                 top_n: int = 3, return_parent_pages: bool = False) -> list[dict[str, float | Any]]:

        if self.all_dbs is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")

        vector_db = self.all_dbs["vector_db"]
        chunk_info = self.all_dbs["chunk_info"]

        # 从合并的向量库中搜索，获取更多候选结果
        search_k = min(top_n * 2, len(chunk_info))  # 搜索更多结果以便过滤

        # 使用LangChain的FAISS进行相似性搜索
        search_results = vector_db.similarity_search_with_score(
            query,
            k=search_k
        )

        _results = []
        for item in search_results:
            metadata = item[0].metadata
            if return_parent_pages:
                pass
            else:
                _results.append({
                    "file_name": metadata['file_name'],
                    "page": metadata['parent_page'],
                    "text": item[0].page_content,
                    'score': item[1]
                })
        return _results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        # 检索公司所有文本块，返回全部内容
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break

        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")

        document = target_report["document"]
        pages = document["content"]["pages"]

        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)

        return all_pages


class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, company_name: str):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir, company_name)
        self.reranker = LLMReranker()

    def retrieve_by_company_name(
            self,
            company_name: str,
            query: str,
            llm_reranking_sample_size: int = 28,
            documents_batch_size: int = 2,
            top_n: int = 6,
            llm_weight: float = 0.7,
            return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        Retrieve and rerank documents using hybrid approach.
        
        Args:
            company_name: Name of the company to search documents for
            query: Search query
            llm_reranking_sample_size: Number of initial results to retrieve from vector DB
            documents_batch_size: Number of documents to analyze in one LLM prompt
            top_n: Number of final results to return after reranking
            llm_weight: Weight given to LLM scores (0-1)
            return_parent_pages: Whether to return full pages instead of chunks
            
        Returns:
            List of reranked document dictionaries with scores
        """
        # Get initial results from vector retriever
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )

        # Rerank results using LLM
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )

        return reranked_results[:top_n]
