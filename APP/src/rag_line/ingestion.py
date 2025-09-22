import os
import json
import pickle
import dashscope
from typing import List, Union
from pathlib import Path

from tqdm import tqdm
from rank_bm25 import BM25Okapi
from tenacity import retry, wait_fixed, stop_after_attempt
from dashscope import TextEmbedding
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings


# BM25Ingestor：BM25索引构建与保存工具
class BM25Ingestor:
    def __init__(self):
        pass

    @classmethod
    def create_bm25_index(cls, chunks: List[str]) -> BM25Okapi:
        """从文本块列表创建BM25索引"""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """
        批量处理所有报告，生成并保存BM25索引。
        参数：
            all_reports_dir (Path): 存放JSON报告的目录
            output_dir (Path): 保存BM25索引的目录
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # 加载报告
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            # 提取文本块并创建BM25索引
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = BM25Ingestor.create_bm25_index(text_chunks)

            # 保存BM25索引，文件名用sha1_name
            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)

        print(f"Processed {len(all_report_paths)} reports")


# VectorDBIngestor：向量库构建与保存工具
class VectorDBIngestor:
    def __init__(self):
        # 初始化DashScope API Key
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, text: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")

        # 保证 input 为一维字符串列表或单个字符串
        if isinstance(text, list):
            text_chunks = text
        else:
            text_chunks = [text]

        # 类型检查，确保每一项都是字符串
        if not all(isinstance(x, str) for x in text_chunks):
            raise ValueError("所有待嵌入文本必须为字符串类型！实际类型: {}".format([type(x) for x in text_chunks]))

        # 过滤空字符串
        text_chunks = [x for x in text_chunks if x.strip()]
        if not text_chunks:
            raise ValueError("所有待嵌入文本均为空字符串！")
        embeddings = []
        MAX_BATCH_SIZE = 10
        LOG_FILE = 'embedding_error.log'
        for i in range(0, len(text_chunks), MAX_BATCH_SIZE):
            batch = text_chunks[i:i + MAX_BATCH_SIZE]
            resp = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v3,
                input=batch
            )
            # 兼容单条和多条输入
            if 'output' in resp and 'embeddings' in resp['output']:
                for emb in resp['output']['embeddings']:
                    if emb['embedding'] is None or len(emb['embedding']) == 0:
                        error_text = batch[emb.text_index] if hasattr(emb, 'text_index') else None
                        with open(LOG_FILE, 'a', encoding='utf-8') as f:
                            f.write(
                                f"DashScope返回的embedding为空，text_index={getattr(emb, 'text_index', None)}，文本内容如下：\n{error_text}\n{'-' * 60}\n")
                        raise RuntimeError(
                            f"DashScope返回的embedding为空，text_index={getattr(emb, 'text_index', None)}，文本内容已写入 {LOG_FILE}")
                    embeddings.append(emb['embedding'])
            elif 'output' in resp and 'embedding' in resp['output']:
                if resp['output']['embedding'] is None or len(resp['output']['embedding']) == 0:
                    with open(LOG_FILE, 'a', encoding='utf-8') as f:
                        f.write(
                            "DashScope返回的embedding为空，文本内容如下：\n{}\n{}\n".format(batch[0] if batch else None,
                                                                                          '-' * 60))
                    raise RuntimeError("DashScope返回的embedding为空，文本内容已写入 {}".format(LOG_FILE))
                embeddings.append(resp['output']['embedding'])
            else:
                raise RuntimeError(f"DashScope embedding API返回格式异常: {resp}")
        return embeddings

    @classmethod
    def _create_vector_db(cls, texts: List[str], embeddings: List[List[float]], metadatas: List[dict] = None) -> FAISS:
        """
        使用LangChain的FAISS创建向量数据库
        
        参数:
            texts: List[str] - 文本块列表
            embeddings: List[List[float]] - 对应的嵌入向量列表
            metadatas: List[dict] - 元数据列表（可选）
            
        返回:
            FAISS - LangChain的FAISS向量存储对象
        """
        # 创建DashScope嵌入模型
        embedding_model = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=dashscope.api_key
        )

        # 使用LangChain的FAISS创建向量存储
        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=embedding_model,
            metadatas=metadatas
        )

        return vector_store

    def embedding_chunk(self, report: dict):
        # 针对单份报告，提取文本块并生成向量库
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        embeddings = self._get_embeddings(text_chunks)
        return embeddings

    def process_reports(self, _obj, _list, output_dir: Path):
        # 批量处理所有报告，生成并保存faiss向量库
        output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化收集所有数据
        all_texts = []  # 存储所有文本块
        all_embeddings = []  # 存储所有嵌入向量
        all_chunk_info = []  # 存储每个chunk的元信息（报告ID、chunk索引等）

        for item in tqdm(_list, desc=f"{_obj} : Processing reports"):
            with open(item.chunked_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)

            # 获取当前报告的文本块
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            embeddings = self._get_embeddings(text_chunks)

            # 收集所有文本块、embeddings和元信息
            all_texts.extend(text_chunks)
            all_embeddings.extend(embeddings)

            # 记录每个chunk的元信息
            for chunk_idx, _ in enumerate(text_chunks):
                all_chunk_info.append({
                    'file_name': report_data['metainfo']['file_name'],
                    'chunk_idx': chunk_idx,
                    'parent_page': report_data['content']['chunks'][chunk_idx]['page'],
                    'file_path': str(item.chunked_path)
                })

        # 创建合并的FAISS索引
        if all_embeddings:
            _db = VectorDBIngestor._create_vector_db(all_texts, all_embeddings, all_chunk_info)
            # 保存FAISS索引
            _db.save_local(str(output_dir / _obj), index_name=_obj)

            # 保存chunk元信息，用于后续检索时定位原始文档
            chunk_info_path = output_dir / f"{_obj}_chunk_info.json"
            with open(chunk_info_path, 'w', encoding='utf-8') as f:
                json.dump(all_chunk_info, f, ensure_ascii=False, indent=2)

        print(f"{_obj} Processed {len(_list)} reports, total chunks: {len(all_embeddings)}")
