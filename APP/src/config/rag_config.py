from pathlib import Path
from typing import Literal
from dataclasses import dataclass
from ..extensions import _yaml

"""
RAG（检索增强生成）系统配置模块

本模块定义了RAG系统的核心配置类RunConfig，支持多种不同的检索和生成策略组合。
系统支持以下主要功能：
1. 多种检索方式：向量检索、BM25检索、混合检索
2. 表格序列化：将PDF表格转换为结构化文本
3. 父文档检索：提供更丰富的上下文信息
4. LLM重排序：提升检索结果的相关性
5. 多模型支持：支持多种LLM服务提供商

配置方法说明：
- base_config(): 基础配置，使用向量检索和思维链推理
- parent_document_retrieval_config(): 启用父文档检索的配置
- max_config(): 最大功能配置，启用所有高级功能
- max_no_ser_tab_config(): 不包含表格序列化的最大配置
- max_nst_o3m_config(): 针对o3-mini模型优化的配置
- max_st_o3m_config(): 包含表格序列化的o3-mini配置
- ibm_llama70b_config(): IBM WatsonX + Llama-3.3-70B配置
- ibm_llama8b_config(): IBM WatsonX + Llama-3.1-8B配置
- gemini_thinking_config(): Google Gemini思考模式配置
- gemini_flash_config(): Google Gemini快速模式配置
- max_nst_o3m_config_big_context(): 大上下文o3-mini配置
- ibm_llama70b_config_big_context(): 大上下文IBM Llama配置
- gemini_thinking_config_big_context(): 大上下文Gemini思考配置
- ollama_config_big_context(): Ollama本地模型配置
"""

ProviderType = Literal["openai", "ibm", "gemini", "dashscope", "ollama"]


@dataclass
class RunConfig:
    # 是否使用序列化表格：启用后会将PDF中的表格通过LLM转换为结构化文本，提升表格数据的检索效果
    use_serialized_tables: bool = False

    # 是否启用父文档检索：启用后会返回包含检索文本块的完整页面内容，提供更丰富的上下文信息
    parent_document_retrieval: bool = False

    # 是否使用向量数据库：启用基于语义相似度的向量检索，支持语义理解
    use_vector_dbs: bool = True

    # 是否使用BM25数据库：启用基于关键词匹配的BM25检索，支持精确词汇匹配
    use_bm25_db: bool = False

    # 是否启用LLM重排序：启用后使用大语言模型对检索结果进行相关性重排序，提升检索精度
    llm_reranking: bool = False

    # LLM重排序样本大小：参与重排序的文档数量，影响重排序效果和计算成本
    llm_reranking_sample_size: int = 30

    # 检索返回数量：每次检索返回的最相关文档数量
    top_n_retrieval: int = 10

    # 并行请求数：同时处理的API请求数量，影响处理速度和API限流
    parallel_requests: int = 1

    # 团队邮箱：用于提交结果的团队标识
    team_email: str = "linzhiqing2014@hotmail.com"

    # 提交名称：用于标识不同版本的实验配置
    submission_name: str = "Ilia_Ris vDB + SO CoT"

    # 流程详情：描述当前配置使用的技术栈和处理流程
    pipeline_details: str = ""

    # 是否生成提交文件：是否输出可用于竞赛提交的结果文件
    submission_file: bool = True

    # 是否使用完整上下文：启用后会将所有检索到的文档作为完整上下文输入LLM
    full_context: bool = False

    # API提供商：指定使用的LLM服务提供商（dashscope/openai/gemini/ibm/ollama等）
    api_provider: ProviderType = "dashscope"

    # 回答模型：指定用于生成答案的具体LLM模型名称
    answering_model: str = "qwen-turbo-latest"

    # 配置后缀：用于区分不同配置版本的文件命名后缀
    config_suffix: str = ""

    @staticmethod
    def base_config():
        """基础配置：使用向量检索和思维链推理，适合快速测试和基础功能验证"""
        return RunConfig(
            parallel_requests=10,
            submission_name="Ilia Ris v.0",
            pipeline_details="Custom pdf parsing + vDB + Router + SO CoT; llm = GPT-4o-mini",
            config_suffix="_base"
        )

    @staticmethod
    def parent_document_retrieval_config():
        """父文档检索配置：启用父文档检索功能，提供更丰富的上下文信息，使用GPT-4o模型"""
        return RunConfig(
            parent_document_retrieval=True,
            parallel_requests=20,
            submission_name="Ilia Ris v.1",
            pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = GPT-4o",
            answering_model="gpt-4o-2024-08-06",
            config_suffix="_pdr"
        )

    @staticmethod
    def max_config():
        """最大功能配置：启用所有高级功能，包括表格序列化、父文档检索、LLM重排序，使用GPT-4o模型"""
        return RunConfig(
            use_serialized_tables=True,
            parent_document_retrieval=True,
            llm_reranking=True,
            parallel_requests=20,
            submission_name="Ilia Ris v.2",
            pipeline_details="Custom pdf parsing + table serialization + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = GPT-4o",
            answering_model="gpt-4o-2024-08-06",
            config_suffix="_max"
        )

    @staticmethod
    def max_no_ser_tab_config():
        """无表格序列化的最大配置：启用除表格序列化外的所有高级功能，使用GPT-4o模型"""
        return RunConfig(
            use_serialized_tables=False,
            parent_document_retrieval=True,
            llm_reranking=True,
            parallel_requests=20,
            submission_name="Ilia Ris v.3",
            pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = GPT-4o",
            answering_model="gpt-4o-2024-08-06",
            config_suffix="_max_no_ser_tab"
        )

    @staticmethod
    def qwen_config():
        """针对Qwen模型优化的配置：无表格序列化，启用父文档检索和重排序，使用qwen-turbo模型"""
        return RunConfig(
            use_serialized_tables=False,
            parent_document_retrieval=True,
            llm_reranking=True,
            parallel_requests=4,
            submission_name="Ilia Ris v.4",
            pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = qwen-turbo",
            answering_model="qwen-turbo-latest",
            config_suffix="_qwen3"
        )

    @staticmethod
    def max_st_o3m_config():
        """包含表格序列化的o3-mini配置：启用所有功能，使用o3-mini模型，适合高质量要求场景"""
        return RunConfig(
            use_serialized_tables=True,
            parent_document_retrieval=True,
            llm_reranking=True,
            parallel_requests=25,
            submission_name="Ilia Ris v.5",
            pipeline_details="Custom pdf parsing + tables serialization + Router + vDB + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini",
            answering_model="o3-mini-2025-01-31",
            config_suffix="_max_st_o3m"
        )

    @staticmethod
    def ibm_llama70b_config():
        """IBM WatsonX + Llama-3.3-70B配置：使用IBM云服务和大参数模型，适合企业级应用"""
        return RunConfig(
            use_serialized_tables=False,
            parent_document_retrieval=True,
            llm_reranking=False,
            parallel_requests=10,
            submission_name="Ilia Ris v.6",
            pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT + SO reparser; IBM WatsonX llm = llama-3.3-70b-instruct",
            api_provider="ibm",
            answering_model="meta-llama/llama-3-3-70b-instruct",
            config_suffix="_ibm_llama70b"
        )

    @staticmethod
    def ibm_llama8b_config():
        """IBM WatsonX + Llama-3.1-8B配置：使用IBM云服务和轻量级模型，平衡性能和成本"""
        return RunConfig(
            use_serialized_tables=False,
            parent_document_retrieval=True,
            llm_reranking=False,
            parallel_requests=10,
            submission_name="Ilia Ris v.7",
            pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT + SO reparser; IBM WatsonX llm = llama-3.1-8b-instruct",
            api_provider="ibm",
            answering_model="meta-llama/llama-3-1-8b-instruct",
            config_suffix="_ibm_llama8b"
        )

    @staticmethod
    def gemini_thinking_config():
        """Google Gemini思考模式配置：启用完整上下文和思考模式，适合复杂推理任务"""
        return RunConfig(
            use_serialized_tables=False,
            parent_document_retrieval=True,
            llm_reranking=False,
            parallel_requests=1,
            full_context=True,
            submission_name="Ilia Ris v.8",
            pipeline_details="Custom pdf parsing + Full Context + Router + SO CoT + SO reparser; llm = gemini-2.0-flash-thinking-exp-01-21",
            api_provider="gemini",
            answering_model="gemini-2.0-flash-thinking-exp-01-21",
            config_suffix="_gemini_thinking_fc"
        )

    @staticmethod
    def gemini_flash_config():
        """Google Gemini快速模式配置：启用完整上下文，使用快速模式，平衡速度和效果"""
        return RunConfig(
            use_serialized_tables=False,
            parent_document_retrieval=True,
            llm_reranking=False,
            parallel_requests=1,
            full_context=True,
            submission_name="Ilia Ris v.9",
            pipeline_details="Custom pdf parsing + Full Context + Router + SO CoT + SO reparser; llm = gemini-2.0-flash",
            api_provider="gemini",
            answering_model="gemini-2.0-flash",
            config_suffix="_gemini_flash_fc"
        )

    @staticmethod
    def max_nst_o3m_config_big_context():
        """大上下文o3-mini配置：增加检索数量和重排序样本，适合需要更多上下文信息的复杂查询"""
        return RunConfig(
            use_serialized_tables=False,
            parent_document_retrieval=True,
            llm_reranking=True,
            parallel_requests=5,
            llm_reranking_sample_size=36,
            top_n_retrieval=14,
            submission_name="Ilia Ris v.10",
            pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini; top_n = 14; topn for rerank = 36",
            answering_model="o3-mini-2025-01-31",
            config_suffix="_max_nst_o3m_bc"
        )

    @staticmethod
    def ibm_llama70b_config_big_context():
        """大上下文IBM Llama配置：结合70B大模型和大上下文，适合高精度要求的复杂任务"""
        return RunConfig(
            use_serialized_tables=False,
            parent_document_retrieval=True,
            llm_reranking=True,
            parallel_requests=5,
            llm_reranking_sample_size=36,
            top_n_retrieval=14,
            submission_name="Ilia Ris v.11",
            pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = llama-3.3-70b-instruct; top_n = 14; topn for rerank = 36",
            api_provider="ibm",
            answering_model="meta-llama/llama-3-3-70b-instruct",
            config_suffix="_ibm_llama70b_bc"
        )

    @staticmethod
    def gemini_thinking_config_big_context():
        """大上下文Gemini思考配置：结合思考模式和大上下文，适合需要深度推理的复杂问题"""
        return RunConfig(
            use_serialized_tables=False,
            parent_document_retrieval=True,
            parallel_requests=1,
            top_n_retrieval=30,
            submission_name="Ilia Ris v.12",
            pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = gemini-2.0-flash-thinking-exp-01-21; top_n = 30;",
            api_provider="gemini",
            answering_model="gemini-2.0-flash-thinking-exp-01-21",
            config_suffix="_gemini_thinking_bc"
        )

    @staticmethod
    def ollama_config_big_context():
        """Ollama本地模型配置：使用本地部署的模型，适合数据隐私要求高的场景"""
        return RunConfig(
            use_serialized_tables=False,
            parallel_requests=1,
            top_n_retrieval=30,
            api_provider="ollama"
        )


@dataclass
class RunPath:
    def __init__(self):
        rag_path = _yaml.config['rag_path']
        self.root_path = Path(rag_path["root_path"])
        suffix = "_ser_tab" if bool(rag_path['format_file']['serialized']) else ""

        self.pdf_reports_dir = self.root_path / rag_path['raw_file']['dir_name']

        self.questions_file_path = self.root_path / rag_path['questions']['file_name']
        self.answers_file_path = self.root_path / f"answers{rag_path['config_suffix']}.json"

        self.json_data_path = self.root_path / rag_path['format_file']['json']['dir_name']
        self.parsed_reports_dirname = rag_path['format_file']['json']['parsed_dir_name']
        self.parsed_debug_flag = bool(rag_path['format_file']['json']['debug_flag'])
        self.parsed_reports_debug_dirname = rag_path['format_file']['json']['debug_dir_name']
        self.merged_reports_dirname = rag_path['format_file']['merged']['dir_name']
        self.reports_markdown_dirname = rag_path['format_file']['markdown']['dir_name']
        self.parsed_reports_path = self.json_data_path / self.parsed_reports_dirname
        self.parsed_reports_debug_path = self.json_data_path / self.parsed_reports_debug_dirname
        self.merged_reports_path = self.root_path / self.merged_reports_dirname
        self.reports_markdown_path = self.root_path / self.reports_markdown_dirname

        self.databases_path = self.root_path / f"{rag_path['vectors']['dir_name']}{suffix}"
        self.vector_db_dir = self.databases_path / rag_path['vectors']['db_dir_name']
        self.documents_dir = self.databases_path / rag_path['vectors']['chunked_dir_name']
        self.bm25_db_path = self.databases_path / rag_path['vectors']['bm25_dir_name']
