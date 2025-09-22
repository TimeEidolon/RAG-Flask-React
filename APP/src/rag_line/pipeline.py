import logging
import json
import pandas as pd

from .es_parsing import ESParsing
from ..domain.entity import FinancialReport
from ..extensions import db
from ..config.rag_config import RunConfig, RunPath
from ..rag_line.pdf_parsing import PDFParser
from ..rag_line.parsed_reports_merging import PageTextPreparation
from ..rag_line.text_splitter import TextSplitter
from ..rag_line.ingestion import VectorDBIngestor, BM25Ingestor
from ..rag_line.questions_processing import QuestionsProcessor
from ..rag_line.tables_serialization import TableSerializer
from _operator import and_, or_
from pathlib import Path
from pyprojroot import here

_log = logging.getLogger(__name__)


def init_metadata():
    """
    初始化元数据，从数据库获取未处理的报告
    
    返回：
        dict: 按公司名分组的报告元数据字典
    """
    metadata_lookup = {}

    reports = FinancialReport.query.filter(and_(
        FinancialReport.file_path.isnot(None),
        or_(
            FinancialReport.vector_flag.is_(None),
            FinancialReport.vector_flag.is_(False)
        )
    )).all()
    for report in reports:
        _name = report.english_name
        if _name not in metadata_lookup:
            metadata_lookup[_name] = []
        metadata_lookup[_name].append(report)

    return metadata_lookup


def init_paths():
    pass


def filter_list(field, _list):
    return [item for item in _list if getattr(item, field) is None or getattr(item, field) == ""]


class Pipeline:
    def __init__(self):
        # 初始化主流程，加载路径和配置
        self.run_config = RunConfig()
        self.paths = RunPath()
        self._convert_json_to_csv_if_needed()

    def _convert_json_to_csv_if_needed(self):
        """
        检查是否存在subset.json且无subset.csv，若是则自动转换为CSV。
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"

        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                df = pd.DataFrame(data)

                df.to_csv(csv_path, index=False)

            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

    @staticmethod
    def download_docling_models():
        # 下载Docling所需模型，避免首次运行时自动下载
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_sequential(self, _obj, _list):
        # 顺序解析PDF报告，输出结构化JSON
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(output_dir=self.paths.parsed_reports_path / _obj)
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path / _obj
        pdf_parser.parse_and_export(_obj, _list)
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """多进程并行解析PDF报告，提升处理效率
        参数：
            chunk_size: 每个worker处理的PDF数
            num_workers: 并发worker数
        """
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(output_dir=self.paths.parsed_reports_path)
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))

        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def serialize_tables(self, max_workers: int = 10):
        """并行处理所有报告中的表格，LLM序列化结构化"""
        serializer = TableSerializer()
        serializer.process_directory_parallel(
            self.paths.parsed_reports_path,
            max_workers=max_workers
        )

    def merge_reports(self, _obj, _list):
        """将复杂JSON报告规整为每页结构化文本，便于后续分块和人工审查"""
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        ptp.process_reports(
            reports=_list,
            output_dir=self.paths.merged_reports_path / _obj,
        )
        print(f"Reports saved to {self.paths.merged_reports_path / _obj}")

    def export_reports_to_markdown(self, _obj, _list):
        """导出规整后报告为markdown，便于人工复核"""
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        ptp.export_to_markdown(
            reports=_list,
            output_dir=self.paths.reports_markdown_path / _obj,
        )
        print(f"Reports saved to {self.paths.reports_markdown_path / _obj}")

    def chunk_reports(self, _obj, _list, include_serialized_tables: bool = False):
        """将规整后报告分块，便于后续向量化和检索"""
        text_splitter = TextSplitter()

        serialized_tables_dir = None
        if include_serialized_tables:
            serialized_tables_dir = self.paths.parsed_reports_path

        text_splitter.split_all_reports(
            _list,
            self.paths.documents_dir / _obj,
            serialized_tables_dir
        )
        print(f"Chunked reports saved to {self.paths.documents_dir / _obj}")

    def create_vector_dbs(self, _obj, _list):
        """从分块报告创建向量数据库"""
        output_dir = self.paths.vector_db_dir

        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(_obj, _list, output_dir)
        print(f"Vector databases created in {output_dir}")

    def es_save(self, _obj, _list):
        """将分块报告存储到es"""
        for temp in _list:
            ESParsing("financial_report").store_text(temp)

    def create_bm25_db(self):
        """从分块报告创建BM25数据库"""
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path

        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")

    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        Returns the next available filename by adding a numbered suffix if the file exists.
        Example: If answers.json exists, returns answers_01.json, etc.
        """
        if not base_path.exists():
            return base_path

        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent

        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename

            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        processor = QuestionsProcessor()

        output_path = self._get_next_available_filename(self.paths.answers_file_path)

        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            team_email=self.run_config.team_email,
            submission_name=self.run_config.submission_name,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")

    @staticmethod
    def do_pipline() -> bool:
        try:
            # 获取要处理的数据
            metadata = init_metadata()

            for key, value in metadata.items():
                _log.info(f"{'#' * 5} 执行【{key}】相关文件转换 {'#' * 5}")
                # 初始化主流程，使用推荐的最佳配置
                pipeline = Pipeline()

                _log.info('1. 解析PDF报告为结构化JSON，输出到 data/json')
                pipeline.parse_pdf_reports_sequential(key, filter_list("json_path", value))

                # 会在 debug/data_01_parsed_reports 的每个表格中新增 "serialized_table" 字段
                # print('2. 序列化表格，输出到 debug/data_01_parsed_reports')
                # pipeline.serialize_tables(max_workers=5)

                _log.info('3. 将解析后的JSON规整为更简单的每页markdown结构，输出到 data/merge')
                pipeline.merge_reports(key, filter_list("merge_path", value))
                #
                _log.info('4. 导出规整后报告为纯markdown文本，仅用于人工复核或全文检索，输出到 data/markdown')
                pipeline.export_reports_to_markdown(key, filter_list("markdown_path", value))

                _log.info('5. 将规整后报告分块，便于后续向量化，输出到 db/chunked')
                pipeline.chunk_reports(key, filter_list("chunked_path", value))

                _log.info('6. 进行es存储')
                pipeline.es_save(key, filter_list("es_flag", value))

                print('7. 从分块报告创建向量数据库，输出到 databases/vector_dbs')
                pipeline.create_vector_dbs(key, value)

                FinancialReport.query.filter(FinancialReport.english_name == key).update({
                    FinancialReport.vector_flag: True,
                })
                db.session.commit()
                _log.info(f'{key} 构建完成')
            return True
        except Exception as e:
            _log.error(f"Rag Build Error: {str(e)}")
            return False
