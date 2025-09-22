import os
import time
import logging
import json

from tabulate import tabulate
from pathlib import Path
from typing import List, Dict
from .es_parsing import ESParsing
from ..domain.entity import FinancialReport
from ..extensions import db
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter, FormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
from docling.datamodel.base_models import InputFormat
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

_log = logging.getLogger(__name__)


def _process_chunk(pdf_paths, pdf_backend, output_dir, num_threads, metadata_lookup, debug_data_path):
    """
    辅助函数：在单独进程中处理一批PDF文件。
    参数：
        pdf_paths: 待处理PDF文件路径列表
        pdf_backend: PDF解析后端
        output_dir: 输出目录
        num_threads: 线程数
        metadata_lookup: 元数据字典
        debug_data_path: 调试数据保存路径
    返回：
        处理结果字符串
    """
    # 创建新的解析器实例
    parser = PDFParser(
        pdf_backend=pdf_backend,
        output_dir=output_dir,
        num_threads=num_threads
    )
    parser.metadata_lookup = metadata_lookup
    parser.debug_data_path = debug_data_path
    parser.parse_and_export(pdf_paths)
    return f"Processed {len(pdf_paths)} PDFs."


class PDFParser:
    def __init__(
            self,
            pdf_backend=DoclingParseV2DocumentBackend,
            output_dir: Path = Path("./parsed_pdfs"),
            num_threads: int = None
    ):
        """
        PDF解析器初始化
        参数：
            pdf_backend: PDF解析后端
            output_dir: 输出目录
            num_threads: 线程数
        """
        self.pdf_backend = pdf_backend
        self.output_dir = output_dir
        self.doc_converter = self._create_document_converter()
        self.num_threads = num_threads
        self.debug_data_path = None

        # 设置线程数环境变量
        if self.num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.num_threads)

    def _create_document_converter(self) -> DocumentConverter:
        """
        创建并返回带有默认pipeline选项的DocumentConverter。
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # 启用OCR
        ocr_options = EasyOcrOptions(lang=['en'], force_full_page_ocr=False, use_gpu=False)
        pipeline_options.ocr_options = ocr_options
        pipeline_options.do_table_structure = True  # 启用表格结构识别
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        format_options = {
            InputFormat.PDF: FormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
                backend=self.pdf_backend
            )
        }

        return DocumentConverter(format_options=format_options)

    def convert_documents(self, id_path_dict: Dict[str, Path]) -> Dict[str, ConversionResult]:
        """
        批量转换PDF文档，返回转换结果迭代器。
        参数：
            input_doc_paths: PDF文件路径列表
        返回：
            转换结果迭代器
        """
        conv_results = {}
        for _id, _path in id_path_dict.items():
            conv_results[_id] = self.doc_converter.convert(_path)
        return conv_results

    def process_documents(self, _obj, conv_results: Dict[str, ConversionResult]):
        """
        处理转换结果，保存为JSON并更新数据库状态。
        参数：
            conv_results: 转换结果迭代器
        返回：
            (成功数, 失败数)
        """
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        success = []
        failure = []

        for _id, conv_res in conv_results.items():
            if conv_res.status == ConversionStatus.SUCCESS:
                success.append(_id)
                processor = JsonReportProcessor(debug_data_path=self.debug_data_path)

                # 归一化文档数据，确保页码连续
                data = conv_res.document.export_to_dict()
                normalized_data = self._normalize_page_sequence(data)

                processed_report = processor.assemble_report(conv_res, normalized_data)
                file_name = self.output_dir / f"{conv_res.input.file.stem}.json"
                if self.output_dir is not None:
                    with file_name.open("w", encoding="utf-8") as fp:
                        json.dump(processed_report, fp, indent=2, ensure_ascii=False)

                    temp = FinancialReport.query.filter(FinancialReport.id == _id).first()
                    if temp:
                        temp.json_path = file_name
                        db.session.commit()
            else:
                failure.append(_id)
                _log.info(f"Document {conv_res.input.file} failed to convert.")

        _log.info(f"Processed {len(success) + len(failure)} docs, of which {len(failure)} failed")
        return success, failure

    def _normalize_page_sequence(self, data: dict) -> dict:
        """
        确保content中的页码连续，不连续则补空页。
        参数：
            data: 文档内容字典
        返回：
            归一化后的内容字典
        """
        if 'content' not in data:
            return data

        # 拷贝数据，避免原数据被修改
        normalized_data = data.copy()

        # 获取已有页码，找出最大页码
        existing_pages = {page['page'] for page in data['content']}
        max_page = max(existing_pages)

        # 空页模板
        empty_page_template = {
            "content": [],
            "page_dimensions": {}  # 可根据需要设置默认尺寸
        }

        # 构建完整页码序列
        new_content = []
        for page_num in range(1, max_page + 1):
            # 查找现有页，否则补空页
            page_content = next(
                (page for page in data['content'] if page['page'] == page_num),
                {"page": page_num, **empty_page_template}
            )
            new_content.append(page_content)

        normalized_data['content'] = new_content
        return normalized_data

    def parse_and_export(self, _obj, _list):
        """
        主流程：解析并导出PDF为JSON。
        参数：
            input_doc_paths: PDF文件路径列表
            doc_dir: PDF文件目录（可选）
        """
        id_path_dict = {item.id: item.file_path for item in _list}
        total_docs = len(id_path_dict.values())
        if total_docs != 0:
            _log.info(f"Starting to process {_obj} : {total_docs} documents")
            start_time = time.time()
            conv_results = self.convert_documents(id_path_dict)
            success, failure = self.process_documents(_obj, conv_results=conv_results)

            if len(failure) > 0:
                error_message = f"Failed converting {_obj} : {len(failure)} out of {total_docs} documents."
                failed_docs = "Paths of failed docs:\n" + '\n'.join(
                    str(path) for key, path in id_path_dict.items() if key in failure)
                _log.error(error_message)
                _log.error(failed_docs)
                raise RuntimeError(error_message)

            _log.info(
                f"{_obj} : Completed in {time.time() - start_time:.2f} seconds. Successfully converted {len(success)}/{total_docs} documents.\n")

    def parse_and_export_parallel(
            self,
            input_doc_paths: List[Path] = None,
            doc_dir: Path = None,
            optimal_workers: int = 10,
            chunk_size: int = None
    ):
        """Parse PDF files in parallel using multiple processes.

        Args:
            input_doc_paths: List of paths to PDF files to process
            doc_dir: Directory containing PDF files (used if input_doc_paths is None)
            optimal_workers: Number of worker processes to use. If None, uses CPU count.
        """

        # Get input paths if not provided
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_pdfs = len(input_doc_paths)
        _log.info(f"Starting parallel processing of {total_pdfs} documents")

        cpu_count = multiprocessing.cpu_count()

        # Calculate optimal workers if not specified
        if optimal_workers is None:
            optimal_workers = min(cpu_count, total_pdfs)

        if chunk_size is None:
            # Calculate chunk size (ensure at least 1)
            chunk_size = max(1, total_pdfs // optimal_workers)

        # Split documents into chunks
        chunks = [
            input_doc_paths[i: i + chunk_size]
            for i in range(0, total_pdfs, chunk_size)
        ]

        start_time = time.time()
        processed_count = 0

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # Schedule all tasks
            futures = [
                executor.submit(
                    _process_chunk,
                    chunk,
                    self.pdf_backend,
                    self.output_dir,
                    self.num_threads,
                    self.metadata_lookup,
                    self.debug_data_path
                )
                for chunk in chunks
            ]

            # Wait for completion and log results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    processed_count += int(result.split()[1])  # Extract number from "Processed X PDFs"
                    _log.info(f"{'#' * 50}\n{result} ({processed_count}/{total_pdfs} total)\n{'#' * 50}")
                except Exception as e:
                    _log.error(f"Error processing chunk: {str(e)}")
                    raise

        elapsed_time = time.time() - start_time
        _log.info(f"Parallel processing completed in {elapsed_time:.2f} seconds.")


class JsonReportProcessor:
    def __init__(self, debug_data_path: Path = None):
        """
        JSON报告处理器初始化
        参数：
            debug_data_path: 调试数据保存路径
        """
        self.debug_data_path = debug_data_path

    def assemble_report(self, conv_result, normalized_data=None):
        """
        组装最终报告，包含元信息、正文、表格、图片等。
        参数：
            conv_result: 单个PDF的转换结果
            normalized_data: 归一化后的文档内容（可选）
        返回：
            结构化JSON报告
        """
        data = normalized_data if normalized_data is not None else conv_result.document.export_to_dict()
        assembled_report = {
            'metainfo': self.assemble_metainfo(data),
            'content': self.assemble_content(data),
            'tables': self.assemble_tables(conv_result.document, data),
            'pictures': self.assemble_pictures(data)
        }
        self.debug_data(data)
        return assembled_report

    def assemble_metainfo(self, data):
        """
        组装元信息，包括sha1、页数、文本块数、表格数等。
        参数：
            data: 文档内容字典
        返回：
            metainfo字典
        """
        metainfo = {
            'file_name': data['name'],
            'pages_amount': len(data.get('pages', [])),
            'text_blocks_amount': len(data.get('texts', [])),
            'tables_amount': len(data.get('tables', [])),
            'pictures_amount': len(data.get('pictures', [])),
            'equations_amount': len(data.get('equations', [])),
            'footnotes_amount': len([t for t in data.get('texts', []) if t.get('label') == 'footnote'])
        }
        return metainfo

    def process_table(self, table_data):
        """
        表格处理逻辑（可自定义扩展）
        参数：
            table_data: 单个表格的原始数据
        返回：
            处理后的表格内容（示例为字符串）
        """
        return 'processed_table_content'

    def debug_data(self, data):
        """
        若设置debug路径，则保存中间数据到指定目录，便于调试。
        参数：
            data: 需保存的文档内容
        """
        if self.debug_data_path is None:
            return
        doc_name = data['name']
        path = self.debug_data_path / f"{doc_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def expand_groups(self, body_children, groups):
        """
        展开正文中的group引用，补充group信息。
        参数：
            body_children: 正文children列表
            groups: group对象列表
        返回：
            展开后的children列表
        """
        expanded_children = []

        for item in body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'groups':
                    group = groups[ref_num]
                    group_id = ref_num
                    group_name = group.get('name', '')
                    group_label = group.get('label', '')

                    for child in group['children']:
                        child_copy = child.copy()
                        child_copy['group_id'] = group_id
                        child_copy['group_name'] = group_name
                        child_copy['group_label'] = group_label
                        expanded_children.append(child_copy)
                else:
                    expanded_children.append(item)
            else:
                expanded_children.append(item)

        return expanded_children

    def _process_text_reference(self, ref_num, data):
        """
        处理正文中的文本引用，生成内容项。
        参数：
            ref_num: 文本在data['texts']中的索引
            data: 文档内容字典
        返回：
            包含文本内容及类型等信息的字典
        """
        text_item = data['texts'][ref_num]
        item_type = text_item['label']
        content_item = {
            'text': text_item.get('text', ''),
            'type': item_type,
            'text_id': ref_num
        }

        # 仅当原文与text不同才保留orig
        orig_content = text_item.get('orig', '')
        if orig_content != text_item.get('text', ''):
            content_item['orig'] = orig_content

        # 补充其他字段
        if 'enumerated' in text_item:
            content_item['enumerated'] = text_item['enumerated']
        if 'marker' in text_item:
            content_item['marker'] = text_item['marker']

        return content_item

    def assemble_content(self, data):
        """
        组装正文内容，按页分组。
        参数：
            data: 文档内容字典
        返回：
            按页分组的内容列表
        """
        pages = {}
        # 展开group引用
        body_children = data['body']['children']
        groups = data.get('groups', [])
        expanded_body_children = self.expand_groups(body_children, groups)

        # 处理正文内容
        for item in expanded_body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'texts':
                    text_item = data['texts'][ref_num]
                    content_item = self._process_text_reference(ref_num, data)

                    # 若有group信息则补充
                    if 'group_id' in item:
                        content_item['group_id'] = item['group_id']
                        content_item['group_name'] = item['group_name']
                        content_item['group_label'] = item['group_label']

                    # 从prov获取页码
                    if 'prov' in text_item and text_item['prov']:
                        page_num = text_item['prov'][0]['page_no']

                        # 初始化页
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': text_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)

                elif ref_type == 'tables':
                    table_item = data['tables'][ref_num]
                    content_item = {
                        'type': 'table',
                        'table_id': ref_num
                    }

                    if 'prov' in table_item and table_item['prov']:
                        page_num = table_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': table_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)

                elif ref_type == 'pictures':
                    picture_item = data['pictures'][ref_num]
                    content_item = {
                        'type': 'picture',
                        'picture_id': ref_num
                    }

                    if 'prov' in picture_item and picture_item['prov']:
                        page_num = picture_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': picture_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)

        # 按页码排序输出
        sorted_pages = [pages[page_num] for page_num in sorted(pages.keys())]
        return sorted_pages

    def assemble_tables(self, doc, data):
        """
        组装表格内容，包含markdown、html、json等多种格式。
        参数：
            doc: 文档对象（含表格结构）
            data: 文档内容字典
        返回：
            结构化表格信息列表
        """
        assembled_tables = []
        for i, table in enumerate(doc.tables):
            table_json_obj = table.model_dump()
            table_md = self._table_to_md(table_json_obj)
            table_html = table.export_to_html(doc=doc)

            table_data = data['tables'][i]
            table_page_num = table_data['prov'][0]['page_no']
            table_bbox = table_data['prov'][0]['bbox']
            table_bbox = [
                table_bbox['l'],
                table_bbox['t'],
                table_bbox['r'],
                table_bbox['b']
            ]

            # 获取表格行列数
            nrows = table_data['data']['num_rows']
            ncols = table_data['data']['num_cols']

            ref_num = table_data['self_ref'].split('/')[-1]
            ref_num = int(ref_num)

            table_obj = {
                'table_id': ref_num,
                'page': table_page_num,
                'bbox': table_bbox,
                '#-rows': nrows,
                '#-cols': ncols,
                'markdown': table_md,
                'html': table_html,
                'json': table_json_obj
            }
            assembled_tables.append(table_obj)
        return assembled_tables

    def _table_to_md(self, table):
        """
        将表格数据转为markdown格式。
        参数：
            table: 单个表格的json结构
        返回：
            markdown字符串
        """
        # 提取单元格文本
        table_data = []
        for row in table['data']['grid']:
            table_row = [cell['text'] for cell in row]
            table_data.append(table_row)

        # 判断是否有表头
        if len(table_data) > 1 and len(table_data[0]) > 0:
            try:
                md_table = tabulate(
                    table_data[1:], headers=table_data[0], tablefmt="github"
                )
            except ValueError:
                md_table = tabulate(
                    table_data[1:],
                    headers=table_data[0],
                    tablefmt="github",
                    disable_numparse=True,
                )
        else:
            md_table = tabulate(table_data, tablefmt="github")

        return md_table

    def assemble_pictures(self, data):
        """
        组装图片内容，包含图片所在页、位置、子内容等。
        参数：
            data: 文档内容字典
        返回：
            结构化图片信息列表
        """
        assembled_pictures = []
        for i, picture in enumerate(data['pictures']):
            children_list = self._process_picture_block(picture, data)

            ref_num = picture['self_ref'].split('/')[-1]
            ref_num = int(ref_num)

            picture_page_num = picture['prov'][0]['page_no']
            picture_bbox = picture['prov'][0]['bbox']
            picture_bbox = [
                picture_bbox['l'],
                picture_bbox['t'],
                picture_bbox['r'],
                picture_bbox['b']
            ]

            picture_obj = {
                'picture_id': ref_num,
                'page': picture_page_num,
                'bbox': picture_bbox,
                'children': children_list,
            }
            assembled_pictures.append(picture_obj)
        return assembled_pictures

    def _process_picture_block(self, picture, data):
        """
        处理图片块的子内容（如图片下的文本等）。
        参数：
            picture: 单个图片对象
            data: 文档内容字典
        返回：
            图片子内容（如文本块）列表
        """
        children_list = []

        for item in picture['children']:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'texts':
                    content_item = self._process_text_reference(ref_num, data)

                    children_list.append(content_item)

        return children_list
