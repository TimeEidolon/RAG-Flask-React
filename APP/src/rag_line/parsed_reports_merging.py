import re
from pathlib import Path
from ..domain.entity import FinancialReport
from ..extensions import db
import json

"""
年报页面文本规整与清洗模块

本模块主要负责处理PDF解析后的原始JSON报告数据，将其转换为结构化的文本格式。
主要功能包括：

1. 文本清洗：去除OCR错误、格式标记和特殊字符
2. 结构化处理：按规则重新组织页面内容，处理标题、段落、表格等
3. 表格处理：支持markdown表格和序列化表格两种格式
4. 列表处理：处理有序列表、无序列表和复选框
5. 脚注处理：正确关联脚注与正文内容
6. 格式输出：支持JSON和Markdown两种输出格式

核心类：
- PageTextPreparation: 页面文本准备器，负责主要的文本处理逻辑

使用场景：
- PDF年报解析后的数据清洗和格式化
- 为RAG系统准备结构化的文本数据
- 生成可读性强的Markdown文档
"""


class PageTextPreparation:
    """
    负责按规则清洗和格式化页面块，处理表格、列表、脚注等连续分组。
    主要功能：
    1. 清洗PDF解析后的原始文本，去除OCR错误和格式标记
    2. 按结构化规则重新组织页面内容
    3. 处理表格、列表、脚注等复杂元素的格式化
    4. 支持序列化表格的插入和渲染
    """

    def __init__(self, use_serialized_tables: bool = False, serialized_tables_instead_of_markdown: bool = False):
        """
        初始化页面文本准备器
        
        参数：
            use_serialized_tables: 是否使用序列化表格，启用后会将表格转换为结构化文本
            serialized_tables_instead_of_markdown: 是否用序列化文本完全替代markdown表格
        """
        self.use_serialized_tables = use_serialized_tables
        self.serialized_tables_instead_of_markdown = serialized_tables_instead_of_markdown

    def process_reports(self, reports, output_dir: Path = None):
        """
        批量处理目录或路径列表下的报告，返回规整后报告列表，可选保存到输出目录
        
        参数：
            reports: 文档信息列表
            output_dir: 输出目录路径，如果提供则保存处理后的报告
            
        返回：
            List[dict]: 处理后的报告列表，每个报告包含metainfo和content
        """
        all_reports = []
        for _report in reports:
            with open(_report.json_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)

            full_report_text = self.process_report(report_data)
            report = {"metainfo": report_data['metainfo'], "content": full_report_text}
            all_reports.append(report)

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                file_name = output_dir / f'{_report.report_name.split(".")[0]}.json'
                with open(file_name, 'w', encoding='utf-8') as fp:
                    json.dump(report, fp, indent=2, ensure_ascii=False)
                temp = FinancialReport.query.filter(FinancialReport.id == _report.id).first()
                if temp:
                    temp.merge_path = file_name
                    db.session.commit()
        return all_reports

    def process_report(self, report_data):
        """
        处理单份报告，返回规整后的每页文本列表，若有修正会打印提示
        
        参数：
            report_data: 包含metainfo、content、tables等字段的报告数据字典
            
        返回：
            dict: 处理后的报告，包含chunks和pages字段
        """
        self.report_data = report_data
        processed_pages = []
        total_corrections = 0
        corrections_list = []

        for page_content in self.report_data["content"]:
            page_number = page_content["page"]
            page_text = self.prepare_page_text(page_number)
            cleaned_text, corrections_count, corrections = self._clean_text(page_text)
            total_corrections += corrections_count
            corrections_list.extend(corrections)
            page_data = {
                "page": page_number,
                "text": cleaned_text
            }
            processed_pages.append(page_data)

        if total_corrections > 0:
            print(
                f"Fixed {total_corrections} occurrences in the file "
                f"{self.report_data['metainfo']['sha1_name']}"
            )
            print(corrections_list[:30])

        processed_report = {
            "chunks": None,
            "pages": processed_pages
        }

        return processed_report

    def prepare_page_text(self, page_number):
        """
        主流程：处理页面块并组装为字符串
        
        参数：
            page_number: 页码
            
        返回：
            str: 处理后的页面文本内容
        """
        page_data = self._get_page_data(page_number)
        if not page_data or "content" not in page_data:
            return ""

        blocks = page_data["content"]

        filtered_blocks = self._filter_blocks(blocks)
        final_blocks = self._apply_formatting_rules(filtered_blocks)

        if final_blocks:
            final_blocks[0] = final_blocks[0].lstrip()
            final_blocks[-1] = final_blocks[-1].rstrip()

        return "\n".join(final_blocks)

    def _get_page_data(self, page_number):
        """
        根据页码返回页面字典
        
        参数：
            page_number: 页码
            
        返回：
            dict: 页面数据字典，包含page和content字段
        """
        all_pages = self.report_data.get("content", [])
        for page in all_pages:
            if page.get("page") == page_number:
                return page
        return None

    def _filter_blocks(self, blocks):
        """
        移除不需要的块类型，如页脚、图片
        
        参数：
            blocks: 页面块列表
            
        返回：
            List[dict]: 过滤后的块列表
        """
        ignored_types = {"page_footer", "picture"}
        filtered_blocks = []
        for block in blocks:
            block_type = block.get("type")
            if block_type in ignored_types:
                continue
            filtered_blocks.append(block)
        return filtered_blocks

    def _clean_text(self, text):
        """
        用正则清洗文本，统计修正次数
        
        参数：
            text: 原始文本
            
        返回：
            Tuple[str, int, List[Tuple]]: (清洗后的文本, 修正次数, 修正记录列表)
        """
        command_mapping = {
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'period': '.',
            'comma': ',',
            'colon': ":",
            'hyphen': "-",
            'percent': '%',
            'dollar': '$',
            'space': ' ',
            'plus': '+',
            'minus': '-',
            'slash': '/',
            'asterisk': '*',
            'lparen': '(',
            'rparen': ')',
            'parenright': ')',
            'parenleft': '(',
            'wedge.1_E': '',
        }

        recognized_commands = "|".join(command_mapping.keys())
        slash_command_pattern = rf"/({recognized_commands})(\.pl\.tnum|\.tnum\.pl|\.pl|\.tnum|\.case|\.sups)"

        occurrences_amount = len(re.findall(slash_command_pattern, text))
        occurrences_amount += len(re.findall(r'glyph<[^>]*>', text))
        occurrences_amount += len(re.findall(r'/([A-Z])\.cap', text))

        corrections = []

        def replace_command(match):
            base_command = match.group(1)
            replacement = command_mapping.get(base_command)
            if replacement is not None:
                corrections.append((match.group(0), replacement))
            return replacement if replacement is not None else match.group(0)

        def replace_glyph(match):
            corrections.append((match.group(0), ''))
            return ''

        def replace_cap(match):
            original = match.group(0)
            replacement = match.group(1)
            corrections.append((original, replacement))
            return replacement

        text = re.sub(slash_command_pattern, replace_command, text)
        text = re.sub(r'glyph<[^>]*>', replace_glyph, text)
        text = re.sub(r'/([A-Z])\.cap', replace_cap, text)

        return text, occurrences_amount, corrections

    def _block_ends_with_colon(self, block):
        """
        判断块文本是否以冒号结尾，仅针对特定类型
        
        参数：
            block: 文本块字典
            
        返回：
            bool: 是否以冒号结尾
        """
        block_type = block.get("type")
        text = block.get("text", "").rstrip()
        if block_type in {"text", "caption", "section_header", "paragraph"}:
            return text.endswith(":")
        return False

    def _apply_formatting_rules(self, blocks):
        """
        按格式化规则处理块，合并表格、列表、脚注等
        
        参数：
            blocks: 页面块列表
            
        返回：
            List[str]: 格式化后的文本块列表
        """
        page_header_in_first_3 = False
        section_header_in_first_3 = False
        for blk in blocks[:3]:
            if blk["type"] == "page_header":
                page_header_in_first_3 = True
            if blk["type"] == "section_header":
                section_header_in_first_3 = True

        final_blocks = []
        first_section_header_index = 0

        i = 0
        n = len(blocks)

        while i < n:
            block = blocks[i]
            block_type = block.get("type")
            text = block.get("text", "").strip()

            # Handle headers
            if block_type == "page_header":
                prefix = "\n# " if i < 3 else "\n## "
                final_blocks.append(f"{prefix}{text}\n")
                i += 1
                continue

            if block_type == "section_header":
                first_section_header_index += 1
                if (
                        first_section_header_index == 1
                        and i < 3
                        and not page_header_in_first_3
                ):
                    prefix = "\n# "
                else:
                    prefix = "\n## "
                final_blocks.append(f"{prefix}{text}\n")
                i += 1
                continue

            if block_type == "paragraph":
                if self._block_ends_with_colon(block) and i + 1 < n:
                    next_block_type = blocks[i + 1].get("type")
                    if next_block_type not in ("table", "list_item"):
                        final_blocks.append(f"\n### {text}\n")
                        i += 1
                        continue
                else:
                    final_blocks.append(f"\n### {text}\n")
                    i += 1
                    continue

            # Handle table groups
            if block_type == "table" or (
                    self._block_ends_with_colon(block)
                    and i + 1 < n
                    and blocks[i + 1].get("type") == "table"
            ):
                group_blocks = []
                header_for_table = None
                if self._block_ends_with_colon(block) and i + 1 < n:
                    header_for_table = block
                    table_block = blocks[i + 1]
                    i += 2
                else:
                    table_block = block
                    i += 1

                if header_for_table:
                    group_blocks.append(header_for_table)
                group_blocks.append(table_block)

                footnote_candidates_start = i
                if i < n:
                    maybe_text_block = blocks[i]
                    if maybe_text_block.get("type") == "text":
                        if (i + 1 < n) and (blocks[i + 1].get("type") == "footnote"):
                            group_blocks.append(maybe_text_block)
                            i += 1

                while i < n and blocks[i].get("type") == "footnote":
                    group_blocks.append(blocks[i])
                    i += 1

                group_text = self._render_table_group(group_blocks)
                final_blocks.append(group_text)
                continue

            # Handle list groups
            if block_type == "list_item" or (
                    self._block_ends_with_colon(block)
                    and i + 1 < n
                    and blocks[i + 1].get("type") == "list_item"
            ):
                group_blocks = []
                if self._block_ends_with_colon(block) and i + 1 < n:
                    header_for_list = block
                    i += 1
                    group_blocks.append(header_for_list)

                while i < n and blocks[i].get("type") == "list_item":
                    group_blocks.append(blocks[i])
                    i += 1

                if i < n and blocks[i].get("type") == "text":
                    if (i + 1 < n) and (blocks[i + 1].get("type") == "footnote"):
                        group_blocks.append(blocks[i])
                        i += 1

                while i < n and blocks[i].get("type") == "footnote":
                    group_blocks.append(blocks[i])
                    i += 1

                group_text = self._render_list_group(group_blocks)
                final_blocks.append(group_text)
                continue

            # Handle normal blocks
            if block_type in (
                    "text",
                    "caption",
                    "footnote",
                    "checkbox_selected",
                    "checkbox_unselected",
                    "formula",
            ):
                if not text.strip():
                    i += 1
                    continue
                else:
                    final_blocks.append(f"{text}\n")
                    i += 1
                continue

            raise ValueError(f"Unknown block type: {block_type}")

        return final_blocks

    def _render_table_group(self, group_blocks):
        """
        渲染表格组，包含可选的标题、文本和脚注
        
        参数：
            group_blocks: 表格组块列表
            
        返回：
            str: 渲染后的表格组文本
        """
        chunk = []
        for blk in group_blocks:
            blk_type = blk.get("type")
            blk_text = blk.get("text", "").strip()
            if blk_type in {"text", "caption", "section_header", "paragraph"}:
                chunk.append(f"{blk_text}\n")

            elif blk_type == "table":
                table_id = blk.get("table_id")
                if table_id is None:
                    continue
                table_markdown = self._get_table_by_id(table_id)
                chunk.append(f"{table_markdown}\n")

            elif blk_type == "footnote":
                chunk.append(f"{blk_text}\n")

            elif blk_type == "text":
                chunk.append(f"{blk_text}\n")

            else:
                raise ValueError(f"Unexpected block type in table group: {blk_type}")

        return "\n" + "".join(chunk) + "\n"

    def _render_list_group(self, group_blocks):
        """
        渲染列表组，包含可选的标题、文本和脚注
        
        参数：
            group_blocks: 列表组块列表
            
        返回：
            str: 渲染后的列表组文本
        """
        chunk = []
        for blk in group_blocks:
            blk_type = blk.get("type")
            blk_text = blk.get("text", "").strip()
            if blk_type in {"text", "caption", "section_header", "paragraph"}:
                chunk.append(f"{blk_text}\n")

            elif blk_type == "list_item":
                chunk.append(f"- {blk_text}\n")

            elif blk_type == "footnote":
                chunk.append(f"{blk_text}\n")

            elif blk_type == "checkbox_selected":
                chunk.append(f"[x] {blk_text}\n")

            elif blk_type == "checkbox_unselected":
                chunk.append(f"[ ] {blk_text}\n")

            else:
                chunk.append(f"{blk_text}\n")

        return "\n" + "".join(chunk) + "\n"

    def _get_table_by_id(self, table_id):
        """
        根据ID从报告数据中获取表格表示
        
        参数：
            table_id: 表格ID
            
        返回：
            str: 表格的markdown或序列化文本表示
            
        异常：
            ValueError: 当找不到指定ID的表格时抛出
        """
        for t in self.report_data.get("tables", []):
            if t.get("table_id") == table_id:
                if self.use_serialized_tables:
                    return self._get_serialized_table_text(t, self.serialized_tables_instead_of_markdown)
                return t.get("markdown", "")
        raise ValueError(f"Table with ID={table_id} not found in report_data!")

    def _get_serialized_table_text(self, table, serialized_tables_instead_of_markdown):
        """
        将序列化表格格式转换为文本字符串
        
        参数：
            table: 包含序列化数据的表格对象
            serialized_tables_instead_of_markdown: 是否用序列化文本完全替代markdown
            
        返回：
            str: 包含连接信息块或markdown回退的字符串
        """
        if not table.get("serialized"):
            return table.get("markdown", "")

        info_blocks = table["serialized"].get("information_blocks", [])
        text_blocks = [block["information_block"] for block in info_blocks]
        serialized_text = "\n".join(text_blocks)
        if serialized_tables_instead_of_markdown:
            return serialized_text
        else:
            markdown = table.get("markdown", "")
            combined_text = f"{markdown}\nDescription of the table entities:\n{serialized_text}"
            return combined_text

    def export_to_markdown(self, reports, output_dir: Path):
        """
        将处理后的报告导出为markdown文件
        
        参数：
            reports_paths: 已合并的JSON报告文件路径列表
            output_dir: 保存markdown文件的目录
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for _report in reports:
            with open(_report.merge_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            document_text = ""
            for page in report_data['content']['pages']:
                document_text += f"\n\n---\n\n# Page {page['page']}\n\n"
                document_text += page['text']

            file_name = output_dir / f'{_report.report_name.split(".")[0]}.md'
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(document_text)
            temp = FinancialReport.query.filter(FinancialReport.id == _report.id).first()
            if temp:
                temp.markdown_path = file_name
                db.session.commit()
