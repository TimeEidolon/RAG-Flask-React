from ..extensions import db
import uuid


class FinancialReport(db.Model):
    __tablename__ = 'financial_report'

    # 字段定义
    id = db.Column(db.String(50), primary_key=True, comment='主键ID', default=str(uuid.uuid4()).replace('-', ''))
    stock_code = db.Column(db.String(255), nullable=True, comment='股票编码')
    company_name = db.Column(db.String(50), nullable=True, comment='公司名称')
    english_name = db.Column(db.String(50), nullable=True, comment='英文名称')
    report_name = db.Column(db.String(255), nullable=True, comment='财报名称')
    publish_time = db.Column(db.DateTime, nullable=True, comment='发布时间')
    file_path = db.Column(db.String(255), nullable=True, comment='原文件位置')
    json_path = db.Column(db.String(255), nullable=True, comment="json文件路径")
    merge_path = db.Column(db.String(255), nullable=True, comment="merge文件路径")
    markdown_path = db.Column(db.String(255), nullable=True, comment="markdown文件路径")
    chunked_path = db.Column(db.String(255), nullable=True, comment="chunked文件路径")
    es_flag = db.Column(db.Boolean, nullable=True, comment='是否是完成ES存储')
    vector_flag = db.Column(db.Boolean, nullable=True, comment='是否是完成向量化存储')
