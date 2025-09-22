import os
import json
import uuid

from ..extensions import db, _yaml
from APP.src.domain.entity import FinancialReport
from flask import Blueprint, jsonify, logging, request

financial_report_bp = Blueprint('financial_report', __name__, url_prefix='/api/financial_reports')


@financial_report_bp.route('/get_reports', methods=['POST'])
def get_reports():
    query = request.json
    FinancialReport.query.filter_by(query).all()
    return jsonify({})


@financial_report_bp.route('/add_report', methods=['POST'])
def add_report():
    # 接收混合数据
    form_data = request.form
    files = request.files

    # 解析结构化数据
    try:
        report_data = json.loads(form_data.get('data'))
        financial_report = FinancialReport(
            id=str(uuid.uuid4()).replace('-', ''),
            company_name=report_data['company_name'],
            english_name=report_data['english_name'],
            stock_code=report_data['stock_code'],
            report_name=report_data['report_name'],
            publish_time=report_data['publish_time'],
            vector_flag=bool(0)
        )
    except Exception as e:
        return jsonify({'error': '数据解析失败', 'detail': str(e)}), 400

    # 处理文件上传
    try:
        if 'report_file' in files:
            pdf_file = files['report_file']
            if not pdf_file.filename.lower().endswith('.pdf'):
                return jsonify({'error': '仅支持PDF文件'}), 400

            # 创建存储目录
            raw_file_dir = _yaml['rag_path']['raw_file']['dir_name']
            save_path = os.path.join(_yaml['rag_path']['root_path'], raw_file_dir)
            os.makedirs(save_path, exist_ok=True)

            # 保存文件并更新路径
            pdf_path = os.path.join(save_path, financial_report.report_name)
            pdf_file.save(pdf_path)
            financial_report.file_path = pdf_path
    except Exception as e:
        return jsonify({'error': '文件上传失败', 'detail': str(e)}), 400

    # 数据库操作
    try:
        db.session.add(financial_report)
        db.session.commit()
        return jsonify({
            'id': financial_report.id,
            'paths': {
                'original': financial_report.file_path,
                'processed': financial_report.json_path
            }
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': '数据库操作失败', 'detail': str(e)}), 500
