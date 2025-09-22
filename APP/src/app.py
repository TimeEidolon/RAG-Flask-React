import os
import io
import sys

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from .rag_line.pipeline import Pipeline
from .extensions import db, _yaml, es_service
from .routes import rag_routes

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 加载环境变量
load_dotenv()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def init_db(_app: Flask):
    mysql_config = _yaml.config['mysql']
    # 配置数据库连接
    database_uri = (
        f"mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}@"
        f"{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"
    )

    _app.config['SQLALCHEMY_DATABASE_URI'] = database_uri
    _app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    _app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_size': 10,
        'max_overflow': 20,
        'pool_recycle': 3600,
        'echo': True
    }
    db.init_app(_app)
    print("数据库初始化完成！")


def init_redis(_app: Flask):
    pass


def create_app():
    """
    创建并配置Flask应用
    
    返回：
        Flask: 配置好的Flask应用实例
    """

    _app = Flask(__name__)
    _yaml.load_config()
    # 初始化数据库
    init_db(_app)
    # 初始化 Elasticsearch
    # es_service.init_es()
    # 初始化 Redis
    init_redis(_app)
    # 注册蓝图
    _app.register_blueprint(rag_routes.rag_bp)
    # 跨域配置
    CORS(_app, resources={r"/*": {"origins": "*"}})
    # 创建数据库表
    with _app.app_context():
        try:
            db.create_all()
            Pipeline.do_pipline()
            print("创建数据库表成功！")
        except Exception as e:
            print(f"创建数据库表失败: {e}")

    print("项目初始化完成！")
    return _app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
