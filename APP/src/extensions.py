import yaml
from elasticsearch import Elasticsearch
from flask_sqlalchemy import SQLAlchemy
from pyprojroot import here

db = SQLAlchemy()


class YamlConfig:
    _instance = None

    def __init__(self):
        self.config = {}
        self.load_config()

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    def load_config(self):
        yaml_path = here() / "APP" / "APP.yaml"
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"配置加载失败: {str(e)}")


class ESService:
    def __init__(self):
        self.es = None
        self.init_es()

    def init_es(self):
        es_config = _yaml.config['es']
        if es_config['switch'] is True:
            self.es = Elasticsearch(
                hosts=[f"http://{es_config['host']}:{es_config['port']}"]
            )
            if not self.es.ping():
                raise ValueError("无法连接到 Elasticsearch！")
            print("Elasticsearch初始化完成！")


# 初始化单例实例
_yaml = YamlConfig.get_instance()
es_service = ESService()
