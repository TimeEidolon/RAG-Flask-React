import json
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..extensions import db
from sqlalchemy import text
from qwen_agent.tools.base import register_tool, BaseTool


@register_tool('exc_sql')
class ExcSQLTool(BaseTool):

    def __init__(self):
        super().__init__()
        self.description = '对于生成的SQL，进行SQL查询，并自动可视化'
        self.parameters = [
            {
                'name': 'sql_input',
                'type': 'string',
                'description': '生成的SQL语句',
                'required': True
            },
            {
                'name': 'need_visualize',
                'type': 'boolean',
                'description': '是否需要可视化和统计信息，默认True。如果是对比分析等场景可设为False，不进行可视化。',
                'required': False,
                'default': True
            }
        ]

    def call(self, params: str, **kwargs) -> str:

        args = json.loads(params)
        sql_input = args['sql_input']
        engine = db.get_engine()
        try:
            df = pd.read_sql(sql_input, engine)
            # 前5行+后5行拼接展示
            if len(df) > 10:
                md = pd.concat([df.head(5), df.tail(5)]).to_markdown(index=False)
            else:
                md = df.to_markdown(index=False)
            # 只返回表格
            if len(df) == 1:
                return md
            need_visualize = args.get('need_visualize', True)
            if not need_visualize:
                return md
            desc_md = df.describe().to_markdown()
            # 自动创建目录
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'stock_{int(time.time() * 1000)}.png'
            save_path = os.path.join(save_dir, filename)
            # 智能选择可视化方式
            self.generate_smart_chart_png(df, save_path)
            img_path = os.path.join('image_show', filename)
            img_md = f'![图表]({img_path})'
            return f"{md}\n\n{desc_md}\n\n{img_md}"
        except Exception as e:
            return f"SQL执行或可视化出错: {str(e)}"

    @staticmethod
    def generate_smart_chart_png(df_sql, save_path):
        columns = df_sql.columns
        if len(df_sql) == 0 or len(columns) < 2:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, '无可视化数据', ha='center', va='center', fontsize=16)
            plt.axis('off')
            plt.savefig(save_path)
            plt.close()
            return
        x_col = columns[0]
        y_cols = columns[1:]
        x = df_sql[x_col]
        # 如果数据点较多，自动采样10个点
        if len(df_sql) > 20:
            idx = np.linspace(0, len(df_sql) - 1, 10, dtype=int)
            x = x.iloc[idx]
            df_plot = df_sql.iloc[idx]
            chart_type = 'line'
        else:
            df_plot = df_sql
            chart_type = 'bar'
        plt.figure(figsize=(10, 6))
        for y_col in y_cols:
            if chart_type == 'bar':
                plt.bar(df_plot[x_col], df_plot[y_col], label=str(y_col))
            else:
                plt.plot(df_plot[x_col], df_plot[y_col], marker='o', label=str(y_col))
        plt.xlabel(x_col)
        plt.ylabel('数值')
        plt.title('股票数据统计')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


@register_tool('get_ddl')
class DDLExtractorTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.description = '获取指定数据表的DDL语句'
        self.parameters = [
            {
                'name': 'table_name',
                'type': 'string',
                'description': '需要查询的数据表名称',
                'required': True
            }
        ]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        table_name = args['table_name']
        engine = db.get_engine()

        try:
            ddl_query = text(f"SHOW CREATE TABLE {table_name}")
            with engine.connect() as conn:
                result = conn.execute(ddl_query)
                ddl = result.scalar()
                return f"```sql\n{ddl}\n```"

        except Exception as e:
            return f"DDL查询失败: {str(e)}"
