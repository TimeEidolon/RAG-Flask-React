import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from ..extensions import db
from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('boll_detection')
class BollDetectionTool(BaseTool):
    description = '对指定股票(ts_code)的收盘价进行布林带异常点检测，默认检测过去1年，也可自定义时间范围，返回超买和超卖日期及布林带图。'
    parameters = [
        {
            'name': 'ts_code',
            'type': 'string',
            'description': '股票代码，必填',
            'required': True
        },
        {
            'name': 'sql',
            'type': 'string',
            'description': '查询语句，必填',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        ts_code = args['ts_code']
        sql = args['sql']
        # 获取数据
        engine = db.get_engine()
        df = pd.read_sql(sql, engine)
        if len(df) < 21:
            return '历史数据不足，无法进行布林带检测。'
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        # 计算布林带
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['STD20'] = df['close'].rolling(window=20).std()
        df['UPPER'] = df['MA20'] + 2 * df['STD20']
        df['LOWER'] = df['MA20'] - 2 * df['STD20']
        # 检测超买/超卖
        overbought = df[df['close'] > df['UPPER']][['trade_date', 'close']]
        oversold = df[df['close'] < df['LOWER']][['trade_date', 'close']]
        # 结果表格
        result_md = f"### 超买日期\n{overbought.to_markdown(index=False)}\n\n### 超卖日期\n{oversold.to_markdown(index=False)}"
        # 绘制布林带图
        save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
        os.makedirs(save_dir, exist_ok=True)
        filename = f'boll_{ts_code}_{int(time.time() * 1000)}.png'
        save_path = os.path.join(save_dir, filename)
        plt.figure(figsize=(12, 6))
        plt.plot(df['trade_date'], df['close'], label='收盘价')
        plt.plot(df['trade_date'], df['MA20'], label='MA20')
        plt.plot(df['trade_date'], df['UPPER'], label='上轨+2σ')
        plt.plot(df['trade_date'], df['LOWER'], label='下轨-2σ')
        plt.fill_between(df['trade_date'], df['UPPER'], df['LOWER'], color='gray', alpha=0.1)
        plt.scatter(overbought['trade_date'], overbought['close'], color='red', label='超买', zorder=5)
        plt.scatter(oversold['trade_date'], oversold['close'], color='blue', label='超卖', zorder=5)
        # 横坐标稀疏显示
        total_len = len(df)
        if total_len > 12:
            step = max(1, total_len // 10)
            show_idx = list(range(0, total_len, step))
            show_labels = [df['trade_date'].iloc[i] for i in show_idx]
            plt.xticks(show_idx, show_labels, rotation=45)
        else:
            plt.xticks(rotation=45)
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.title(f'{ts_code} 布林带异常点检测')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        img_path = os.path.join('image', filename)
        img_md = f'![布林带检测]({img_path})'
        return f"{result_md}\n\n{img_md}"
