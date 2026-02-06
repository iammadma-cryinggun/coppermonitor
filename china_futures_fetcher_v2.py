# -*- coding: utf-8 -*-
"""
中国期货数据获取器 - 改进版
使用更合理的4小时K线分割，匹配期货实际交易时间
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ChinaFuturesFetcherV2:
    """
    中国期货数据获取器 - 改进版

    4小时K线定义（匹配实际交易时间）：
    - 02:00 - 代表夜盘（21:00-01:00）
    - 12:00 - 代表早盘（09:00-11:30）
    - 16:00 - 代表午盘（13:30-15:00）
    - 22:00 - 代表夜盘开始（21:00的后续）
    """

    # 期货品种映射
    FUTURES_MAPPING = {
        'NI': {'name': '沪镍', 'sina_code': 'NI0'},
        'SA': {'name': '纯碱', 'sina_code': 'SA0'},
        'V': {'name': 'PVC', 'sina_code': 'V0'},
        'CU': {'name': '沪铜', 'sina_code': 'CU0'},
        'SN': {'name': '沪锡', 'sina_code': 'SN0'},
        'PB': {'name': '沪铅', 'sina_code': 'PB0'},
        'FG': {'name': '玻璃', 'sina_code': 'FG0'},
    }

    def __init__(self):
        """初始化数据获取器"""
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = timedelta(minutes=5)

    def get_supported_symbols(self) -> Dict[str, str]:
        """获取支持的期货品种列表"""
        return {code: info['name'] for code, info in self.FUTURES_MAPPING.items()}

    def get_futures_info(self, code: str) -> Optional[Dict]:
        """获取期货品种信息"""
        return self.FUTURES_MAPPING.get(code.upper())

    def get_historical_data(self, code: str, days: int = 252) -> Optional[pd.DataFrame]:
        """
        获取期货历史行情数据（4小时K线）

        改进版：根据实际交易时间重采样
        """
        code = code.upper()
        info = self.get_futures_info(code)

        if not info:
            logger.error(f"[ChinaFutures] 不支持的期货品种: {code}")
            return None

        # 检查缓存
        cache_key = f"{code}_4h_v2"
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                logger.debug(f"[ChinaFutures] 使用缓存数据: {code} (4小时K线 v2)")
                return cached_data

        try:
            import akshare as ak

            logger.info(f"[ChinaFutures] 获取 {info['name']} ({code}) 4小时K线数据...")

            sina_code = info['sina_code']

            # 获取60分钟数据
            df_60m = ak.futures_zh_minute_sina(symbol=sina_code, period='60')

            if df_60m is None or df_60m.empty:
                logger.warning(f"[ChinaFutures] {code} 60分钟数据为空")
                return None

            # 转换datetime列
            df_60m['datetime'] = pd.to_datetime(df_60m['datetime'])
            df_60m = df_60m.set_index('datetime')

            # 改进的重采样逻辑：根据实际交易时间
            df_4h = self._resample_4h_trading_hours(df_60m)

            if df_4h is None or df_4h.empty:
                logger.warning(f"[ChinaFutures] {code} 4小时重采样失败")
                return None

            # 标准化列名
            df_4h = self._standardize_columns(df_4h)

            # 缓存数据
            self._cache[cache_key] = (df_4h, datetime.now())

            logger.info(f"[ChinaFutures] 获取成功 {code}: {len(df_4h)} 条4小时K线记录")
            logger.info(f"[ChinaFutures] 数据范围: {df_4h['datetime'].iloc[0]} ~ {df_4h['datetime'].iloc[-1]}")
            return df_4h

        except Exception as e:
            logger.error(f"[ChinaFutures] 获取 {code} 数据失败: {e}")
            return None

    def _resample_4h_trading_hours(self, df_60m: pd.DataFrame) -> pd.DataFrame:
        """
        根据期货实际交易时间重采样为4小时K线

        时间段定义：
        - 02:00 - 夜盘（21:00-01:00）
        - 12:00 - 早盘（09:00-11:30）
        - 16:00 - 午盘（13:30-15:00）
        - 22:00 - 夜盘开始（21:00后续）
        """
        # 创建时间标签
        df_60m['time_label'] = None

        for idx, row in df_60m.iterrows():
            hour = row['hour'] if 'hour' in row else idx.hour
            minute = row['minute'] if 'minute' in row else idx.minute

            # 夜盘（21:00-01:00）-> 标记为 02:00
            if hour >= 21 or hour < 1:
                df_60m.loc[idx, 'time_label'] = idx.replace(hour=2, minute=0, second=0)
            # 早盘（09:00-11:30）-> 标记为 12:00
            elif 9 <= hour < 12:
                df_60m.loc[idx, 'time_label'] = idx.replace(hour=12, minute=0, second=0)
            # 午盘（13:30-15:00）-> 标记为 16:00
            elif 13 <= hour < 15:
                df_60m.loc[idx, 'time_label'] = idx.replace(hour=16, minute=0, second=0)
            # 其他时间段标记为 22:00
            else:
                df_60m.loc[idx, 'time_label'] = idx.replace(hour=22, minute=0, second=0)

        # 按time_label分组聚合
        df_4h = df_60m.groupby('time_label').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'hold': 'last'
        }).dropna()

        # 重置索引
        df_4h = df_4h.reset_index()
        df_4h = df_4h.rename(columns={'time_label': 'datetime'})

        # 排序
        df_4h = df_4h.sort_values('datetime').reset_index(drop=True)

        return df_4h

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化列名
        """
        column_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'hold': 'hold',
            'datetime': 'datetime'
        }

        df = df.rename(columns=column_mapping)

        # 确保必要的列都存在
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"[ChinaFutures] 缺少列: {col}")
                if col == 'volume':
                    df[col] = 0
                elif col in ['open', 'high', 'low', 'close']:
                    df[col] = df['close']  # 如果没有OHLC，用收盘价填充

        # 只保留需要的列
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

        return df
