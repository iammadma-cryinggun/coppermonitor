# -*- coding: utf-8 -*-
"""
===================================
国内商品期货数据源 - China Futures Fetcher (简化版)
===================================

支持的交易所：
1. 上期所 (SHFE): 黄金、白银、铜、铝、锌、铅、镍、锡、螺纹钢、热轧卷板
2. 大商所 (DCE): 铁矿石、豆粕、豆油、玉米、棕榈油、聚乙烯
3. 能源中心 (INE): 原油
4. 郑商所 (CZCE): 甲醇、PTA、白糖、棉花、玻璃

数据来源：akshare 库（免费，无需 Token）
主力合约代码格式：
- 原油: V0
- 黄金: AU0
- 白银: AG0
- 铜: CU0
- 铁矿石: I0
等等
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ChinaFuturesFetcher:
    """
    国内商品期货数据获取器（简化版）
    """

    # 国内期货主力合约代码映射
    # 格式: 品种代码 -> {名称, sina_code, 交易所, 单位}
    FUTURES_MAPPING = {
        # === 能源中心 (INE) ===
        'SC': {'name': '原油', 'sina_code': 'V0', 'exchange': 'INE', 'unit': '元/桶'},

        # === 上期所 (SHFE) ===
        'AU': {'name': '黄金', 'sina_code': 'AU0', 'exchange': 'SHFE', 'unit': '元/克'},
        'AG': {'name': '白银', 'sina_code': 'AG0', 'exchange': 'SHFE', 'unit': '元/千克'},
        'CU': {'name': '铜', 'sina_code': 'CU0', 'exchange': 'SHFE', 'unit': '元/吨'},
        'AL': {'name': '铝', 'sina_code': 'AL0', 'exchange': 'SHFE', 'unit': '元/吨'},
        'ZN': {'name': '锌', 'sina_code': 'ZN0', 'exchange': 'SHFE', 'unit': '元/吨'},
        'PB': {'name': '铅', 'sina_code': 'PB0', 'exchange': 'SHFE', 'unit': '元/吨'},
        'NI': {'name': '镍', 'sina_code': 'NI0', 'exchange': 'SHFE', 'unit': '元/吨'},
        'SN': {'name': '锡', 'sina_code': 'SN0', 'exchange': 'SHFE', 'unit': '元/吨'},
        'RB': {'name': '螺纹钢', 'sina_code': 'RB0', 'exchange': 'SHFE', 'unit': '元/吨'},
        'HC': {'name': '热轧卷板', 'sina_code': 'HC0', 'exchange': 'SHFE', 'unit': '元/吨'},

        # === 大商所 (DCE) ===
        'I': {'name': '铁矿石', 'sina_code': 'I0', 'exchange': 'DCE', 'unit': '元/吨'},
        'M': {'name': '豆粕', 'sina_code': 'M0', 'exchange': 'DCE', 'unit': '元/吨'},
        'Y': {'name': '豆油', 'sina_code': 'Y0', 'exchange': 'DCE', 'unit': '元/吨'},
        'C': {'name': '玉米', 'sina_code': 'C0', 'exchange': 'DCE', 'unit': '元/吨'},
        'P': {'name': '棕榈油', 'sina_code': 'P0', 'exchange': 'DCE', 'unit': '元/吨'},
        'L': {'name': '聚乙烯', 'sina_code': 'L0', 'exchange': 'DCE', 'unit': '元/吨'},
        'V': {'name': '聚氯乙烯', 'sina_code': 'V0', 'exchange': 'DCE', 'unit': '元/吨'},
        'PP': {'name': '聚丙烯', 'sina_code': 'PP0', 'exchange': 'DCE', 'unit': '元/吨'},
        'JD': {'name': '鸡蛋', 'sina_code': 'JD0', 'exchange': 'DCE', 'unit': '元/500千克'},
        'A': {'name': '豆一', 'sina_code': 'A0', 'exchange': 'DCE', 'unit': '元/吨'},
        'B': {'name': '豆二', 'sina_code': 'B0', 'exchange': 'DCE', 'unit': '元/吨'},

        # === 郑商所 (CZCE) ===
        'MA': {'name': '甲醇', 'sina_code': 'MA0', 'exchange': 'CZCE', 'unit': '元/吨'},
        'TA': {'name': 'PTA', 'sina_code': 'TA0', 'exchange': 'CZCE', 'unit': '元/吨'},
        'SR': {'name': '白糖', 'sina_code': 'SR0', 'exchange': 'CZCE', 'unit': '元/吨'},
        'CF': {'name': '棉花', 'sina_code': 'CF0', 'exchange': 'CZCE', 'unit': '元/吨'},
        'FG': {'name': '玻璃', 'sina_code': 'FG0', 'exchange': 'CZCE', 'unit': '元/吨'},
        'SA': {'name': '纯碱', 'sina_code': 'SA0', 'exchange': 'CZCE', 'unit': '元/吨'},
        'UR': {'name': '尿素', 'sina_code': 'UR0', 'exchange': 'CZCE', 'unit': '元/吨'},
    }

    def __init__(self):
        """初始化数据获取器"""
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = timedelta(minutes=15)

    def get_supported_symbols(self) -> Dict[str, str]:
        """获取支持的期货品种列表"""
        return {code: info['name'] for code, info in self.FUTURES_MAPPING.items()}

    def get_futures_info(self, code: str) -> Optional[Dict]:
        """获取期货品种信息"""
        return self.FUTURES_MAPPING.get(code.upper())

    def get_historical_data(self, code: str, days: int = 252) -> Optional[pd.DataFrame]:
        """获取期货历史行情数据"""
        code = code.upper()
        info = self.get_futures_info(code)

        if not info:
            logger.error(f"[ChinaFutures] 不支持的期货品种: {code}")
            return None

        # 检查缓存
        cache_key = f"{code}_{days}"
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                logger.debug(f"[ChinaFutures] 使用缓存数据: {code}")
                return cached_data

        try:
            import akshare as ak

            logger.info(f"[ChinaFutures] 获取 {info['name']} ({code}) 历史数据...")

            sina_code = info['sina_code']
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

            # 获取历史数据
            df = ak.futures_main_sina(
                symbol=sina_code,
                start_date=start_date,
                end_date=end_date
            )

            if df is None or df.empty:
                logger.warning(f"[ChinaFutures] {code} 数据为空")
                return None

            # 标准化列名
            df = self._standardize_columns(df)

            # 缓存数据
            self._cache[cache_key] = (df, datetime.now())

            logger.info(f"[ChinaFutures] 获取成功 {code}: {len(df)} 条记录")
            return df

        except Exception as e:
            logger.error(f"[ChinaFutures] 获取 {code} 数据失败: {e}")
            return None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        column_mapping = {
            '日期': 'date',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '收盘价': 'close',
            '成交量': 'volume',
            '持仓量': 'open_interest',
            '动态结算价': 'settlement',
            '涨跌': 'change',
            '涨跌幅': 'pct_change',
        }

        df = df.rename(columns=column_mapping)
        return df

    def calculate_historical_volatility(self, code: str, window: int = 20) -> Optional[float]:
        """计算历史波动率 (HV)"""
        df = self.get_historical_data(code, days=max(window * 2, 60))

        if df is None or len(df) < window:
            logger.warning(f"[ChinaFutures] 数据不足，无法计算 {code} 的 HV")
            return None

        try:
            df = df.sort_values('date')
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))

            log_returns = df['log_return'].dropna()
            hv = log_returns.tail(window).std()

            # 年化 (252 个交易日)
            hv = hv * np.sqrt(252)

            return float(hv * 100)

        except Exception as e:
            logger.error(f"[ChinaFutures] 计算 {code} HV 失败: {e}")
            return None

    def estimate_implied_volatility(self, code: str) -> Optional[float]:
        """
        估算隐含波动率 (IV)

        使用 HV + 风险溢价的方法估算 IV
        """
        hv = self.calculate_historical_volatility(code, window=20)

        if hv is None:
            return None

        # 使用 HV + 风险溢价估算 IV
        # 典型溢价：5-15%
        premium = 0.10  # 10% 的风险溢价
        iv = hv * (1 + premium)

        return float(iv)

    def get_current_price(self, code: str) -> Optional[float]:
        """获取当前期货价格（最新收盘价）"""
        df = self.get_historical_data(code, days=5)

        if df is None or df.empty:
            return None

        latest = df.iloc[-1]
        return float(latest['close'])


# 便捷函数
_china_futures_fetcher: Optional[ChinaFuturesFetcher] = None


def get_china_futures_fetcher() -> ChinaFuturesFetcher:
    """获取国内期货数据获取器单例"""
    global _china_futures_fetcher
    if _china_futures_fetcher is None:
        _china_futures_fetcher = ChinaFuturesFetcher()
    return _china_futures_fetcher


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )

    fetcher = get_china_futures_fetcher()

    print("=" * 60)
    print("国内商品期货数据源测试")
    print("=" * 60)

    # 测试数据获取
    test_codes = ['SC', 'AU', 'AG', 'CU', 'I']  # 原油、黄金、白银、铜、铁矿石

    print("\n[测试] 获取期货数据:")
    for code in test_codes:
        info = fetcher.get_futures_info(code)
        if info:
            print(f"\n  {code} - {info['name']} ({info['exchange']})")

            # 获取当前价格
            price = fetcher.get_current_price(code)
            print(f"    当前价格: {price:.2f} {info['unit']}" if price else "    当前价格: N/A")

            # 计算 HV
            hv = fetcher.calculate_historical_volatility(code)
            print(f"    历史波动率 (20日): {hv:.2f}%" if hv else "    历史波动率: N/A")

            # 估算 IV
            iv = fetcher.estimate_implied_volatility(code)
            print(f"    隐含波动率 (估算): {iv:.2f}%" if iv else "    隐含波动率: N/A")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
