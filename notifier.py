# -*- coding: utf-8 -*-
"""
===================================
Telegram 通知模块（支持环境变量）
===================================
"""

import os
import requests
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Telegram通知器（支持环境变量配置）"""

    def __init__(self, token: str, chat_id: str):
        """
        初始化Telegram通知器

        Args:
            token: Telegram Bot Token (从BotFather获取或环境变量TELEGRAM_BOT_TOKEN)
            chat_id: 接收消息的聊天ID (从环境变量TELEGRAM_CHAT_ID获取)
        """
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        发送消息

        Args:
            message: 消息内容
            parse_mode: 解析模式 (Markdown/HTML/None)

        Returns:
            bool: 发送是否成功
        """
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }

            response = requests.post(url, json=data, timeout=10)
            result = response.json()

            if result.get("ok"):
                logger.info(f"[Telegram] 消息发送成功")
                return True
            else:
                logger.error(f"[Telegram] 发送失败: {result.get('description')}")
                return False

        except Exception as e:
            logger.error(f"[Telegram] 发送异常: {e}")
            return False

    def send_monitoring_report(self, signal: dict, position: dict, data_source: str) -> bool:
        """
        发送详细监控报告

        Args:
            signal: 信号字典
            position: 持仓状态
            data_source: 数据源

        Returns:
            bool: 发送是否成功
        """
        message = self._format_detailed_report(signal, position, data_source)
        return self.send_message(message, parse_mode="Markdown")

    def _format_detailed_report(self, signal: dict, position: dict, data_source: str) -> str:
        """格式化详细监控报告"""

        indicators = signal['indicators']

        # 根据信号类型选择不同的报告格式
        if signal['buy_signal']:
            return self._format_buy_signal(signal, position, data_source)
        elif signal['sell_signal']:
            return self._format_sell_signal(signal, position, data_source)
        else:
            return self._format_monitoring_update(signal, position, data_source)

    def _format_buy_signal(self, signal: dict, position: dict, data_source: str) -> str:
        """格式化买入信号报告"""

        indicators = signal['indicators']
        entry_price = signal['price']
        stop_loss = signal['stop_loss']
        position_size = signal['position_size']

        # 计算止盈目标（基于2:1盈亏比）
        risk = entry_price - stop_loss
        target_1 = entry_price + risk * 1.5  # 第一止盈位（1.5倍风险）
        target_2 = entry_price + risk * 2.0  # 第二止盈位（2倍风险）
        target_3 = entry_price + risk * 3.0  # 第三止盈位（3倍风险）

        # 风险收益比
        risk_pct = (risk / entry_price) * 100
        reward_1_pct = ((target_1 - entry_price) / entry_price) * 100

        # 技术分析
        trend_strength = self._get_trend_strength(signal)
        signal_quality = self._get_signal_quality(signal)

        # 提取数据时间用于显示
        data_time = signal['datetime']

        message = f"""🟢 *买入信号 - {signal['signal_type'].upper()}*

━━━━━━━━━━━━━━━━━━━━

*📊 市场状态*
• 数据时间: `{data_time}`
• 当前价格: `{entry_price:.0f}`
• 趋势: `{signal['trend'].upper()}` ({signal['strength']})
• 波动率: `{indicators['volatility']*100:.2f}%`

*📈 技术指标*
• Ratio: `{indicators['ratio']:.3f}` (上一根: `{indicators['ratio_prev']:.3f}`)
• RSI: `{indicators['rsi']:.1f}` ({'超买' if indicators['rsi'] > 70 else '超卖' if indicators['rsi'] < 30 else '中性'})
• STC: `{indicators['stc']:.1f}` (上一根: `{indicators['stc_prev']:.1f}`)
• EMA_Fast: `{indicators['ema_fast']:.0f}`
• EMA_Slow: `{indicators['ema_slow']:.0f}`

━━━━━━━━━━━━━━━━━━━━

*💰 交易计划*

*📍 开仓信息*
• 入场价格: `{entry_price:.0f}`
• 建议仓位: `{position_size:.1f}x`
• 信号类型: `{signal['signal_type']}` ({'狙击点' if signal['signal_type'] == 'sniper' else '追涨'})

*🛡️ 风险控制*
• 止损价格: `{stop_loss:.0f}` (`{risk_pct:.2f}%`)
• 止损金额: `{risk * position_size:.0f}` 点/手

*🎯 止盈目标*
• 第一目标: `{target_1:.0f}` (`{reward_1_pct:.2f}%`) ← 建议50%仓位
• 第二目标: `{target_2:.0f}` (`{((target_2-entry_price)/entry_price)*100:.2f}%`) ← 建议30%仓位
• 第三目标: `{target_3:.0f}` (`{((target_3-entry_price)/entry_price)*100:.2f}%`) ← 剩余20%

*📊 风险收益比*
• 风险: `{risk:.0f}` 点 (`{risk_pct:.2f}%`)
• 第一目标收益: `{target_1 - entry_price:.0f}` 点 (`{reward_1_pct:.2f}%`)
• 盈亏比: `1:{(target_1 - entry_price) / risk:.1f}`

━━━━━━━━━━━━━━━━━━━━

*🔍 技术分析*
• 趋势强度: `{trend_strength}`
• 信号质量: `{signal_quality}`
• {self._get_trading_advice(signal)}

━━━━━━━━━━━━━━━━━━━━

_数据源: {data_source}_ | _生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_

---
🤖 *沪铜策略实盘监控*
⚠️ *风险提示：仅供参考，实际交易请结合市场情况*
"""

        return message

    def _format_sell_signal(self, signal: dict, position: dict, data_source: str) -> str:
        """格式化卖出信号报告"""

        entry_price = position['entry_price']
        exit_price = signal['price']
        position_size = position['position_size']

        # 计算盈亏
        pnl_points = (exit_price - entry_price) * position_size
        pnl_pct = (exit_price - entry_price) / entry_price * 100

        # 持仓天数
        entry_dt = datetime.fromisoformat(position['entry_datetime'])
        signal_dt = datetime.fromisoformat(signal['datetime'])
        days_held = (signal_dt - entry_dt).days

        # 盈亏状态
        if pnl_points > 0:
            pnl_emoji = "✅"
            pnl_status = "盈利"
        else:
            pnl_emoji = "❌"
            pnl_status = "亏损"

        # 提取数据时间用于显示
        data_time = signal['datetime']

        message = f"""🔴 *卖出信号 - {signal['signal_type'].upper()}*

━━━━━━━━━━━━━━━━━━━━

*💼 平仓信息*

*📍 交易结果*
• 入场价格: `{entry_price:.0f}`
• 出场价格: `{exit_price:.0f}`
• 仓位大小: `{position_size:.1f}x`
• 持仓天数: `{days_held}` 天

*💰 盈亏结算*
{pnl_emoji} • 盈亏: `{pnl_points:.0f}` 点
{pnl_emoji} • 盈亏率: `{pnl_pct:+.2f}%`
{'✅' if pnl_points > 0 else '❌'} • 状态: `{pnl_status}`

━━━━━━━━━━━━━━━━━━━━

*📊 当前市场状态*
• 数据时间: `{data_time}`
• 当前价格: `{signal['price']:.0f}`
• 趋势: `{signal['trend'].upper()}` ({signal['strength']})
• Ratio: `{signal['indicators']['ratio']:.3f}`
• RSI: `{signal['indicators']['rsi']:.1f}`
• STC: `{signal['indicators']['stc']:.1f}`

*🔔 卖出原因*
• 信号类型: `{signal['signal_type']}`
• {self._get_exit_reason(signal)}

━━━━━━━━━━━━━━━━━━━━

_数据源: {data_source}_ | _生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_

---
🤖 *沪铜策略实盘监控*
"""

        return message

    def _format_monitoring_update(self, signal: dict, position: dict, data_source: str) -> str:
        """格式化监控更新报告（无信号）"""

        indicators = signal['indicators']

        # 提取数据时间
        data_time = signal['datetime']
        if isinstance(data_time, str):
            # 从字符串中提取时间部分 (例如 "2026-02-04 20:00:00" -> "20:00")
            time_part = data_time.split()[-1][:5] if ' ' in data_time else data_time[:5]
        else:
            time_part = data_time.strftime('%H:%M')

        message = f"""⚪ *市场监控更新*

━━━━━━━━━━━━━━━━━━━━

*📊 当前市场状态*
• 数据时间: `{data_time}`
• 当前价格: `{signal['price']:.0f}`
• 趋势: `{signal['trend'].upper()}` ({signal['strength']})
• 波动率: `{indicators['volatility']*100:.2f}%`

*📈 技术指标*
• Ratio: `{indicators['ratio']:.3f}` (上一根: `{indicators['ratio_prev']:.3f}`)
  └─ {'📈 上升' if indicators['ratio'] > indicators['ratio_prev'] else '📉 下降'}
• RSI: `{indicators['rsi']:.1f}` ({'超买' if indicators['rsi'] > 70 else '超卖' if indicators['rsi'] < 30 else '中性'})
• STC: `{indicators['stc']:.1f}` (上一根: `{indicators['stc_prev']:.1f}`)
  └─ {'📈 上升' if indicators['stc'] > indicators['stc_prev'] else '📉 下降'}
• EMA_Fast: `{indicators['ema_fast']:.0f}`
• EMA_Slow: `{indicators['ema_slow']:.0f}`
  └─ {'金叉 🟢' if indicators['ema_fast'] > indicators['ema_slow'] else '死叉 🔴'}

━━━━━━━━━━━━━━━━━━━━

"""

        # 添加持仓信息（如果有）
        if position['holding']:
            pnl = (signal['price'] - position['entry_price']) * position['position_size']
            pnl_pct = (signal['price'] - position['entry_price']) / position['entry_price'] * 100

            entry_dt = datetime.fromisoformat(position['entry_datetime'])
            days_held = (datetime.now() - entry_dt).days

            message += f"""*💼 当前持仓*
• 入场价: `{position['entry_price']:.0f}`
• 当前价: `{signal['price']:.0f}`
• 仓位: `{position['position_size']:.1f}x`
• 持仓天数: `{days_held}` 天
• 止损价: `{position['stop_loss']:.0f}`
• 浮动盈亏: `{pnl:+.0f}` 点 (`{pnl_pct:+.2f}%`)
{'✅ 盈利' if pnl > 0 else '❌ 亏损' if pnl < 0 else '⚪ 平衡'}

━━━━━━━━━━━━━━━━━━━━

"""
        else:
            message += """*💼 当前持仓*: 空仓

━━━━━━━━━━━━━━━━━━━━

"""

        # 添加市场分析
        message += f"""*🔍 市场分析*
• 趋势强度: `{self._get_trend_strength(signal)}`
• 信号状态: `{self._get_signal_status(signal)}`
• 操作建议: `{self._get_action_advice(signal)}`

━━━━━━━━━━━━━━━━━━━━

_数据源: {data_source}_ | _生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_

---
🤖 *沪铜策略实盘监控*
"""

        return message

    def _get_trend_strength(self, signal: dict) -> str:
        """获取趋势强度描述"""
        indicators = signal['indicators']
        ratio = indicators['ratio']

        if ratio > 2.0:
            return "极强 🚀"
        elif ratio > 1.5:
            return "强势 📈"
        elif ratio > 1.0:
            return "中等 ➡️"
        elif ratio > 0.5:
            return "弱势 📉"
        else:
            return "极弱 ⚠️"

    def _get_signal_quality(self, signal: dict) -> str:
        """获取信号质量描述"""
        indicators = signal['indicators']
        rsi = indicators['rsi']
        ratio = indicators['ratio']

        if signal['signal_type'] == 'sniper':
            # Sniper信号：Ratio收缩+RSI强
            if ratio < 0.5 and rsi > 50:
                return "优秀 ⭐⭐⭐"
            elif ratio < 0.8 and rsi > 45:
                return "良好 ⭐⭐"
            else:
                return "一般 ⭐"
        else:
            # Chase信号：EMA金叉
            if rsi > 55:
                return "良好 ⭐⭐"
            else:
                return "一般 ⭐"

    def _get_exit_reason(self, signal: dict) -> str:
        """获取离场原因"""
        reason = signal['reason'].get('sell', '')

        if reason == 'stc':
            return "STC指标从高位回落，获利了结"
        elif reason == 'trend':
            return "趋势反转，EMA死叉"
        elif reason == 'stop_loss':
            return "触发止损"
        else:
            return f"其他原因: {reason}"

    def _get_signal_status(self, signal: dict) -> str:
        """获取信号状态"""
        indicators = signal['indicators']

        checks = []

        # 趋势检查
        if signal['trend'] == 'up':
            checks.append("趋势向上 🟢")
        else:
            checks.append("趋势向下 🔴")

        # Ratio检查
        if 0 < indicators['ratio'] < 1.15:
            checks.append("Ratio安全 ✅")
        else:
            checks.append("Ratio风险 ⚠️")

        # RSI检查
        if indicators['rsi'] > 45:
            checks.append("RSI强势 ✅")
        else:
            checks.append("RSI弱势 ⚠️")

        return " | ".join(checks)

    def _get_action_advice(self, signal: dict) -> str:
        """获取操作建议"""
        indicators = signal['indicators']

        if signal['trend'] == 'up' and 0 < indicators['ratio'] < 1.0:
            return "等待Ratio回缩后的买入机会"
        elif signal['trend'] == 'up' and indicators['ratio'] >= 1.0:
            return "趋势向上但Ratio偏高，观望"
        elif signal['trend'] == 'down':
            return "趋势向下，等待入场机会"
        else:
            return "市场整理中，继续观望"

    def _get_trading_advice(self, signal: dict) -> str:
        """获取交易建议"""
        if signal['signal_type'] == 'sniper':
            return "狙击点入场：Ratio回缩+趋势向上+RSI强势，可靠性较高"
        elif signal['signal_type'] == 'chase':
            return "追涨入场：EMA金叉+趋势确认，注意控制仓位"
        else:
            return "观望为主，等待更好的入场点"


def load_config(config_path: str = None) -> Optional[Dict[str, str]]:
    """
    加载Telegram配置（环境变量优先）

    优先级：
    1. 环境变量 (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    2. 配置文件

    Args:
        config_path: 配置文件路径（可选）

    Returns:
        配置字典，失败返回None
    """
    import json

    # 优先从环境变量读取
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    # 调试：输出环境变量状态
    logger.info(f"[调试] TELEGRAM_BOT_TOKEN 存在: {bool(token)}")
    logger.info(f"[调试] TELEGRAM_CHAT_ID 存在: {bool(chat_id)}")
    if token:
        logger.info(f"[调试] BOT_TOKEN 长度: {len(token)}")
    if chat_id:
        logger.info(f"[调试] CHAT_ID 值: {chat_id}")

    if token and chat_id:
        logger.info("[配置] 使用环境变量配置")
        return {'token': token, 'chat_id': chat_id}

    # 环境变量未配置，尝试从文件读取
    if config_path is None:
        # 根据脚本位置自动推断配置文件路径
        script_dir = Path(__file__).parent
        config_path = script_dir / 'config' / 'telegram.json'

    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"[配置] Telegram配置文件不存在: {config_path}")
        logger.warning(f"[配置] 请设置环境变量或创建配置文件:")
        logger.warning(f"  环境变量: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
        logger.warning(f"  配置文件: {config_path}")
        logger.warning(f'  格式: {{"token": "YOUR_BOT_TOKEN", "chat_id": "YOUR_CHAT_ID"}}')
        return None

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if 'token' not in config or 'chat_id' not in config:
            logger.error(f"[配置] 配置文件缺少必要字段 (token/chat_id)")
            return None

        logger.info(f"[配置] 使用配置文件: {config_path}")
        return config

    except Exception as e:
        logger.error(f"[配置] 加载配置失败: {e}")
        return None


def get_notifier() -> Optional[TelegramNotifier]:
    """
    获取Telegram通知器实例

    支持环境变量和配置文件两种方式

    Returns:
        TelegramNotifier实例，配置失败返回None
    """
    config = load_config()

    if config is None:
        return None

    return TelegramNotifier(token=config['token'], chat_id=config['chat_id'])


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    # 测试配置加载
    config = load_config()

    if config:
        # 隐藏Token中间部分
        token_masked = f"{config['token'][:10]}...{config['token'][-4:]}"
        print(f"[配置] Bot Token: {token_masked}")
        print(f"[配置] Chat ID: {config['chat_id']}")

        # 测试发送
        notifier = TelegramNotifier(token=config['token'], chat_id=config['chat_id'])

        # 发送测试消息
        test_message = """🧪 *测试消息*

Telegram通知配置成功！

━━━━━━━━━━━━━━━━━━━━

*✅ 配置状态*
• Bot Token: 已配置
• Chat ID: 已配置
• 连接状态: 正常

━━━━━━━━━━━━━━━━━━━━

*📝 下一步*
1. 设置定时任务: `setup_task.bat`
2. 或手动运行: `python copper_monitor.py`

---
🤖 *沪铜策略实盘监控*
"""

        success = notifier.send_message(test_message)

        if success:
            print("\n[成功] 测试消息已发送！")
            print("请检查Telegram是否收到消息")
        else:
            print("\n[失败] 消息发送失败，请检查配置")
    else:
        print("[配置] 配置加载失败")
        print("\n请通过以下方式配置：")
        print("1. 环境变量（推荐）:")
        print("   set TELEGRAM_BOT_TOKEN=your_token")
        print("   set TELEGRAM_CHAT_ID=your_chat_id")
        print("\n2. 配置文件:")
        print("   编辑 config/telegram.json")
