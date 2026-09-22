#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCI 铜期货 纸面交易系统
━━━━━━━━━━━━━━━━━━━━━━━
每日收盘后运行:
  1. 下载最新日线数据 (akshare)
  2. 计算 CCI + STC 指标
  3. 检查入场/出场信号
  4. 管理纸面持仓 (模拟下单)
  5. 记录交易到 CSV
  6. 保存状态到 JSON

监控品种: 玻璃/白银/铜/铝/锡 (OOS验证通过)
仓位: 1手起 (最小交易单位), 30%保证金上限
入场: 信号日收盘计算 → 次日开盘价成交
退出: 4%止损 / 20%止盈 / CCI超买 / CCI死叉
"""
import pandas as pd
import numpy as np
import os, sys, json, time
from datetime import datetime, timedelta

# ===== 配置 =====
DATA_DIR = os.environ.get('DATA_DIR', '/data')  # Zeabur 挂载卷路径
BASE_DIR = DATA_DIR
PAPER_CAPITAL = 500000  # 纸面本金 50万
MARGIN_RATE = 0.10      # 保证金比例 10%
SL_PCT = 0.04           # 止损 4%
TP_PCT = 0.20           # 止盈 20%
SLIPPAGE = 0.0002       # 滑点 0.02%
COMMISSION = 0.0003     # 手续费 0.03%

# 模块与 paper_trade.py 同目录 (ASCII文件名, 兼容容器/中文路径)
from cci_calculations import calculate_stc, f_normalize
from optimal_params import OPTIMAL_PARAMS

import akshare as ak
import signal
import time as _time

# 监控品种 (OOS验证通过的)
WATCH_LIST = {
    '玻璃': {'code': 'FG0', 'exchange': 'CZCE', 'multiplier': 20, 'margin_rate': 0.12},
    '白银': {'code': 'AG0', 'exchange': 'SHFE', 'multiplier': 15, 'margin_rate': 0.12},
    '铜':   {'code': 'CU0', 'exchange': 'SHFE', 'multiplier': 5,  'margin_rate': 0.10},
    '铝':   {'code': 'AL0', 'exchange': 'SHFE', 'multiplier': 5,  'margin_rate': 0.10},
    '锡':   {'code': 'SN0', 'exchange': 'SHFE', 'multiplier': 1,  'margin_rate': 0.12},
}

# 路径
TRADE_LOG = os.path.join(BASE_DIR, 'paper_trade_log.csv')
STATE_FILE = os.path.join(BASE_DIR, 'paper_trade_state.json')

def calc_indicators(df):
    """计算CCI和STC"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(12).mean()
    mad = tp.rolling(12).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad)
    cci_ma = cci.rolling(20).mean()
    stc_raw = calculate_stc(df.copy(), length=10, fast=23, slow=50, aaa=0.5)
    stc = f_normalize(stc_raw, 20, 80)
    df = df.copy()
    df['cci'] = cci
    df['cci_ma'] = cci_ma
    df['stc'] = stc
    return df

def check_entry_signal(row, params):
    """检查入场信号 (基于当前行的cci/stc)"""
    cci = row['cci']
    cci_ma = row['cci_ma']
    stc = row['stc']
    cci_prev = row.get('cci_prev', None)
    cci_ma_prev = row.get('cci_ma_prev', None)

    if cci_prev is None or cci_ma_prev is None:
        return False

    # 信号1: CCI超卖 + STC超卖
    if cci < params['cci_oversold'] and stc >= params['stc_oversold']:
        return True
    # 信号2: CCI金叉 + CCI<=cci_cross_max + STC>=stc_cross
    if (cci_prev <= cci_ma_prev and cci > cci_ma and
        cci <= params['cci_cross_max'] and
        stc >= params['stc_cross']):
        return True
    return False

def check_exit_signal(row, params, entry_price, direction='LONG'):
    """检查出场信号"""
    cci = row['cci']
    cci_ma = row['cci_ma']
    cci_prev = row.get('cci_prev', None)
    cci_ma_prev = row.get('cci_ma_prev', None)

    price = row['close']
    pnl_pct = (price - entry_price) / entry_price if direction == 'LONG' else (entry_price - price) / entry_price

    # 止损4%
    if pnl_pct <= -SL_PCT:
        return 'SL', price
    # 止盈20%
    if pnl_pct >= TP_PCT:
        return 'TP', price
    # CCI超买
    if cci > params['cci_overbought']:
        return 'overbought', price
    # CCI死叉
    if cci_prev is not None and cci_ma_prev is not None:
        if cci_prev >= cci_ma_prev and cci < cci_ma:
            return 'death_cross', price
    return None, None

def get_latest_data(symbol, code):
    """通过akshare获取最新日线数据(需要足够warmup)"""
    # 获取最近1年数据 (足够计算CCI/STC)
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y%m%d')

    try:
        df = ak.futures_main_sina(symbol=code, start_date=start_date, end_date=end_date)
        df = df.rename(columns={
            '日期': 'date', '开盘价': 'open', '最高价': 'high',
            '最低价': 'low', '收盘价': 'close', '成交量': 'volume'
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        return df
    except Exception as e:
        print(f"  [ERROR] {symbol} 数据获取失败: {e}")
        return None

def load_state():
    """加载纸面交易状态"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'positions': {}, 'trades': [], 'equity': [], 'balance': PAPER_CAPITAL}

def save_state(state):
    """保存纸面交易状态"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)

def load_trade_log():
    """加载交易记录"""
    if os.path.exists(TRADE_LOG):
        return pd.read_csv(TRADE_LOG)
    return pd.DataFrame()

def append_trade_log(trade):
    """追加交易记录"""
    df = pd.DataFrame([trade])
    if os.path.exists(TRADE_LOG):
        df.to_csv(TRADE_LOG, mode='a', header=False, index=False)
    else:
        df.to_csv(TRADE_LOG, index=False)

# ===== 交易数据库 (SQLite) =====
# 每笔已平仓交易清晰记录: 开仓价/平仓价/盈亏/胜率 等, 落在持久卷 /data 上
import sqlite3
DB_FILE = os.path.join(BASE_DIR, 'paper_trade.db')

def recompute_db_pnl():
    """幂等重算数据库内所有平仓记录的 PnL (用正确公式).

    用于修正历史旧数据 (早期版本 PnL 公式错误), 每次启动调用无害。
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        rows = conn.execute(
            "SELECT id, direction, entry_price, exit_price, lots, multiplier FROM closed_trades"
        ).fetchall()
        for id_, direction, ep, xp, lots, mult in rows:
            price_diff = (xp - ep) if direction == 'LONG' else (ep - xp)
            pnl = price_diff * (mult or 1) * (lots or 1)
            pnl_pct = (price_diff / ep * 100) if ep else 0.0
            conn.execute(
                "UPDATE closed_trades SET pnl=?, pnl_pct=?, is_win=? WHERE id=?",
                (pnl, pnl_pct, 1 if pnl > 0 else 0, id_))
        conn.commit()
    finally:
        conn.close()


def init_db():
    """初始化平仓交易数据库 (幂等)"""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""CREATE TABLE IF NOT EXISTS closed_trades (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol      TEXT NOT NULL,   -- 品种名 (如 铜)
        code        TEXT,            -- 合约代码 (如 CU0)
        direction   TEXT,            -- LONG / SHORT
        entry_date  TEXT,            -- 开仓日期
        entry_price REAL,            -- 开仓价格
        exit_date   TEXT,            -- 平仓日期
        exit_price  REAL,            -- 平仓价格
        lots        INTEGER,         -- 手数
        multiplier  REAL,            -- 合约乘数
        pnl         REAL,            -- 盈亏 (正=盈利, 负=亏损)
        pnl_pct     REAL,            -- 盈亏百分比
        is_win      INTEGER,         -- 1=盈利 0=亏损
        reason      TEXT,            -- 平仓原因 (SL/TP/overbought/death_cross)
        created_at  TEXT DEFAULT (datetime('now'))
    )""")
    conn.commit()
    conn.close()
    recompute_db_pnl()  # 修正历史/异常 PnL 记录

def record_closed_trade(trade):
    """记录一笔已平仓交易到数据库 (按 品种+开仓日+平仓日 去重, 可重复调用)

    PnL 一律用开/平仓价重算 (期货PnL=价格差×乘数×手数), 不信任调用方传入的 pnl,
    以保证历史补录与未来平仓口径一致。
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        exists = conn.execute(
            "SELECT 1 FROM closed_trades WHERE symbol=? AND entry_date=? AND exit_date=?",
            (trade.get('symbol'), trade.get('entry_date'), trade.get('exit_date'))
        ).fetchone()
        if not exists:
            ep = trade.get('entry_price') or 0
            xp = trade.get('exit_price') or 0
            lots = trade.get('lots') or 1
            mult = trade.get('multiplier') or 1
            direction = trade.get('side', 'LONG')
            price_diff = (xp - ep) if direction == 'LONG' else (ep - xp)
            pnl = price_diff * mult * lots
            pnl_pct = (price_diff / ep * 100) if ep else 0.0
            conn.execute("""INSERT INTO closed_trades
                (symbol, code, direction, entry_date, entry_price, exit_date, exit_price,
                 lots, multiplier, pnl, pnl_pct, is_win, reason)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (trade.get('symbol'),
                 WATCH_LIST.get(trade.get('symbol'), {}).get('code', ''),
                 direction,
                 trade.get('entry_date'), ep,
                 trade.get('exit_date'), xp,
                 lots, mult,
                 pnl, pnl_pct,
                 1 if pnl > 0 else 0,
                 trade.get('reason', '')))
            conn.commit()
    finally:
        conn.close()

def migrate_existing_trades(state):
    """首次运行: 将历史 JSON 中已有的平仓记录补录进数据库"""
    init_db()
    n = 0
    for t in state.get('trades', []):
        if t.get('action') == 'EXIT':
            record_closed_trade(t)
            n += 1
    return n

def get_db_summary():
    """从数据库汇总: 胜率 / 总盈利 / 总亏损 / 净盈亏"""
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT pnl FROM closed_trades").fetchall()
    conn.close()
    total = len(rows)
    wins = sum(1 for (p,) in rows if p > 0)
    losses = total - wins
    win_rate = (wins / total * 100) if total else 0.0
    total_profit = sum(p for (p,) in rows if p > 0)
    total_loss = sum(p for (p,) in rows if p < 0)
    return {
        'total': total, 'wins': wins, 'losses': losses,
        'win_rate': win_rate,
        'total_profit': total_profit, 'total_loss': total_loss,
        'net': total_profit + total_loss,
    }

def print_db_report(state=None):
    """打印清晰的交易台账: 已平仓 + 当前持仓, 开仓价一览无余

    state: 可选, 传入则一并展示当前持仓(含开仓价/浮盈亏); 不传则只展示已平仓。
    """
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(
        "SELECT symbol, direction, entry_date, entry_price, exit_date, exit_price, "
        "pnl, pnl_pct, is_win, reason FROM closed_trades ORDER BY exit_date"
    ).fetchall()
    conn.close()
    s = get_db_summary()

    positions = state.get('positions', {}) if state else {}

    print(f"\n{'='*94}")
    print(f"交易台账  (SQLite: {os.path.basename(DB_FILE)})  —  已平仓 + 当前持仓")
    print(f"{'='*94}")
    hdr = (f"| {'品种':<5} | {'方向':<4} | {'开仓日期':<10} | {'开仓价':>10} | "
           f"{'平/现价日期':<12} | {'平/现价':>10} | {'盈亏':>12} | {'盈亏%':>7} | {'状态':<5} | 原因")
    print(hdr)
    print(f"|{'-'*7}|{'-'*6}|{'-'*12}|{'-'*12}|{'-'*14}|{'-'*12}|{'-'*14}|{'-'*9}|{'-'*7}|{'-'*6}")
    # 已平仓
    for symbol, direction, ed, ep, xd, xp, pnl, pct, is_win, reason in rows:
        res = '盈利' if is_win else '亏损'
        print(f"| {symbol:<5} | {direction:<4} | {str(ed):<10} | {float(ep):>10.0f} | "
              f"{str(xd):<12} | {float(xp):>10.0f} | {float(pnl):>+12.1f} | {float(pct):>6.1f}% | {'已平':<5} | {reason}")
    # 当前持仓 (浮盈亏, 以最近一次扫描收盘价估算)
    for name, pos in positions.items():
        ep = pos.get('entry_price', 0) or 0
        ed = pos.get('entry_date', '') or ''
        cp = pos.get('current_price', None)
        up = pos.get('unrealized_pnl', None)
        upct = pos.get('unrealized_pct', None)
        cd = pos.get('current_date', '') or datetime.now().strftime('%Y-%m-%d')
        if cp is not None and up is not None:
            price_col = f"{float(cp):>10.0f}"
            pnl_col = f"{float(up):>+12.1f}"
            pct_col = f"{float(upct):>6.1f}%"
            reason_col = '浮盈亏'
        else:
            price_col = f"{'(未更新)':>10}"
            pnl_col = f"{'--':>12}"
            pct_col = f"{'--':>7}"
            reason_col = '待扫描'
        print(f"| {name:<5} | {pos.get('direction',''):<4} | {str(ed):<10} | {float(ep):>10.0f} | "
              f"{str(cd):<12} | {price_col} | {pnl_col} | {pct_col} | {'持仓中':<5} | {reason_col}")
    if not rows and not positions:
        print("| (暂无记录)")
    print(f"{'='*94}")
    print(f"【已平仓统计】胜率: {s['win_rate']:.1f}%   ({s['wins']}胜 / {s['losses']}负, 共 {s['total']} 笔)")
    print(f"  总盈利: {s['total_profit']:>+14,.1f}   总亏损: {s['total_loss']:>+14,.1f}   净盈亏: {s['net']:>+14,.1f}")
    print(f"【当前持仓】{len(positions)} 个 (浮盈亏以最近一次扫描收盘价估算, 未实现, 不计入胜率)")
    print(f"{'='*94}")

def run_daily_scan():
    """每日扫描: 检查所有品种的信号"""
    state = load_state()
    init_db()
    migrate_existing_trades(state)
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*70}")
    print(f"纸面交易扫描 — {today}")
    print(f"本金: {state['balance']:,.0f} | 当前持仓: {len(state['positions'])}个")
    print(f"{'='*70}")

    new_trades = 0

    for name, cfg in WATCH_LIST.items():
        params = OPTIMAL_PARAMS[name]
        print(f"\n--- {name} ({cfg['code']}) ---")

        # 获取数据
        df = get_latest_data(name, cfg['code'])
        if df is None or len(df) < 40:
            print(f"  数据不足, 跳过")
            continue

        # 计算指标
        df = calc_indicators(df)
        df['cci_prev'] = df['cci'].shift(1)
        df['cci_ma_prev'] = df['cci_ma'].shift(1)

        # 检查是否有持仓
        pos = state['positions'].get(name, None)

        if pos:
            # --- 管理现有持仓 ---
            print(f"  持仓中: 入场{pos['entry_date']} 价格{pos['entry_price']:.0f} "
                  f"方向{pos['direction']} 止损{pos['stop_loss']:.0f}")

            # 记录当前价与浮盈亏 (供 report 台账展示, 无需联网)
            current_price = df.iloc[-1]['close']
            price_diff_c = (current_price - pos['entry_price']) if pos['direction'] == 'LONG' else (pos['entry_price'] - current_price)
            pnl_pct_c = price_diff_c / pos['entry_price']
            pos['current_price'] = float(current_price)
            pos['current_date'] = today
            pos['unrealized_pnl'] = float(price_diff_c * cfg['multiplier'] * pos['lots'])
            pos['unrealized_pct'] = float(pnl_pct_c * 100)

            # 用最新bar检查出场信号
            latest_row = df.iloc[-1]
            exit_signal, exit_price = check_exit_signal(
                latest_row, params, pos['entry_price'], pos['direction'])

            if exit_signal:
                # 平仓
                # 期货PnL = 价格差 × 合约乘数 × 手数 (LONG: 价涨盈利; SHORT: 价跌盈利)
                price_diff = (exit_price - pos['entry_price']) if pos['direction'] == 'LONG' else (pos['entry_price'] - exit_price)
                pnl_pct = price_diff / pos['entry_price']
                pnl = price_diff * cfg['multiplier'] * pos['lots']
                state['balance'] += pnl

                trade = {
                    'date': today, 'symbol': name, 'side': pos['direction'],
                    'action': 'EXIT', 'reason': exit_signal,
                    'entry_date': pos['entry_date'], 'exit_date': today,
                    'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    'pnl_pct': pnl_pct * 100, 'pnl': pnl,
                    'lots': pos['lots'], 'multiplier': cfg['multiplier']
                }
                append_trade_log(trade)
                state['trades'].append(trade)
                record_closed_trade(trade)
                state['equity'].append(state['balance'])

                print(f"  [平仓] {exit_signal} @ {exit_price:.0f} PnL {pnl_pct*100:+.1f}%")
                del state['positions'][name]
                new_trades += 1
            else:
                # 检查是否需要移动止损 (如果盈利>5%, 止损上移)
                current_price = df.iloc[-1]['close']
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] if pos['direction'] == 'LONG' else (pos['entry_price'] - current_price) / pos['entry_price']
                if pnl_pct > 0.05:
                    new_sl = pos['entry_price'] * (1 + pnl_pct * 0.5) if pos['direction'] == 'LONG' else pos['entry_price'] * (1 - pnl_pct * 0.5)
                    pos['stop_loss'] = new_sl

        else:
            # --- 检查入场信号 ---
            latest_row = df.iloc[-1]
            prev_row = df.iloc[-2]

            # 构造prev_row的属性用于check_entry_signal
            latest_row = latest_row.copy()
            latest_row['cci_prev'] = prev_row['cci']
            latest_row['cci_ma_prev'] = prev_row['cci_ma']

            if check_entry_signal(latest_row, params):
                # 入场: 次日开盘价 (今天收盘价确定信号, 明天开盘成交)
                # 用最新bar的close作为信号确认, entry = 明天的open
                entry_price = df.iloc[-1]['close'] * (1 + SLIPPAGE)

                # 计算仓位: 30%保证金上限
                margin_per_lot = entry_price * cfg['multiplier'] * cfg['margin_rate']
                cap = state['balance'] * 0.30  # 30%保证金上限
                lots = int(cap / margin_per_lot)
                lots = max(0, lots)

                if lots >= 1:
                    # 入场
                    state['positions'][name] = {
                        'direction': 'LONG',
                        'entry_date': today,
                        'entry_price': entry_price,
                        'lots': lots,
                        'stop_loss': entry_price * (1 - SL_PCT),
                        'multiplier': cfg['multiplier']
                    }
                    fee = entry_price * cfg['multiplier'] * lots * COMMISSION * 2
                    state['balance'] -= fee

                    trade = {
                        'date': today, 'symbol': name, 'side': 'LONG',
                        'action': 'ENTRY', 'reason': 'cci_oversold/golden_cross',
                        'entry_date': today, 'exit_date': '',
                        'entry_price': entry_price, 'exit_price': '',
                        'pnl_pct': 0, 'pnl': 0,
                        'lots': lots, 'multiplier': cfg['multiplier']
                    }
                    append_trade_log(trade)
                    state['trades'].append(trade)
                    state['equity'].append(state['balance'])

                    print(f"  [入场] 信号触发! 价格{entry_price:.0f} "
                          f"手数{lots} 止损{entry_price*(1-SL_PCT):.0f}")
                    new_trades += 1
                else:
                    need_pct = min(margin_per_lot / state['balance'], 1.0) * 100
                    print(f"  [无仓位] 信号触发但保证金不足 (需要{need_pct:.0f}%保证金)")
            else:
                print(f"  [等待] 无信号")

    # 保存状态
    state['last_scan'] = today
    save_state(state)

    # 汇总
    print(f"\n{'='*70}")
    print(f"扫描完成 — 新增交易: {new_trades} | 当前持仓: {len(state['positions'])}个")
    print(f"当前权益: {state['balance']:,.0f}")

    # 统计历史交易 (直接来自数据库, 清晰聚合)
    s = get_db_summary()
    if s['total']:
        print(f"历史: {s['total']}笔平仓, 胜率{s['win_rate']:.1f}%, "
              f"总盈利{s['total_profit']:+,.0f} 总亏损{s['total_loss']:+,.0f} "
              f"净盈亏{s['net']:+,.0f}")

    return state

def show_status():
    """显示当前状态"""
    state = load_state()
    print(f"\n{'='*70}")
    print(f"纸面交易状态")
    print(f"{'='*70}")
    print(f"本金: {state['balance']:,.0f}")
    print(f"持仓: {len(state['positions'])}个")
    for name, pos in state['positions'].items():
        print(f"  {name}: {pos['direction']} 入场{pos['entry_date']} "
              f"价格{pos['entry_price']:.0f} 止损{pos['stop_loss']:.0f}")

    if state['trades']:
        exits = [t for t in state['trades'] if t['action'] == 'EXIT']
        if exits:
            wins = sum(1 for t in exits if t['pnl'] > 0)
            total_pnl = sum(t['pnl'] for t in exits)
            print(f"\n交易统计: {len(exits)}笔平仓, 胜率{wins/len(exits)*100:.1f}%")
            print(f"累计PnL: {total_pnl:+,.0f}")
            print(f"\n最近5笔:")
            for t in state['trades'][-5:]:
                if t['action'] == 'EXIT':
                    print(f"  {t['date']} {t['symbol']} {t['reason']:<12} "
                          f"入{t['entry_price']:.0f} 出{t['exit_price']:.0f} "
                          f"PnL {t['pnl']:+,.0f}")
    print(f"{'='*70}")
    print_db_report(state)
    print(f"{'='*70}")

def reset():
    """重置纸面交易"""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    if os.path.exists(TRADE_LOG):
        os.remove(TRADE_LOG)
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
        except OSError:
            print("  [警告] 交易数据库文件被占用, 未能删除 (下次启动会自动重建为空库)")
    print("纸面交易状态已重置 (含交易数据库)")

def run_daemon():
    """常驻模式: 工作日 15:35 自动扫描，24小时保持运行"""
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"[daemon] 数据目录: {BASE_DIR}")
    print(f"[daemon] 工作日 15:35 自动扫描，Ctrl-C 退出")

    def _scan_once():
        try:
            run_daily_scan()
        except Exception as e:
            print(f"[daemon] 扫描异常: {e}")
            import traceback; traceback.print_exc()

    _scan_once()  # 启动先跑一次

    while True:
        now = datetime.now()
        # 工作日(周一~周五) 15:35 触发
        if now.weekday() < 5 and now.hour == 15 and now.minute == 35:
            _scan_once()
            _time.sleep(61)  # 避免同分钟内重复触发
        _time.sleep(30)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'scan':
            run_daily_scan()
        elif cmd == 'status':
            show_status()
        elif cmd == 'report':
            init_db()
            print_db_report(load_state())
        elif cmd == 'reset':
            reset()
        elif cmd == 'daemon':
            run_daemon()
        else:
            print(f"用法: python paper_trade.py [scan|status|reset|daemon]")
    else:
        run_daily_scan()
