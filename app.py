import os
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import requests

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import vectorbt as vbt

# Fear & Greed (CNN)
try:
    from fear_greed import get_fear_and_greed
    HAS_FEAR_GREED_LIB = True
except ImportError:
    HAS_FEAR_GREED_LIB = False

# ==============================
# 基本設定與工具函數
# ==============================

US_TZ = timezone(timedelta(hours=-5))  # 美東時間

ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_API_SECRET = os.getenv("APCA_API_SECRET_KEY")

@st.cache_resource(show_spinner=False)
def get_alpaca_client():
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        return None
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)


@st.cache_data(show_spinner=False)
def get_bars(symbol, start, end, timeframe: TimeFrame):
    client = get_alpaca_client()
    if client is None:
        return pd.DataFrame()

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment=None,
        limit=10000,
    )
    bars = client.get_stock_bars(req)
    if symbol not in bars:
        return pd.DataFrame()

    df = bars[symbol].df.reset_index()
    df = df.rename(
        columns={
            "timestamp": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(US_TZ)
    return df


def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()

    # RSI 14
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df["rsi14"] = rsi.values

    # ATR 14
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    # VWAP（對 intraday 用）
    if "vwap" not in df.columns:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        df["cum_vol"] = df["volume"].cumsum()
        df["cum_vp"] = (typical_price * df["volume"]).cumsum()
        df["vwap"] = df["cum_vp"] / df["cum_vol"]

    # Chaikin Money Flow (CMF 20)
    period = 20
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    )
    mfm = mfm.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    mfv = mfm * df["volume"]
    df["cmf20"] = mfv.rolling(period).sum() / df["volume"].rolling(period).sum()

    return df


def classify_rsi_zone(rsi):
    if np.isnan(rsi):
        return "－"
    if rsi >= 70:
        return "超買區"
    if rsi <= 30:
        return "超賣區"
    return "正常"


def get_fear_greed_value():
    # 優先用 fear-greed 套件，否則用簡單 HTTP 備援
    try:
        if HAS_FEAR_GREED_LIB:
            data = get_fear_and_greed()
            # library 回傳格式：{"fear_and_greed": {"now": {"value": 67, "text": "Greed"}, ...}}
            val = data["fear_and_greed"]["now"]["value"]
            return int(val)
    except Exception:
        pass

    # 簡單 HTTP 備援（如失敗則回 None）
    try:
        resp = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            timeout=5,
        )
        if resp.ok:
            js = resp.json()
            latest = js["fear_and_greed"]["now"]["score"]
            return int(latest)
    except Exception:
        return None

    return None


def interpret_fear_greed(val: int) -> str:
    if val is None:
        return "無法取得 Fear & Greed 指數"
    if val > 75:
        return "極度貪婪：市場高風險區，避免追高，適合分批止盈與提高止損。"
    if val < 30:
        return "極度恐慌：泥沙俱下，往往是黃金建倉區，留意大膽進場機會。"
    if 40 <= val <= 70:
        return "正常震盪區：適合做突破交易與趨勢跟隨。"
    return "中性區：可按照個股技術面靈活操作。"


# ==============================
# 股票池與 RS 計算
# ==============================

@st.cache_data(show_spinner=True)
def load_universe():
    # 你可以改成從 CSV 或資料庫載入，只要回傳有 symbol / name / sector 欄位的 DataFrame 即可
    data = [
        ("AAPL", "Apple", "Technology"),
        ("MSFT", "Microsoft", "Technology"),
        ("NVDA", "NVIDIA", "Technology"),
        ("GOOGL", "Alphabet", "Communication"),
        ("META", "Meta Platforms", "Communication"),
        ("AMZN", "Amazon", "Consumer Discretionary"),
        ("TSLA", "Tesla", "Consumer Discretionary"),
        ("AVGO", "Broadcom", "Technology"),
        ("LLY", "Eli Lilly", "Healthcare"),
        ("JPM", "JPMorgan", "Financial"),
        ("GS", "Goldman Sachs", "Financial"),
        ("XOM", "Exxon Mobil", "Energy"),
    ]
    df = pd.DataFrame(data, columns=["symbol", "name", "sector"])
    return df


@st.cache_data(show_spinner=True)
def compute_relative_strength(
    base_symbol: str = "SPY",
    lookback_days: int = 60,
):
    today = datetime.now(tz=US_TZ)
    start = today - timedelta(days=lookback_days * 2)

    universe = load_universe()
    symbols = universe["symbol"].tolist()
    if base_symbol not in symbols:
        symbols.append(base_symbol)

    # 取日線
    rs_data = []
    for sym in symbols:
        df = get_bars(sym, start, today, TimeFrame.Day)
        if df.empty:
            continue
        df = df.sort_values("timestamp")
        df = add_technicals(df)
        # 近 lookback_days 報酬
        df_recent = df.tail(lookback_days)
        if len(df_recent) < lookback_days // 2:
            continue
        ret = df_recent["close"].iloc[-1] / df_recent["close"].iloc[0] - 1.0
        rs_data.append((sym, ret))

    rs_df = pd.DataFrame(rs_data, columns=["symbol", "ret"])
    if rs_df.empty or base_symbol not in rs_df["symbol"].values:
        return pd.DataFrame()

    base_ret = rs_df.loc[rs_df["symbol"] == base_symbol, "ret"].iloc[0]
    rs_df = rs_df[rs_df["symbol"] != base_symbol]
    rs_df = rs_df.merge(load_universe(), on="symbol", how="left")
    rs_df["rs_score"] = rs_df["ret"] - base_ret
    rs_df = rs_df.sort_values("rs_score", ascending=False)
    return rs_df


# ==============================
# Day Trade / VWAP / ATR / RVOL
# ==============================

def calc_rvol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    vol = df["volume"]
    avg_vol = vol.rolling(window).mean()
    rvol = vol / avg_vol
    return rvol


def evaluate_daytrade_conditions(df: pd.DataFrame):
    if df.empty or len(df) < 30:
        return None

    df = df.copy()
    df = add_technicals(df)
    last = df.iloc[-1]

    price = last["close"]
    ma5 = last["ma5"]
    ma20 = last["ma20"]
    rsi = last["rsi14"]
    atr = last["atr14"]
    vwap = last["vwap"]
    cmf = last["cmf20"]

    cond_price_ma = price > ma5 and price > ma20
    cond_rsi = 45 <= rsi <= 65
    cond_vwap = price > vwap
    cond_cmf = cmf > 0.1  # 主力吸籌

    return {
        "price": price,
        "ma5": ma5,
        "ma20": ma20,
        "rsi14": rsi,
        "atr14": atr,
        "vwap": vwap,
        "cmf20": cmf,
        "cond_price_ma": cond_price_ma,
        "cond_rsi": cond_rsi,
        "cond_vwap": cond_vwap,
        "cond_cmf": cond_cmf,
    }


def atr_adaptive_stops(last_price, atr_value, atr_window_pct, stop_min_pct, stop_max_pct):
    # atr_window_pct: ATR / Close 的百分比
    if np.isnan(atr_value) or atr_value <= 0:
        stop_loss_pct = stop_min_pct
    else:
        atr_pct = atr_value / last_price
        stop_loss_pct = atr_pct * atr_window_pct  # 例如 1.5 倍 ATR
        stop_loss_pct = max(stop_min_pct, min(stop_loss_pct, stop_max_pct))

    take_profit_pct = stop_loss_pct * 1.8  # 簡單設為 1.8 倍 RR
    stop_loss_price = last_price * (1 - stop_loss_pct)
    take_profit_price = last_price * (1 + take_profit_pct)

    return {
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
    }


# ==============================
# AI 參數優化 (vectorbt)
# ==============================

def run_param_search(symbol: str, years: int = 2, n_samples: int = 100):
    today = datetime.now(tz=US_TZ)
    start = today - timedelta(days=365 * years)

    df = get_bars(symbol, start, today, TimeFrame.Day)
    if df.empty or len(df) < 60:
        return None

    df = df.sort_values("timestamp").reset_index(drop=True)
    price = df["close"]

    best_cfg = None
    best_final_equity = -np.inf

    # 隨機抽樣 100 組（RSI 買入 25~45, MA 10~30）
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        rsi_buy = int(rng.integers(25, 46))
        ma_window = int(rng.integers(10, 31))

        # 計算 MA 與 RSI
        ma = price.rolling(ma_window).mean()
        delta = price.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(gain).rolling(14).mean()
        roll_down = pd.Series(loss).rolling(14).mean()
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))

        entries = (price > ma) & (rsi < rsi_buy)
        exits = (rsi > 70) | (price < ma)

        pf = vbt.Portfolio.from_signals(
            price,
            entries=entries,
            exits=exits,
            init_cash=100000,
            fees=0.0005,
            freq="1D",
        )
        final_equity = pf.values[-1]

        if final_equity > best_final_equity:
            best_final_equity = final_equity
            best_cfg = {
                "rsi_buy": rsi_buy,
                "ma_window": ma_window,
                "final_equity": final_equity,
                "pf": pf,
            }

    return best_cfg


def generate_signal_from_cfg(symbol: str, cfg: dict):
    today = datetime.now(tz=US_TZ)
    start = today - timedelta(days=365)

    df = get_bars(symbol, start, today, TimeFrame.Day)
    if df.empty:
        return None

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_technicals(df)
    last = df.iloc[-1]

    price = last["close"]
    ma = df["close"].rolling(cfg["ma_window"]).mean().iloc[-1]
    rsi = last["rsi14"]

    if price > ma and rsi < cfg["rsi_buy"]:
        signal = "Buy"
    elif price < ma or rsi > 70:
        signal = "Sell"
    else:
        signal = "Hold"

    # ATR-based 風控
    atr = last["atr14"]
    atr_pct = atr / price if atr and price else 0.02
    stop_loss_price = price * (1 - atr_pct * 1.5)
    take_profit_price = price * (1 + atr_pct * 2.0)

    return {
        "signal": signal,
        "last_price": price,
        "ma": ma,
        "rsi14": rsi,
        "atr14": atr,
        "atr_pct": atr_pct,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
    }


# ==============================
# Smart Money Board (預留)
# ==============================

def get_smart_money_metrics(symbol: str):
    # 這裡先預留欄位，之後可接其他 API（如 FMP / Finnhub）
    return {
        "institutional_held": None,
        "insider_held": None,
        "short_float": None,
    }


# ==============================
# Streamlit 介面
# ==============================

st.set_page_config(
    page_title="美股 AI 強勢股雷達 v0.1",
    layout="wide",
)

st.title("🚀 美股 AI 強勢股雷達 v0.1")

# ---- Sidebar ----
st.sidebar.header("⚙️ 系統設定")

mode = st.sidebar.radio(
    "交易模式",
    ["波段/位置交易", "日內極速 (Day Trade)"],
)

rs_lookback = st.sidebar.slider("RS Lookback 天數", 30, 120, 60, step=5)
daytrade_stop_min = st.sidebar.slider("日內止損下限 (%)", 1.0, 5.0, 3.0, step=0.5)
daytrade_stop_max = st.sidebar.slider("日內止損上限 (%)", 3.0, 10.0, 6.0, step=0.5)
atr_window_multiplier = st.sidebar.slider(
    "ATR 波動度乘數", 1.0, 2.5, 1.5, step=0.1
)

with st.sidebar.expander("🧠 AI 參數優化設定"):
    param_years = st.slider("回測年數", 1, 5, 2, step=1)
    param_samples = st.slider("參數樣本數 (類 Genetic)", 20, 200, 100, step=20)

manual_symbol = st.sidebar.text_input("🔍 個股診斷代號（如 NVDA）", value="NVDA")
run_ai_opt = st.sidebar.button("啟動無監督網格自學習優化")

# ==============================
# 區塊 1：大盤情緒 (Fear & Greed)
# ==============================

st.subheader("📊 大盤情緒診斷 - CNN Fear & Greed Index")

fg_val = get_fear_greed_value()
col1, col2 = st.columns([1, 3])

with col1:
    if fg_val is not None:
        if fg_val > 75:
            color = "red"
        elif fg_val < 30:
            color = "green"
        else:
            color = "orange"

        st.markdown(
            f"""
            <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center; color:white;">
                <div style="font-size:14px;">Fear & Greed</div>
                <div style="font-size:36px; font-weight:bold;">{fg_val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("無法取得 Fear & Greed 指數，請稍後再試。")

with col2:
    st.write(interpret_fear_greed(fg_val))

# ==============================
# 區塊 2：RS 強勢股 & 行業領頭羊
# ==============================

st.subheader("🏆 全市場 RS 強勢股 & 行業領頭羊（基於 SPY）")

rs_df = compute_relative_strength(lookback_days=rs_lookback)

if rs_df.empty:
    st.warning("無法取得 RS 排名，請檢查 Alpaca API 是否設定正確。")
else:
    sort_order = st.radio("RS 排序方向", ["由強到弱", "由弱到強"], horizontal=True)
    ascending = sort_order == "由弱到強"
    rs_df_display = rs_df.copy()
    rs_df_display["RS% vs SPY"] = (rs_df_display["rs_score"] * 100).round(2)
    rs_df_display["近期報酬%"] = (rs_df_display["ret"] * 100).round(2)
    rs_df_display = rs_df_display[["symbol", "name", "sector", "近期報酬%", "RS% vs SPY"]]
    rs_df_display = rs_df_display.sort_values("RS% vs SPY", ascending=ascending)

    st.dataframe(rs_df_display, use_container_width=True)

    # 每日各行業領頭羊
    st.markdown("**各行業 RS 領頭羊**")
    leaders = (
        rs_df.assign(rank=rs_df["rs_score"].rank(ascending=False, method="first"))
        .sort_values(["sector", "rank"])
        .groupby("sector")
        .head(1)
    )
    leaders_display = leaders.copy()
    leaders_display["RS% vs SPY"] = (leaders_display["rs_score"] * 100).round(2)
    leaders_display["近期報酬%"] = (leaders_display["ret"] * 100).round(2)
    leaders_display = leaders_display[
        ["sector", "symbol", "name", "近期報酬%", "RS% vs SPY"]
    ]
    st.dataframe(leaders_display, use_container_width=True)

# ==============================
# 區塊 3：開市前 & 開盤衝刺 VWAP + RVOL 雷達
# ==============================

st.subheader("🔥 開市前 5–10 分鐘 & 開盤 15 分鐘 VWAP + RVOL 衝刺雷達")

if rs_df.empty:
    st.info("等待 RS 列表載入完成後再顯示衝刺雷達。")
else:
    # 取 RS 前幾名作為候選（全美股會太重，這裡先用前 20 名）
    top_symbols = rs_df.head(20)["symbol"].tolist()
    today = datetime.now(tz=US_TZ)
    start_intraday = today - timedelta(hours=4)

    breakout_rows = []
    for sym in top_symbols:
        df_i = get_bars(sym, start_intraday, today, TimeFrame.Minute)
        if df_i.empty:
            continue
        df_i = df_i.sort_values("timestamp")
        df_i = add_technicals(df_i)
        df_i["rvol"] = calc_rvol(df_i, window=20)
        last = df_i.iloc[-1]

        price = last["close"]
        vwap = last["vwap"]
        rvol = last["rvol"]
        cond_breakout = (price > vwap) and (rvol is not None) and (rvol > 1.5)

        breakout_rows.append(
            {
                "symbol": sym,
                "price": round(price, 2),
                "vwap": round(vwap, 2),
                "rvol": round(rvol, 2) if not math.isinf(rvol) and not math.isnan(rvol) else None,
                "status": "🔥 Breakout!" if cond_breakout else "",
            }
        )

    if not breakout_rows:
        st.info("目前沒有符合 VWAP + RVOL 條件的爆發股候選。")
    else:
        breakout_df = pd.DataFrame(breakout_rows)
        breakout_df = breakout_df.sort_values("rvol", ascending=False)
        st.dataframe(breakout_df, use_container_width=True)

# ==============================
# 區塊 4：個股診斷 & AI 決策
# ==============================

st.subheader("🩺 個股診斷 & AI 決策燈號")

symbol = manual_symbol.strip().upper()
if symbol:
    today = datetime.now(tz=US_TZ)
    start_daily = today - timedelta(days=365 * 2)

    df_daily = get_bars(symbol, start_daily, today, TimeFrame.Day)
    if df_daily.empty:
        st.warning(f"無法取得 {symbol} 日線數據。請確認代號或 Alpaca 權限。")
    else:
        df_daily = add_technicals(df_daily.sort_values("timestamp"))
        last_d = df_daily.iloc[-1]
        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("最新收盤價", f"{last_d['close']:.2f}")
            st.metric("RSI(14)", f"{last_d['rsi14']:.1f}", help=classify_rsi_zone(last_d["rsi14"]))

        with colB:
            st.metric("MA5 / MA20", f"{last_d['ma5']:.2f} / {last_d['ma20']:.2f}")
            st.metric("MA50 / MA200", f"{last_d['ma50']:.2f} / {last_d['ma200']:.2f}")

        with colC:
            st.metric("ATR(14)", f"{last_d['atr14']:.2f}")
            st.metric("CMF(20)", f"{last_d['cmf20']:.2f}")

        st.write(f"RSI 狀態：**{classify_rsi_zone(last_d['rsi14'])}**")

        # Smart Money Board (預留)
        sm = get_smart_money_metrics(symbol)
        st.markdown("#### 💼 華爾街籌碼追蹤（預留）")
        st.write(
            f"- 機構持股比例：{sm['institutional_held'] if sm['institutional_held'] is not None else '待接 API'}"
        )
        st.write(
            f"- 內部人持股比例：{sm['insider_held'] if sm['insider_held'] is not None else '待接 API'}"
        )
        st.write(
            f"- 空頭持倉佔比：{sm['short_float'] if sm['short_float'] is not None else '待接 API'}"
        )

        # AI 參數優化
        if run_ai_opt:
            with st.spinner("🧠 AI 正在為你摸索黃金參數組合（約 1–3 秒）..."):
                cfg = run_param_search(
                    symbol, years=param_years, n_samples=param_samples
                )
            if cfg is None:
                st.error("參數優化失敗，可能是歷史數據不足。")
            else:
                st.success(
                    f"🎆 AI 完成自學習！最佳組合：RSI 買入 < {cfg['rsi_buy']}, MA{cfg['ma_window']}。"
                )
                pf = cfg["pf"]
                st.write(
                    f"模擬初始資金 100,000，最終權益約為 {pf.values[-1]:,.0f}。"
                )
                signal_info = generate_signal_from_cfg(symbol, cfg)
                if signal_info:
                    sig_color = {
                        "Buy": "🟢 Buy",
                        "Hold": "🟡 Hold",
                        "Sell": "🔴 Sell",
                    }.get(signal_info["signal"], "⚪ 中性")

                    st.markdown(f"### 今日 AI 決策燈號：{sig_color}")
                    st.write(
                        f"- 現價：{signal_info['last_price']:.2f}  "
                        f"- MA{cfg['ma_window']}：{signal_info['ma']:.2f}  "
                        f"- RSI(14)：{signal_info['rsi14']:.1f}"
                    )
                    st.write(
                        f"- 建議建倉點：以現價附近分批建倉  "
                        f"- 止損點：約 {signal_info['stop_loss_price']:.2f}  "
                        f"- 目標止盈點：約 {signal_info['take_profit_price']:.2f}"
                    )
        else:
            st.info("如要啟動 AI 自學習優化，請在左側輸入代號後按按鈕。")

        # Day Trade 模式診斷
        st.markdown("---")
        st.markdown("### ⚡ 日內模式診斷 (5 分鐘 K 線 + VWAP + ATR)")

        start_intraday = today - timedelta(hours=6)
        df_5m = get_bars(symbol, start_intraday, today, TimeFrame.Minute)
        if df_5m.empty:
            st.warning("無法取得日內分時數據。")
        else:
            # 壓成 5 分鐘
            df_5m = df_5m.set_index("timestamp").resample("5T").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            df_5m = df_5m.dropna(subset=["open", "high", "low", "close"])
            df_5m.reset_index(inplace=True)
            df_5m = add_technicals(df_5m)
            info_intraday = evaluate_daytrade_conditions(df_5m)

            if not info_intraday:
                st.info("分時數據不足以作日內診斷。")
            else:
                dt_col1, dt_col2 = st.columns(2)
                with dt_col1:
                    st.write(
                        f"現價 {info_intraday['price']:.2f}，"
                        f"MA5 {info_intraday['ma5']:.2f}，MA20 {info_intraday['ma20']:.2f}"
                    )
                    st.write(
                        f"RSI(14) {info_intraday['rsi14']:.1f}，VWAP {info_intraday['vwap']:.2f}，CMF(20) {info_intraday['cmf20']:.2f}"
                    )

                with dt_col2:
                    atr = info_intraday["atr14"]
                    last_price = info_intraday["price"]
                    atr_pct = atr / last_price if atr and last_price else 0.02
                    stops = atr_adaptive_stops(
                        last_price,
                        atr,
                        atr_window_multiplier,
                        daytrade_stop_min / 100.0,
                        daytrade_stop_max / 100.0,
                    )
                    st.write(
                        f"ATR% ≈ {atr_pct*100:.2f}%，自適應止損 ≈ {stops['stop_loss_pct']*100:.2f}%"
                    )
                    st.write(
                        f"建議日內止損價 ≈ {stops['stop_loss_price']:.2f}，止盈價 ≈ {stops['take_profit_price']:.2f}"
                    )

                cond_buy = (
                    info_intraday["cond_price_ma"]
                    and info_intraday["cond_rsi"]
                    and info_intraday["cond_vwap"]
                )

                if mode == "日內極速 (Day Trade)":
                    if cond_buy:
                        st.markdown("### ✅ 日內條件符合：建議日內買入 (Day Trade Buy) 🟢")
                    else:
                        st.markdown("### ⏸ 日內條件未完全符合：暫時觀望 🟡")
                else:
                    st.info("目前處於波段/位置交易模式，如要使用日內判定，請在側邊選單切換。")

else:
    st.info("請在側邊欄輸入想診斷的股票代碼。")
