"""ChanSmartMoneyStrategy — 缠论骨架 × Smart Money 血肉 × BTC 市场锚

设计立场(对 smart-money-concepts 库与 SMC/ICT 体系的批判性采纳):

缠论提供"结构骨架": 笔/线段/中枢/递归级别, 时序上诚实(分型需后续K线确认)。
SMC/ICT/Wyckoff 提供"意图血肉": 流动性猎杀(Spring/UTAD)、失衡(FVG)、
资金时段(Killzone)。NFI 的精要提供"市场锚": 山寨是 BTC 的高贝塔。

对 smart-money-concepts 库的逐条评估见 _add_smc_columns 文档;
流动性池用 chan 笔极值(因果), FVG 因果版自实现, Killzone 直接采纳;
OB 不做门控(与结构止损位重叠+逆向选择), 留作 ML 特征。

入场四通道(选择性 BTC 闸门):
- 空头3a(需SMC共振+BTC非强) / 空头1p(直通) / 多头1p(需BTC非弱) / 多头3a(直通)
出场: 沿用 ChanFuturesStrategy(结构跟踪/中枢阶梯/背驰卖点/段级离场)。

ML 导出: 设置环境变量 CHAN_ML_EXPORT=1 时, populate 阶段把每个信号行
的全部因果特征 + 前视三重屏障标签落盘 user_data/ml_signals.csv,
供离线训练(前视仅用于离线标注, 不进入交易路径)。
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from ChanFuturesStrategy import ChanFuturesStrategy
from freqtrade.persistence import Trade


class ChanSmartMoneyStrategy(ChanFuturesStrategy):
    """在 ChanFuturesStrategy(默认 v8 稳健开关)之上叠加 SMC 共振过滤层"""

    # ---- SMC 参数 ----
    smc_score_required = 2        # 入场所需 SMC 共振分(猎杀2/FVG1/Killzone1)
    sweep_lookback = 3            # 猎杀事件有效窗口(根)
    sweep_stop_window = 4         # 猎杀极值止损的参考窗口(根)
    # 猎杀极值止损(SMC教义: 止损放流动性池外): v2实测比结构止损更宽,
    # 失败单亏损放大(空头1p +11.7->+5.9, 多头3a +1.3->-2.1), 默认关闭;
    # SMC 层的价值在入场过滤(空头3a +8.1->+12.2), 不在止损位
    use_sweep_stop = False
    fvg_half_mitigate = True      # FVG 以半位回补视为失效
    killzone_hours = (7, 8, 13, 14, 15, 16)  # UTC 伦敦/纽约开盘时段

    # ---- 1R 风险管理(Route A): 实验完成, 结论为负, 全部关闭 ----
    # 实验1(+1R落袋一半): 收益+26.4->+6.6, 胜率40.2->39.3 —— 169笔硬止损
    #        单从未触及1R(信号失败即直接止损), 分批只在最小价位切走赢家
    # 实验2(纯1R动态保本): 收益+26.4->+18.0, 胜率40.2->38.2 —— 保本位
    #        与结构无关, 回撤中被踢出的仓位本会恢复成大赢家
    # 结论: 本系统是"输家速死+赢家长跑"的趋势收割型, 胜率由信号质量决定,
    #        出场端任何干预只降不升; 抬胜率的唯一路径是ML信号提纯
    position_adjustment_enable = True
    enable_1r_partial = False
    partial_fraction = 0.5
    be_at_1r = False              # False = 使用固定 be_trigger(冠军配置)
    min_r_ratio = 0.012           # 1R 下限(过近的结构位止损不作为R基准)

    # ---- BTC 市场级闸门(NFI 设计精要: 山寨是 BTC 的高贝塔) ----
    # BTC 1h MA20 排列作为全市场 regime: 山寨做多需 BTC 非弱势, 做空需非强势
    market_anchor_pair = "BTC/USDT:USDT"

    # ------------------------------------------------------------------
    # 指标: 缠论回放(继承) + SMC 因果列 + BTC 市场锚
    # ------------------------------------------------------------------
    def informative_pairs(self):
        return [(self.market_anchor_pair, self.timeframe)]

    # ---- ML 门控(仅多头: AUC 0.655, thr=0.65 保留22%胜率68%; 空头AUC<0.53不部署) ----
    enable_ml_gate = True
    ml_proba_threshold = 0.65
    _ml_model = None
    _ml_feats = None

    def _load_ml_model(self):
        """惰性加载训练产物(train_ml.py 输出), 不存在则门控自动失效"""
        if self._ml_model is None:
            import joblib
            here = Path(__file__).resolve().parent
            candidates = [
                here.parent / "ml" / "ml_best_long.joblib",       # 本仓库
                Path("/freqtrade/user_data/ml/ml_best_long.joblib"),
                Path("/freqtrade/user_data/ml_best_long.joblib"),
            ]
            art = None
            for p in candidates:
                if p.is_file():
                    art = joblib.load(p)
                    break
            if art is None:
                self._ml_model = False   # 无模型, 门控直通
            else:
                self._ml_model, self._ml_feats = art["model"], art["feats"]
        return self._ml_model

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe = self._add_smc_columns(dataframe)
        dataframe = self._add_btc_regime(dataframe, metadata)
        dataframe = self._add_ml_proba(dataframe, metadata)
        if os.environ.get("CHAN_ML_EXPORT"):
            self._dump_ml_rows(dataframe, metadata)
        return dataframe

    def _add_ml_proba(self, df: DataFrame, metadata: dict) -> DataFrame:
        """多头信号质量分(与训练同构的因子 -> 模型概率), 因果逐行"""
        df["ml_proba"] = 0.0
        if not self.enable_ml_gate:
            return df
        model = self._load_ml_model()
        if not model:
            return df
        f = self._ml_derived(df, metadata)
        # 拼接基础因子, 补齐训练时三个特殊列(多头公式的 risk_dist)
        X = f.copy()
        X["hour"] = df["date"].dt.hour
        X["is_major"] = 1
        X["risk_dist"] = (df["close"] - df["chan_stop_ref"]) / df["close"]
        for c in self.ML_FEATURES:
            if c not in X.columns and c in df.columns:
                X[c] = df[c]
        X["chan_bsp_div"] = X.get("chan_bsp_div", 0)
        X = X[list(self._ml_feats)].fillna(0)
        df["ml_proba"] = model.predict_proba(X)[:, 1]
        return df

    # ------------------------------------------------------------------
    # ML 训练数据导出: 信号行的因果特征 + 前视三重屏障标签(仅离线标注用)
    # ------------------------------------------------------------------
    ML_FEATURES = [
        "chan_bsp_div", "chan_bsp_amp", "chan_stop_amp", "chan_bi_cnt",
        "chan_seg_dir", "chan_segseg_dir", "chan_ma_trend",
        "smc_sweep_bull_recent", "smc_sweep_bear_recent",
        "smc_fvg_bull", "smc_fvg_bear", "smc_killzone", "smc_btc_trend",
    ]
    ML_TP = 0.03          # 前视屏障: +3% 止盈
    ML_SL = -0.02         # 前视屏障: -2% 止损
    ML_HORIZON = 24       # 前视窗口: 24 根 1h

    def _ml_derived(self, df: DataFrame, metadata: dict) -> DataFrame:
        """ML 因子族(因果向量化): 导出与实盘打分共用同一实现, 杜绝训练/线上一致性漂移"""
        import numpy as np
        close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
        f = pd.DataFrame(index=df.index)
        rng = (high - low).replace(0, np.nan)
        f["candle_body"] = (close - df["open"]).abs() / rng
        f["wick_up"] = (high - pd.concat([close, df["open"]], axis=1).max(axis=1)) / rng
        f["wick_dn"] = (pd.concat([close, df["open"]], axis=1).min(axis=1) - low) / rng
        f["close_pos"] = (close - low) / rng
        f["ret1"] = close.pct_change()
        f["ret3"] = close.pct_change(3)
        f["ret6"] = close.pct_change(6)
        f["ret24"] = close.pct_change(24)
        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr20 = tr.rolling(20).mean()
        atr7 = tr.rolling(7).mean()
        f["atr20"] = atr20 / close
        f["atr_ratio"] = atr7 / atr20
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        f["boll_bw"] = 4 * std20 / ma20
        f["pct_b"] = (close - (ma20 - 2 * std20)) / (4 * std20)
        f["pct_rank100"] = close.rolling(100).apply(
            lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-12), raw=True
        )
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        f["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        f["macd_dif"] = dif / close
        f["macd_hist"] = (dif - dea) / close
        f["macd_hist_slope"] = f["macd_hist"].diff(3)
        for t in (5, 10, 20, 60):
            ma = close.rolling(t).mean()
            f[f"close_ma{t}"] = close / ma - 1
            f[f"ma{t}_slope"] = ma.pct_change(6)
        f["zs_low_dist"] = (close - df["chan_zs_low"]) / close
        f["zs_high_dist"] = (df["chan_zs_high"] - close) / close
        f["rvol"] = vol / vol.rolling(24).mean()
        f["vol_trend"] = vol.rolling(6).mean() / vol.rolling(48).mean()
        # 注: 曾实验性加入 qlib Alpha158 的 BETA/RSQR/RESI/CORR 等14因子,
        # 实测 AUC 0.655->0.570(61维对1203样本=维度灾难), 已回滚。
        # 结论: 因子按"留出集AUC增量"准入, 不按数量堆叠。
        # 市场锚特征(锚币自身用本币计算, 保持列一致)
        if metadata["pair"] == self.market_anchor_pair:
            f["btc_ret24"] = close.pct_change(24)
            f["btc_atr"] = atr20 / close
        else:
            btc = self.dp.get_pair_dataframe(self.market_anchor_pair, self.timeframe)
            if btc is not None and len(btc) > 30:
                bc = btc["close"]
                btc_ret = bc / bc.shift(24) - 1
                btr = pd.concat(
                    [btc["high"] - btc["low"],
                     (btc["high"] - btc["close"].shift()).abs(),
                     (btc["low"] - btc["close"].shift()).abs()], axis=1,
                ).max(axis=1).rolling(20).mean()
                btc_feat = pd.DataFrame({
                    "date": btc["date"],   # 保持 tz-aware, 勿用 .values
                    "btc_ret24": btc_ret.values,
                    "btc_atr": (btr / bc).values,
                })
                d2 = df.drop(columns=["btc_ret24", "btc_atr"], errors="ignore").merge(
                    btc_feat, on="date", how="left"
                )
                f["btc_ret24"] = d2["btc_ret24"].ffill().values
                f["btc_atr"] = d2["btc_atr"].ffill().values
        return f

    def _dump_ml_rows(self, df: DataFrame, metadata: dict) -> None:
        """信号行 + 全量因子 + 双标签(固定/ATR自适应屏障)落盘, 前视仅用于离线标注"""
        import numpy as np
        f = self._ml_derived(df, metadata)
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        atrs = (f["atr20"] * df["close"]).to_numpy()
        dates = df["date"].tolist()
        n = len(df)
        majors = {
            "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
            "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT",
            "AVAX/USDT:USDT", "LINK/USDT:USDT", "LTC/USDT:USDT",
        }
        feat_cols = list(f.columns)
        rows = []
        for i in range(n):
            for side, flag in (("long", df["chan_bsp_buy"].iat[i]),
                               ("short", df["chan_bsp_sshort"].iat[i])):
                if flag != 1:
                    continue
                entry = closes[i]
                a = atrs[i] if atrs[i] == atrs[i] else entry * 0.02
                tp_a = max(0.015, 1.5 * a / entry)
                sl_a = max(0.010, 1.0 * a / entry)
                j_end = min(i + 1 + self.ML_HORIZON, n)
                lab_fixed = lab_atr = 0
                unf, una = True, True
                for j in range(i + 1, j_end):
                    if unf:
                        if side == "long":
                            if lows[j] / entry - 1 <= self.ML_SL:
                                lab_fixed, unf = 0, False
                            elif highs[j] / entry - 1 >= self.ML_TP:
                                lab_fixed, unf = 1, False
                        else:
                            if highs[j] / entry - 1 >= -self.ML_SL:
                                lab_fixed, unf = 0, False
                            elif lows[j] / entry - 1 <= -self.ML_TP:
                                lab_fixed, unf = 1, False
                    if una:
                        if side == "long":
                            if lows[j] / entry - 1 <= -sl_a:
                                lab_atr, una = 0, False
                            elif highs[j] / entry - 1 >= tp_a:
                                lab_atr, una = 1, False
                        else:
                            if highs[j] / entry - 1 >= sl_a:
                                lab_atr, una = 0, False
                            elif lows[j] / entry - 1 <= -tp_a:
                                lab_atr, una = 1, False
                if unf:
                    lab_fixed = int((closes[j_end - 1] / entry - 1) * (1 if side == "long" else -1) > 0)
                if una:
                    lab_atr = int((closes[j_end - 1] / entry - 1) * (1 if side == "long" else -1) > 0)
                row = {
                    "pair": metadata["pair"], "date": dates[i], "side": side,
                    "tag": (df["chan_bsp_buy_tag"].iat[i] if side == "long"
                            else df["chan_bsp_sshort_tag"].iat[i]),
                    "hour": dates[i].hour, "is_major": int(metadata["pair"] in majors),
                    "label_fixed": lab_fixed, "label_atr": lab_atr,
                    "risk_dist": float((entry - df["chan_stop_ref"].iat[i]) / entry)
                    if side == "long"
                    else float((df["chan_stop_ref_s"].iat[i] - entry) / entry),
                }
                for c in self.ML_FEATURES:
                    row[c] = df[c].iat[i]
                for c in feat_cols:
                    row[c] = f[c].iat[i]
                rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(
                "/freqtrade/user_data/ml_signals.csv",
                mode="a", header=not os.path.exists("/freqtrade/user_data/ml_signals.csv"),
                index=False,
            )

    def _add_btc_regime(self, df: DataFrame, metadata: dict) -> DataFrame:
        """BTC 同周期(已收盘)K线的 MA 排列 -> 市场级趋势列, 逐行因果合并"""
        if metadata["pair"] == self.market_anchor_pair:
            df["smc_btc_trend"] = df["chan_ma_trend"]
            return df
        btc = self.dp.get_pair_dataframe(self.market_anchor_pair, self.timeframe)
        if btc is None or len(btc) < 60:
            df["smc_btc_trend"] = 0
            return df
        ma20 = btc["close"].rolling(20).mean()
        ma60 = btc["close"].rolling(60).mean()
        trend = np.where(
            btc["close"] > ma20, 1, np.where(btc["close"] < ma20, -1, 0)
        )
        btc_regime = pd.DataFrame({"date": btc["date"], "smc_btc_trend": trend})
        df = df.drop(columns=["smc_btc_trend"], errors="ignore").merge(
            btc_regime, on="date", how="left"
        )
        df["smc_btc_trend"] = df["smc_btc_trend"].ffill().fillna(0)
        return df

    def _add_smc_columns(self, df: DataFrame) -> DataFrame:
        """在缠论列之上追加 SMC 列。

        所有列只依赖当前行及之前的K线:
        - smc_sweep_bull/bear:   本K线发生猎杀并收回
        - smc_sweep_bull_recent: 近 sweep_lookback 根内出现过(评分用)
        - smc_sweep_ext_low/high: 猎杀窗口内的极值(止损参考)
        - smc_fvg_bull/bear:     近窗口内存在未回补的同向FVG且价格在其外侧
        - smc_killzone:          当前小时处于资金时段
        """
        # 1) 流动性猎杀: 池 = 前一K线时刻已确认的笔极值(严格因果)
        prev_pool_low = df["chan_stop_ref"].shift(1)
        prev_pool_high = df["chan_stop_ref_s"].shift(1)
        sweep_bull = (
            (df["low"] < prev_pool_low) & (df["close"] > prev_pool_low)
        ).fillna(False)
        sweep_bear = (
            (df["high"] > prev_pool_high) & (df["close"] < prev_pool_high)
        ).fillna(False)
        df["smc_sweep_bull"] = sweep_bull.astype(int)
        df["smc_sweep_bear"] = sweep_bear.astype(int)
        df["smc_sweep_bull_recent"] = (
            df["smc_sweep_bull"].rolling(self.sweep_lookback).max().fillna(0)
        )
        df["smc_sweep_bear_recent"] = (
            df["smc_sweep_bear"].rolling(self.sweep_lookback).max().fillna(0)
        )
        # 猎杀极值: 猎杀有效窗口内的最低/最高价(止损放到它的外侧)
        df["smc_sweep_ext_low"] = np.where(
            df["smc_sweep_bull_recent"] == 1,
            df["low"].rolling(self.sweep_stop_window).min(),
            np.nan,
        )
        df["smc_sweep_ext_high"] = np.where(
            df["smc_sweep_bear_recent"] == 1,
            df["high"].rolling(self.sweep_stop_window).max(),
            np.nan,
        )

        # 2) FVG: 三根K线(i-2,i-1,i)在 i 收盘时缺口可知; 半位回补即失效
        df["smc_fvg_bull"] = self._active_fvg(df, bullish=True)
        df["smc_fvg_bear"] = self._active_fvg(df, bullish=False)

        # 3) Killzone(UTC 小时)
        df["smc_killzone"] = (
            df["date"].dt.hour.isin(self.killzone_hours).astype(int)
        )
        return df

    def _active_fvg(self, df: DataFrame, bullish: bool) -> pd.Series:
        """逐根维护活跃FVG集合, 返回每行是否存在未回补的同向缺口。

        O(n) 单遍扫描: 缺口在第三根K线收盘时登记; 之后收盘穿越半位则剔除;
        行值 = 存在活跃缺口 且 收盘价仍站在缺口外侧(磁吸尚未满足)。
        """
        n = len(df)
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        open_ = df["open"].to_numpy()
        res = np.zeros(n, dtype=np.int8)
        active: list[tuple[float, float]] = []  # (top, mid)

        for i in range(2, n):
            # 登记: candles (i-2, i-1, i) 组成的缺口此刻才完全可知
            if bullish:
                if high[i - 2] < low[i] and close[i - 1] > open_[i - 1]:
                    active.append((low[i], (low[i] + high[i - 2]) / 2))
            else:
                if low[i - 2] > high[i] and close[i - 1] < open_[i - 1]:
                    active.append((high[i], (high[i] + low[i - 2]) / 2))
            # 剔除: 收盘穿越半位 = 回补失效
            if active:
                c = close[i]
                if bullish:
                    active = [g for g in active if c > g[1]]
                    if any(c > g[0] for g in active):
                        res[i] = 1
                else:
                    active = [g for g in active if c < g[1]]
                    if any(c < g[0] for g in active):
                        res[i] = 1
        return pd.Series(res, index=df.index)

    # ------------------------------------------------------------------
    # 信号: 缠论事件 × SMC 共振分
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bull_score = (
            dataframe["smc_sweep_bull_recent"] * 2
            + dataframe["smc_fvg_bull"]
            + dataframe["smc_killzone"]
        )
        bear_score = (
            dataframe["smc_sweep_bear_recent"] * 2
            + dataframe["smc_fvg_bear"]
            + dataframe["smc_killzone"]
        )
        # 融合规则(三轮实测收敛):
        # - 空头趋势通道(3a/3b): 需 SMC 流动性共振 —— 空头3a +8.1->+11.7,
        #   胜率 38.8->44.9, 且止损流 -71.9->-51.9(过滤掉高风险单)
        # - 背驰通道(1p): 自带聪明钱衰竭证据, 直通(门控反而+11.7->+1.2)
        # - 多头趋势通道: 门控后 +1.3->-0.8, 维持缠论裸信号
        sell_trend = dataframe["chan_bsp_sshort_tag"].str.contains("3a|3b", na=False)
        long_ok = dataframe["chan_bsp_buy"] == 1
        short_ok = (dataframe["chan_bsp_sshort"] == 1) & (
            (~sell_trend) | (bear_score >= self.smc_score_required)
        )
        dataframe["enter_long"] = long_ok.astype(int)
        dataframe.loc[long_ok, "enter_tag"] = "smc_" + dataframe["chan_bsp_buy_tag"]
        dataframe["enter_short"] = short_ok.astype(int)
        dataframe.loc[short_ok, "enter_tag"] = "smc_" + dataframe[
            "chan_bsp_sshort_tag"
        ]
        # BTC 市场级闸门(选择性): 实测全通道门控 +26.4->+21.5, 通道分化明确 ——
        # β依赖型通道需要市场确认(空头3a: 单笔0.86->2.27胜率54%; 多头1p: 单笔×4),
        # 特质型通道反而被误杀(空头1p: +7.5->-0.2, 山寨独立走弱多发生于BTC中性期)
        if "smc_btc_trend" in dataframe.columns:
            btc_t = dataframe["smc_btc_trend"]
            long_is_1p = dataframe["chan_bsp_buy_tag"].str.contains("1p", na=False)
            short_is_3a = dataframe["chan_bsp_sshort_tag"].str.contains("3a", na=False)
            dataframe["enter_long"] = (
                dataframe["enter_long"] & ((btc_t >= 0) | ~long_is_1p)
            ).astype(int)
            dataframe["enter_short"] = (
                dataframe["enter_short"] & ((btc_t <= 0) | ~short_is_3a)
            ).astype(int)
        # ML 质量门控(仅多头): 模型概率达标才放行
        if self.enable_ml_gate and "ml_proba" in dataframe.columns:
            ml_ok = dataframe["ml_proba"] >= self.ml_proba_threshold
            dataframe["enter_long"] = (
                dataframe["enter_long"] & ml_ok
            ).astype(int)
            dataframe.loc[dataframe["enter_long"] == 1, "enter_tag"] = (
                "ml_" + dataframe["chan_bsp_buy_tag"]
            )
        return dataframe

    # ------------------------------------------------------------------
    # 1R 风险度量与动态保本
    # ------------------------------------------------------------------
    def _risk_ratio(self, pair: str, trade: Trade) -> Optional[float]:
        """R = |开仓价 - 初始结构止损| / 开仓价(入场时刻确定, 已缓存)"""
        try:
            stop = (
                self._entry_stop_price_short(pair, trade) if trade.is_short
                else self._entry_stop_price(pair, trade)
            )
        except Exception:
            return None
        if stop is None or trade.open_rate <= 0:
            return None
        return abs(trade.open_rate - stop) / trade.open_rate

    def _be_threshold(self, pair: str, trade: Trade) -> float:
        """保本触发阈值: 1R 动态(下限 min_r_ratio), 无结构位时退回固定值"""
        if self.be_at_1r:
            r = self._risk_ratio(pair, trade)
            if r is not None:
                return max(r, self.min_r_ratio)
        return super()._be_threshold(pair, trade)

    def adjust_trade_position(
        self, trade: Trade, current_time, current_rate: float, current_profit: float,
        min_stake, max_stake, **kwargs
    ):
        """+1R 先落袋一半; 只触发一次, 剩余仓位由结构离场+保本跟踪接管"""
        if not self.enable_1r_partial or trade.has_open_orders:
            return None
        if trade.nr_of_successful_exits != 0:
            return None
        if current_profit < self._be_threshold(trade.pair, trade):
            return None
        stake = trade.stake_amount * self.partial_fraction
        if stake <= 0 or (min_stake is not None and stake < min_stake):
            return None
        return -stake, "tp_1r"

    # ------------------------------------------------------------------
    # 止损: 猎杀单的初始止损 = 猎杀极值外侧(流动性池之外)
    # ------------------------------------------------------------------
    def _special_entry_stop(
        self, pair: str, trade: Trade, row: pd.Series
    ) -> Optional[float]:
        tag = trade.enter_tag or ""
        if self.use_sweep_stop and tag.startswith("smc_"):
            if trade.is_short:
                ext = row.get("smc_sweep_ext_high")
                if ext is not None and not pd.isna(ext) and ext > 0:
                    stop = float(ext) * (1 + self.min_stop_buffer)
                    if stop > trade.open_rate:
                        return stop
            else:
                ext = row.get("smc_sweep_ext_low")
                if ext is not None and not pd.isna(ext) and ext > 0:
                    stop = float(ext) * (1 - self.min_stop_buffer)
                    if stop < trade.open_rate:
                        return stop
        return super()._special_entry_stop(pair, trade, row)
