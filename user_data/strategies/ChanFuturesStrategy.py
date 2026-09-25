"""ChanFuturesStrategy — 缠论结构 双向(多+空)期货策略

在 ChanStructStrategy(现货多头版)基础上的镜像扩展:

- 做多: 与现货版完全一致(1/1p 背驰买点 + 底分型确认 + segseg!=-1 闸门)
- 做空: 镜像逻辑 —— 1/1p 背驰卖点 + 顶分型确认 + segseg!=+1 闸门(高级别
  确认上升趋势时不做空, 与"不接飞刀"对称的"不摸天花板")
- 平空: 背驰买点信号; 平多: 背驰卖点信号(空头入场信号与多头离场信号
  共用同一个"强卖点"事件, 方向闸门各自独立)

止损(空头三级, 与多头镜像):
- 入场期: 入场上笔高点 + max(0.8%, 0.5*笔振幅) —— 卖点结构失效位
- 持仓期: 开仓价下方出现中枢后, 切换为 中枢上沿 + 1% (升破中枢=最近
  中枢被破坏); 否则跟踪最近上笔高点 + 窄垫
- 兜底: stoploss=-0.20 硬上限 + 冷却期保护

杠杆固定 1x(缠论结构策略不靠杠杆放大, 靠方向双倍化样本)。
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_absolute

from ChanStructStrategy import ChanStructStrategy
from Common.CEnum import FX_TYPE, TREND_TYPE


class ChanFuturesStrategy(ChanStructStrategy):
    INTERFACE_VERSION = 3
    can_short = True

    # v5实验: bi_fx_check=loss -> 信号量×3但质量崩塌(478笔硬止损), 已回退strict
    # v7实验: bi_strict=False -> 计算时间9->85分钟且多头更差, 已回退strict
    # 保留社区验证有效的其余参数组合:
    #   gap_as_kl + bsp1_only_multibi_zs + macd_algo=peak + divergence_rate=inf
    #   (inf=计算全部候选, 提纯由策略层 max_div_1/1p 完成; 启用RSI/KDJ/均线
    #    等指标为后续ML特征做准备)
    chan_config = {
        **ChanStructStrategy.chan_config,
        "bsp3_peak": True,
        "gap_as_kl": True,
        "bsp1_only_multibi_zs": True,
        "macd_algo": "peak",
        "divergence_rate": float("inf"),
        # v9: 2/2s 类趋势阶梯 —— 2买回撤上限(黄金分割), 2s必须跟随2, 阶梯最多2级
        "max_bs2_rate": 0.618,
        "bsp2s_follow_2": True,
        "max_bsp2s_lv": 2,
        "cal_rsi": True,
        "cal_kdj": True,
        "mean_metrics": [5, 10, 20, 60],
        "trend_metrics": [5, 10, 20, 60],
    }

    # ---- 信号参数 ----
    short_bsp_types = {"1", "1p"}    # 卖点事件类型(平空/平多判定用)
    buy_bsp_types = {"1", "1p"}      # 买点事件类型
    # 入场通道: 仅 1p(盘整背驰) + 3a/3b(趋势延续, 独立闸门) + 2/2s(趋势阶梯)
    # 纯"1"趋势背驰两侧皆亏(v6: chan_1 -7.8%, chanS_1 -12.9%), 但作为
    # 平仓事件极优(cov_1 平均+4~6%, 100%胜率) -> 只出场不进场
    entry_bsp_types = {"1p"}
    # 3a/3b 趋势延续买卖点: 仅当高阶结构同向确认时放行
    enable_trend_bsp = True
    # 2s 阶梯加仓需 MA 趋势因子同向确认(MA20/60排列 + 价格位置)
    enable_2s = True
    max_bs2_retrace = 0.618      # 策略层再过滤一次回撤比例
    # 多头趋势跟踪: 前低结构阶梯垫(上一个回调笔低点下方)
    hl_trail_min_buffer = 0.005
    # ---- 部署开关 -----------------------------------------------------
    # v8(默认, 双半期为正最稳健): 多头趋势入场开 + 多头贴身跟踪 + 无空头阶梯
    # v12(全期最高+23.1%但后半期-4.8%): 三项全关/开如下注释
    #   v12 = enable_long_trend_entry=False, long_tight_trail=False,
    #         enable_short_ladder=True
    enable_long_trend_entry = True
    long_tight_trail = True
    enable_short_ladder = False
    enable_trend_3b = False
    # 线段级卖点足够"新鲜"(距最新K线不超过N根合并K)才触发离场
    segbsp_fresh_klc = 8
    # 多单笔级止盈: 浮盈超过该阈值后遇强背驰卖点落袋
    bi_tp_trigger = 0.04
    # 结构止损距离闸门: 入场笔振幅过大 -> 结构止损位远过硬止损上限, 放弃该信号
    max_stop_dist = 0.08

    # chan 状态列: 在基类基础上追加空头侧
    _CHAN_COLS = {
        **ChanStructStrategy._CHAN_COLS,
        "chan_bsp_sshort": 0,
        "chan_bsp_sshort_tag": "",
        "chan_bsp_cshort": 0,
        "chan_bsp_cshort_tag": "",
        "chan_stop_ref_s": float("nan"),
        "chan_stop_amp_s": float("nan"),
        "chan_entry_stop_s": float("nan"),
        "chan_segbsp_buy": 0,
        "chan_segbsp_sell": 0,
        "chan_stop_ref2": float("nan"),
        "chan_stop_amp2": float("nan"),
        "chan_stop_ref2_s": float("nan"),
        "chan_stop_amp2_s": float("nan"),
        "chan_ma_trend": 0,
    }

    plot_config = {
        "main_plot": {
            "chan_stop_ref": {"color": "orange"},
            "chan_stop_ref_s": {"color": "purple"},
        },
        "subplots": {
            "Chan": {
                "chan_bsp_buy": {"color": "green"},
                "chan_bsp_sell": {"color": "red"},
                "chan_bsp_sshort": {"color": "darkred"},
                "chan_bsp_cshort": {"color": "lightgreen"},
            },
        },
    }

    # ------------------------------------------------------------------
    # 重写快照: 双向信号(多头逻辑与基类一致, 追加空头镜像)
    # ------------------------------------------------------------------
    def _snapshot(self, state) -> dict:
        out = {k: v for k, v in self._CHAN_COLS.items()}
        kl = state.chan[0]
        bi_list = kl.bi_list
        klc_lst = kl.lst

        out["chan_bi_cnt"] = len(bi_list)
        if len(kl.seg_list) > 0:
            out["chan_seg_dir"] = 1 if kl.seg_list[-1].is_up() else -1
        for s in reversed(kl.segseg_list):
            if s.is_sure:
                out["chan_segseg_dir"] = 1 if s.is_up() else -1
                break
        if len(kl.zs_list) > 0:
            zs = kl.zs_list[-1]
            out["chan_zs_low"] = float(zs.low)
            out["chan_zs_high"] = float(zs.high)
        # 多头结构参考: 最近下笔低点 + 上一个下笔低点(2买失效规则的前低)
        down_cnt = 0
        for bi in reversed(bi_list):
            if bi.is_down():
                down_cnt += 1
                low = float(bi._low())
                amp = abs(float(bi.get_begin_val()) - float(bi.get_end_val())) / max(low, 1e-12)
                if down_cnt == 1:
                    out["chan_stop_ref"] = low
                    out["chan_stop_amp"] = amp
                else:
                    out["chan_stop_ref2"] = low
                    out["chan_stop_amp2"] = amp
                    break
        # 空头镜像: 最近上笔高点 + 上一个上笔高点
        up_cnt = 0
        for bi in reversed(bi_list):
            if bi.is_up():
                up_cnt += 1
                high = float(bi._high())
                amp = abs(float(bi.get_begin_val()) - float(bi.get_end_val())) / max(high, 1e-12)
                if up_cnt == 1:
                    out["chan_stop_ref_s"] = high
                    out["chan_stop_amp_s"] = amp
                else:
                    out["chan_stop_ref2_s"] = high
                    out["chan_stop_amp2_s"] = amp
                    break
        # MA 趋势因子: MA20/60 排列 + 价格位置(+1多头/-1空头/0混沌)
        last_klu = kl.lst[-1][-1]
        ma_dict = getattr(last_klu, "trend", {}).get(TREND_TYPE.MEAN, {})
        ma20, ma60 = ma_dict.get(20), ma_dict.get(60)
        if ma20 is not None and ma60 is not None:
            if ma20 > ma60 and last_klu.close > ma20:
                out["chan_ma_trend"] = 1
            elif ma20 < ma60 and last_klu.close < ma20:
                out["chan_ma_trend"] = -1

        if len(bi_list) == 0 or len(klc_lst) < 2:
            return out
        flat = kl.bs_point_lst.bsp_store_flat_dict
        if not flat:
            return out
        bsp = flat[max(flat.keys())]
        out["chan_bsp_last"] = ("b" if bsp.is_buy else "s") + bsp.type2str()
        div = dict(bsp.features.items()).get("divergence_rate")
        if div is not None:
            out["chan_bsp_div"] = float(div)

        n = len(klc_lst)
        if bsp.klu.klc.idx != n - 2:
            return out
        if len(bi_list) < self.min_bi_cnt:
            return out
        if self.require_sure_seg and not any(s.is_sure for s in kl.seg_list):
            return out

        types = {t.value for t in bsp.type}
        div_v = out["chan_bsp_div"]
        segseg_dir = out["chan_segseg_dir"]
        feat = dict(bsp.features.items())
        # 通道A 强背驰反转: 1类强背驰 / 1p盘整背驰(略宽)
        strong = (
            ("1" in types and div_v <= self.max_div_1)
            or ("1p" in types and "1" not in types and div_v <= self.max_div_1p)
        )
        # 通道B 趋势延续: 3a(/3b) 买卖点, 且高阶结构同向确认
        trend_buy = self.enable_trend_bsp and (
            "3a" in types or (self.enable_trend_3b and "3b" in types)
        ) and segseg_dir == 1
        trend_short = self.enable_trend_bsp and (
            "3a" in types or (self.enable_trend_3b and "3b" in types)
        ) and segseg_dir == -1
        # 通道C 趋势阶梯: 2(回调不破前低) / 2s(阶梯加仓, 需MA趋势因子同向)
        # v9/v10实验: 多头2/2s两种门槛(segseg!=-1 / ==1)均出血(V顶反转最狠),
        #             彻底关闭; 空头全通道为正, 保留
        enable_long_ladder = False
        two_buy = (
            enable_long_ladder and "2" in types and segseg_dir == 1
            and feat.get("bsp2_retrace_rate", float("inf")) <= self.max_bs2_retrace
        )
        two_short = (
            self.enable_short_ladder and "2" in types and segseg_dir != 1
            and feat.get("bsp2_retrace_rate", float("inf")) <= self.max_bs2_retrace
        )
        twos_buy = (
            enable_long_ladder and self.enable_2s and "2s" in types and segseg_dir == 1
            and out["chan_ma_trend"] == 1
            and feat.get("bsp2s_retrace_rate", float("inf")) <= self.max_bs2_retrace
        )
        twos_short = (
            self.enable_short_ladder and self.enable_2s and "2s" in types
            and segseg_dir != 1
            and out["chan_ma_trend"] == -1
            and feat.get("bsp2s_retrace_rate", float("inf")) <= self.max_bs2_retrace
        )

        # ---- 线段级买卖点: "笔进场, 段离场"(过滤笔级噪音, 拿足趋势) ----
        seg_flat = kl.seg_bs_point_lst.bsp_store_flat_dict
        if seg_flat:
            sbsp = seg_flat[max(seg_flat)]
            last_klu_idx = kl.lst[-1][-1].idx
            if sbsp.klu.idx >= last_klu_idx - self.segbsp_fresh_klc:
                stypes = {t.value for t in sbsp.type}
                if stypes & {"1", "1p"}:
                    if sbsp.is_buy:
                        out["chan_segbsp_buy"] = 1   # 段级买点: 平空(兜底)
                    else:
                        out["chan_segbsp_sell"] = 1  # 段级卖点: 平多

        if bsp.is_buy and klc_lst[-2].fx == FX_TYPE.BOTTOM and (
            strong or trend_buy or two_buy or twos_buy
        ):
            # 买点事件: 平空信号
            out["chan_bsp_cshort"] = 1
            out["chan_bsp_cshort_tag"] = f"cov_{bsp.type2str()}"
            # v12实验: 多头3a/3b趋势入场三版皆负(v9/-1.6 v10/-1.6 v11/-2.95)
            #      且消耗仓位容量与止损额度 -> 可开关(默认v8开启);
            #      买点事件保留用于平空。多头仅保留1p盘整背驰通道
            gate_long = (
                (self.enable_long_trend_entry and trend_buy)
                or (strong and types & self.entry_bsp_types and segseg_dir != -1)
            )
            # 2/2s 类的初始止损用前低(2买跌破前低即失效); 其余用最近下笔低点
            if two_buy or twos_buy:
                ref, amp = out["chan_stop_ref2"], out["chan_stop_amp2"]
                if not (ref == ref and amp == amp):
                    ref, amp = out["chan_stop_ref"], out["chan_stop_amp"]
            else:
                ref, amp = out["chan_stop_ref"], out["chan_stop_amp"]
            close = float(klc_lst[-1][-1].close)
            if gate_long and ref == ref and amp == amp:
                buf = max(self.min_stop_buffer, self.entry_stop_amp_k * amp)
                estop = ref * (1 - buf)
                if estop / close > 1 - self.max_stop_dist:
                    out["chan_bsp_buy"] = 1
                    out["chan_bsp_buy_tag"] = f"chan_{bsp.type2str()}"
                    out["chan_bsp_amp"] = amp
                    out["chan_entry_stop"] = estop
        elif (
            (not bsp.is_buy)
            and klc_lst[-2].fx == FX_TYPE.TOP
            and (strong or trend_short or two_short or twos_short)
        ):
            # 强卖点事件: 供 custom_exit 多单止盈使用(浮盈达标时落袋)
            if strong:
                out["chan_bsp_sell"] = 1
                out["chan_bsp_sell_tag"] = f"chan_{bsp.type2str()}"
            gate_short = (
                (strong and types & self.entry_bsp_types and segseg_dir != 1)
                or trend_short or two_short or twos_short
            )
            if two_short or twos_short:
                ref, amp = out["chan_stop_ref2_s"], out["chan_stop_amp2_s"]
                if not (ref == ref and amp == amp):
                    ref, amp = out["chan_stop_ref_s"], out["chan_stop_amp_s"]
            else:
                ref, amp = out["chan_stop_ref_s"], out["chan_stop_amp_s"]
            close = float(klc_lst[-1][-1].close)
            if gate_short and ref == ref and amp == amp:
                buf = max(self.min_stop_buffer, self.entry_stop_amp_k * amp)
                estop = ref * (1 + buf)
                if estop / close - 1 < self.max_stop_dist:
                    out["chan_bsp_sshort"] = 1
                    out["chan_bsp_sshort_tag"] = f"chanS_{bsp.type2str()}"
                    out["chan_entry_stop_s"] = estop
        return out

    # ------------------------------------------------------------------
    # 双向信号挂接
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = dataframe["chan_bsp_buy"]
        dataframe.loc[dataframe["chan_bsp_buy"] == 1, "enter_tag"] = dataframe[
            "chan_bsp_buy_tag"
        ]
        dataframe["enter_short"] = dataframe["chan_bsp_sshort"]
        dataframe.loc[dataframe["chan_bsp_sshort"] == 1, "enter_tag"] = dataframe[
            "chan_bsp_sshort_tag"
        ]
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 笔进场, 段离场: 多头拿到线段级背驰卖点才离场(结构止损兜底);
        # 空头更快平仓(加密市场上行漂移): 笔级强买点或段级买点任一触发
        dataframe["exit_long"] = dataframe["chan_segbsp_sell"]
        dataframe.loc[dataframe["chan_segbsp_sell"] == 1, "exit_tag"] = "seg_sell"
        dataframe["exit_short"] = (
            (dataframe["chan_bsp_cshort"] == 1) | (dataframe["chan_segbsp_buy"] == 1)
        ).astype(int)
        dataframe.loc[dataframe["chan_bsp_cshort"] == 1, "exit_tag"] = dataframe[
            "chan_bsp_cshort_tag"
        ]
        dataframe.loc[
            (dataframe["chan_segbsp_buy"] == 1) & (dataframe["chan_bsp_cshort"] == 0),
            "exit_tag",
        ] = "seg_cover"
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return 1.0

    # ------------------------------------------------------------------
    # 多单趋势止盈: 浮盈超过阈值后, 出现笔级强背驰卖点即落袋
    # (段级卖点太稀少, 笔级卖点+浮盈门槛兼顾及时性与趋势持有)
    # ------------------------------------------------------------------
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        if trade.is_short or current_profit < self.bi_tp_trigger:
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None
        if dataframe.iloc[-1].get("chan_bsp_sell") == 1:
            return "bi_sell_tp"
        return None

    # ------------------------------------------------------------------
    # 空头结构止损(镜像); 多头沿用基类
    # ------------------------------------------------------------------
    def _special_entry_stop(self, pair: str, trade: Trade, row) -> Optional[float]:
        """按类型的教科书初始止损:
        - 3a/3b: 中枢边界(3买论点=回抽不回中枢, 跌回即失效)
        - 2/2s: 前低/前高(2买论点=回调不破前低, 破即失效)
        """
        tag = trade.enter_tag or ""
        if "3a" in tag or "3b" in tag:
            if trade.is_short:
                zs_low = row.get("chan_zs_low")
                if zs_low is not None and not pd.isna(zs_low) and zs_low > 0:
                    stop = float(zs_low) * (1 + self.min_stop_buffer)
                    return stop if stop > trade.open_rate else None
            else:
                zs_high = row.get("chan_zs_high")
                if zs_high is not None and not pd.isna(zs_high) and zs_high > 0:
                    stop = float(zs_high) * (1 - self.min_stop_buffer)
                    return stop if stop < trade.open_rate else None
            return None
        if "_2" in tag or "2s" in tag:
            if trade.is_short:
                ref2 = row.get("chan_stop_ref2_s")
                if ref2 is not None and not pd.isna(ref2) and ref2 > 0:
                    stop = float(ref2) * (1 + self.min_stop_buffer)
                    return stop if stop > trade.open_rate else None
            else:
                ref2 = row.get("chan_stop_ref2")
                if ref2 is not None and not pd.isna(ref2) and ref2 > 0:
                    stop = float(ref2) * (1 - self.min_stop_buffer)
                    return stop if stop < trade.open_rate else None
        return None

    def _entry_stop_price_short(self, pair: str, trade: Trade) -> Optional[float]:
        """空单入场结构止损: 入场时刻结构位 + 宽垫(一次性缓存)"""
        cached = trade.get_custom_data("chan_entry_stop_s")
        if cached is not None:
            return float(cached)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None
        idx = int(dataframe["date"].searchsorted(trade.open_date_utc, side="right")) - 1
        row = dataframe.iloc[max(idx, 0)]
        price = self._special_entry_stop(pair, trade, row)
        if price is None:
            ref, amp = row.get("chan_stop_ref_s"), row.get("chan_stop_amp_s")
            if ref is None or pd.isna(ref) or ref <= 0:
                return None
            amp = 0.0 if (amp is None or pd.isna(amp)) else float(amp)
            buf = max(self.min_stop_buffer, self.entry_stop_amp_k * amp)
            price = float(ref) * (1 + buf)
        trade.set_custom_data("chan_entry_stop_s", price)
        return price

    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
        current_profit: float, after_fill: bool, **kwargs
    ) -> Optional[float]:
        """多头(趋势持仓): 入场止损 + 保本 + 中枢下沿 + 前低结构阶梯
           (不再贴最后一笔跟踪 —— 趋势中的正常回调应被持有, 破前低才是结构破坏)
           空头(快进快出): 入场止损 + 保本 + 中枢上沿 + 贴身笔高点跟踪"""
        cands: list = []
        has_structure = False
        if trade.is_short:
            entry_stop = self._entry_stop_price_short(pair, trade)
            if entry_stop is not None:
                has_structure = True
                if entry_stop > current_rate:
                    cands.append(entry_stop)
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and len(dataframe) > 0:
                row = dataframe.iloc[-1]
                zs_high = row.get("chan_zs_high")
                if zs_high is not None and not pd.isna(zs_high) and zs_high > 0:
                    has_structure = True
                    if zs_high < trade.open_rate and zs_high > current_rate:
                        cands.append(float(zs_high) * (1 + self.zs_trail_buffer))
                ref, amp = row.get("chan_stop_ref_s"), row.get("chan_stop_amp_s")
                if ref is not None and not pd.isna(ref) and ref > 0:
                    has_structure = True
                    a = 0.0 if (amp is None or pd.isna(amp)) else float(amp)
                    buf = max(self.min_stop_buffer, self.trail_stop_amp_k * a)
                    trail = float(ref) * (1 + buf)
                    if trail > current_rate:
                        cands.append(trail)
                if current_profit > self._be_threshold(pair, trade):
                    be = trade.open_rate * (1 - self.be_lock)
                    if be > current_rate:
                        cands.append(be)
            if cands:
                return stoploss_from_absolute(
                    min(cands), current_rate, is_short=True, leverage=trade.leverage
                )
            if not has_structure:
                return None
            return -0.001

        # ---- 多头 ----
        entry_stop = self._entry_stop_price(pair, trade)
        if entry_stop is not None:
            has_structure = True
            if entry_stop < current_rate:
                cands.append(entry_stop)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is not None and len(dataframe) > 0:
            row = dataframe.iloc[-1]
            # 中枢阶梯: 开仓价上方且仍低于现价的中枢下沿(新中枢抬高时棘轮上移)
            zs_low = row.get("chan_zs_low")
            if zs_low is not None and not pd.isna(zs_low) and zs_low > 0:
                has_structure = True
                if zs_low > trade.open_rate and zs_low < current_rate:
                    cands.append(float(zs_low) * (1 - self.zs_trail_buffer))
            if self.long_tight_trail:
                # v8模式: 贴最近下笔低点(窄垫, 快速保护)
                ref, amp = row.get("chan_stop_ref"), row.get("chan_stop_amp")
                if ref is not None and not pd.isna(ref) and ref > 0:
                    has_structure = True
                    a = 0.0 if (amp is None or pd.isna(amp)) else float(amp)
                    buf = max(self.min_stop_buffer, self.trail_stop_amp_k * a)
                    trail = float(ref) * (1 - buf)
                    if trail < current_rate:
                        cands.append(trail)
            else:
                # v12模式: 前低结构阶梯(上一个回调笔低点, 趋势中正常回调被持有)
                ref2, amp2 = row.get("chan_stop_ref2"), row.get("chan_stop_amp2")
                if ref2 is not None and not pd.isna(ref2) and ref2 > 0:
                    has_structure = True
                    a2 = 0.0 if (amp2 is None or pd.isna(amp2)) else float(amp2)
                    buf = max(self.hl_trail_min_buffer, self.trail_stop_amp_k * a2)
                    trail2 = float(ref2) * (1 - buf)
                    if trail2 < current_rate:
                        cands.append(trail2)
            if current_profit > self._be_threshold(pair, trade):
                be = trade.open_rate * (1 + self.be_lock)
                if be < current_rate:
                    cands.append(be)
        if cands:
            return stoploss_from_absolute(
                max(cands), current_rate, is_short=False, leverage=trade.leverage
            )
        if not has_structure:
            return None
        return -0.001  # 结构已被破坏, 贴身立即离场

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float,
        time_in_force: str, current_time: datetime, entry_tag: Optional[str],
        side: str, **kwargs
    ) -> bool:
        if side == "short":
            # 空头镜像追空保护: 价格已深跌离结构参考位(>3%)则放弃追空
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or len(dataframe) == 0:
                return True
            ref = dataframe.iloc[-1].get("chan_stop_ref_s")
            if ref is not None and not pd.isna(ref) and float(ref) > 0:
                if rate < float(ref) * 0.97:
                    return False
            return True
        return super().confirm_trade_entry(
            pair, order_type, amount, rate, time_in_force, current_time,
            entry_tag, side, **kwargs
        )
