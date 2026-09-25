"""ChanStructStrategy — 基于 chan.py 缠论结构的 freqtrade 策略

架构说明
========
1. 数据流: freqtrade populate_indicators(dataframe) 中的每根K线 -> CKLine_Unit ->
   CChan.trigger_load() 增量喂入 chan.py（trigger_step=True 回放模式）。
   chan.py 核心为纯标准库实现, 无需向镜像添加任何依赖。

2. 无未来函数: 无论回测还是实盘, 均逐根K线投喂, 每喂一根就记录"当前帧"的
   缠论状态(笔/段/中枢/买卖点)到 dataframe 行上。回测列 == 实盘当下可见信息,
   回测与实盘语义严格一致(可用 freqtrade lookahead-analysis 验证)。

3. 多层级: 不使用多时间框架区间套(对齐脆弱), 而是用 chan.py 的结构递归:
   笔 -> 线段 -> 线段的线段(segseg), 以及 笔买卖点 + 段买卖点 双层信号。

4. 开单: 官方 demo 同款入场确认 —— 最新买卖点挂在倒数第二根合并K(分型中间
   元素)上, 且该合并K的分型已确认(底分型买/顶分型卖), 再叠加质量过滤
   (背驰率/结构成熟度/至少出现过一个确定线段)。

5. 止损(缠论结构止损):
   - 硬止损 stoploss=-0.20 为绝对风险上限;
   - custom_stoploss 跟踪 chan_stop_ref = 最近一个向下笔的低点:
     入场时即入场笔低点(一买被跌破即结构失效), 持仓中随新的更高低点抬升
     (freqtrade 止损只升不降的棘轮 == 缠论移动止损)。
   - 离场: 反向买卖点(顶分型确认) 或结构止损, 不设固定 ROI。

6. 实盘窗口: 冷启动受 startup_candle_count 限制(Binance 最大 5*1000-1=4994,
   本策略取 999 保持通用), 此后增量累积, 结构视野只增不减。
   live 模式 dataframe 左端会被裁剪, 通过 time->idx 映射维护增量对齐。
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, stoploss_from_absolute


# ---------------------------------------------------------------------------
# chan.py 源码定位: 仓库自带 user_data/chan/(自包含), 兼容旧部署路径
# ---------------------------------------------------------------------------
def _locate_chansrc() -> str:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "chan",                # 本仓库: user_data/chan
        Path("/freqtrade/chanpy"),           # 旧容器挂载路径
    ]
    if len(here.parents) > 3:                # 旧开发布局: ../chan.py
        candidates.append(here.parents[3] / "chan.py")
    for c in candidates:
        if (c / "Chan.py").is_file():
            return str(c)
    raise RuntimeError(
        "chan.py source not found; expected at user_data/chan "
        "(vendored in this repository)"
    )


_CHANSRC = _locate_chansrc()
if _CHANSRC not in sys.path:
    sys.path.insert(0, _CHANSRC)

from Chan import CChan  # noqa: E402
from ChanConfig import CChanConfig  # noqa: E402
from Common.CEnum import BSP_TYPE, DATA_FIELD, FX_TYPE, KL_TYPE  # noqa: E402
from Common.CTime import CTime  # noqa: E402
from KLine.KLine_Unit import CKLine_Unit  # noqa: E402

logger = logging.getLogger(__name__)


_TF2KLTYPE = {
    "5m": KL_TYPE.K_5M,
    "15m": KL_TYPE.K_15M,
    "30m": KL_TYPE.K_30M,
    "1h": KL_TYPE.K_60M,
    "1d": KL_TYPE.K_DAY,
}

# chan 状态列(挂到类上, 便于子类扩展空头侧列)
_BASE_CHAN_COLS = {
    "chan_bsp_buy": 0,
    "chan_bsp_buy_tag": "",
    "chan_bsp_sell": 0,
    "chan_bsp_sell_tag": "",
    "chan_bsp_div": float("nan"),
    "chan_bsp_amp": float("nan"),
    "chan_bsp_last": "",
    "chan_bi_cnt": 0,
    "chan_seg_dir": 0,
    "chan_segseg_dir": 0,
    "chan_zs_low": float("nan"),
    "chan_zs_high": float("nan"),
    "chan_stop_ref": float("nan"),
    "chan_stop_amp": float("nan"),
    "chan_entry_stop": float("nan"),
}


class _PairChanState:
    """每个交易对一个常驻 CChan 实例 + 喂入进度"""

    def __init__(self, chan: CChan, lv: KL_TYPE):
        self.chan = chan
        self.lv = lv
        self.fed_idx = 0          # 已喂入的 klu 总数(klu.idx 从 0 递增)
        self.time2idx: dict[str, int] = {}  # 时间串 -> chan klu idx
        self.last_ts: Optional[pd.Timestamp] = None
        self.broken = False


class ChanStructStrategy(IStrategy):
    INTERFACE_VERSION = 3

    _CHAN_COLS = _BASE_CHAN_COLS

    timeframe = "1h"
    can_short = False
    startup_candle_count = 999      # Binance 允许最大 5*ohlcv_limit-1; 之后增量累积
    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # 结构性离场为主, 固定 ROI 实际关闭(100 = 10000%)
    minimal_roi = {"0": 100}
    # 硬性最大风险; 实际止损由结构止损动态给出
    stoploss = -0.20
    use_custom_stoploss = True
    trailing_stop = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    plot_config = {
        "main_plot": {
            "chan_stop_ref": {"color": "orange"},
            "chan_zs_low": {"color": "grey"},
            "chan_zs_high": {"color": "grey"},
        },
        "subplots": {
            "Chan": {
                "chan_bsp_buy": {"color": "green"},
                "chan_bsp_sell": {"color": "red"},
                "chan_seg_dir": {"color": "blue"},
            },
        },
    }

    # ------------------------------------------------------------------
    # 缠论计算配置(含义参见 chan.py/README.md CChanConfig 一节)
    # ------------------------------------------------------------------
    chan_config = {
        "trigger_step": True,       # 回放模式: trigger_load 增量喂入
        "bi_strict": True,          # 严格笔(顶底间>=4根合并K)
        "bi_fx_check": "strict",    # 分型有效性最严校验
        "bi_end_is_peak": True,
        "bi_allow_sub_peak": True,
        "seg_algo": "chan",         # 特征序列划段(默认且唯一维护)
        "left_seg_method": "peak",
        "zs_algo": "normal",        # 段内中枢(经典)
        "zs_combine": True,
        "zs_combine_mode": "zs",
        # ---- 买卖点 ----
        "divergence_rate": 0.9,     # 1类买卖点背驰比例(出/入中枢笔指标比)
        "min_zs_cnt": 1,            # 1类买卖点至少经历中枢数
        "bs1_peak": True,           # 1类买卖点须为中枢极值
        "max_bs2_rate": 0.618,      # 2类买卖点最大回撤比例(黄金分割)
        "macd_algo": "full_area",   # 笔背驰用MACD面积(段买卖点框架自动用slope)
        "bs_type": "1,1p,2,2s,3a,3b",
        "bsp2_follow_1": False,     # 2类不必跟随真实1类(小转大容错)
        "bsp3_follow_1": False,
        "print_warning": False,
        "print_err_time": True,
        "kl_data_check": False,     # 单级别, 无需多级别对齐检查
    }

    # ---- 信号过滤参数(逐轮回测诊断驱动, 见仓库分析报告) ----
    # v1: 2类买点在震荡市被结构止损反复打穿(-5.63%) -> 剔除
    # v3: 3a在趋势未确认时接飞刀(-2.82%) -> 只保留背驰反转买点 1/1p
    # v4: 弱卖点(2/3a/3b)提前掐断趋势单(近乎零收益) -> 只认 1/1p 背驰卖点,
    #     浅回调交给中枢下沿跟踪止损扛(缠论: 回调不破中枢则走势未坏)
    buy_bsp_types = {"1", "1p"}
    sell_bsp_types = {"1", "1p"}
    min_bi_cnt = 10                # 结构成熟度: 至少10笔
    require_sure_seg = True        # 出现过确定线段后才允许交易(quick_guide 建议)
    max_div_1 = 0.75               # 1类买点: 只做强背驰
    max_div_1p = 0.95              # 盘整背驰1类: 背驰要求略宽
    # ---- 结构止损参数: 止损垫自适应于笔的振幅 ----
    min_stop_buffer = 0.008        # 止损垫下限 0.8%(加密1h噪音)
    entry_stop_amp_k = 0.5         # 初始止损垫 = max(下限, k * 入场笔相对振幅)
    trail_stop_amp_k = 0.3         # 笔低点跟踪止损垫系数
    zs_trail_buffer = 0.01         # 中枢下沿跟踪垫 1%
    # 保本止损: 浮盈超过阈值后, 开仓价+锁定比例 进入止损候选
    be_trigger = 0.03
    be_lock = 0.001

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
        ]

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._states: dict[str, _PairChanState] = {}
        if self.timeframe not in _TF2KLTYPE:
            raise ValueError(
                f"ChanStructStrategy supports {sorted(_TF2KLTYPE)}, got {self.timeframe}"
            )
        self._lv = _TF2KLTYPE[self.timeframe]

    # ------------------------------------------------------------------
    # chan.py 实例管理
    # ------------------------------------------------------------------
    def _new_chan(self) -> CChan:
        # trigger_step=True 时构造函数不会拉取数据, data_src 参数不会被使用
        return CChan(
            code="",
            data_src=None,  # type: ignore[arg-type]
            lv_list=[self._lv],
            config=CChanConfig(dict(self.chan_config)),
        )

    @staticmethod
    def _tstr(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%d %H:%M")

    def _make_klu(self, row: pd.Series) -> CKLine_Unit:
        dt = row["date"].to_pydatetime()
        # 分钟/小时级K线必须 auto=False: 加密货币存在 0:00 时间戳,
        # auto=True 会被误判为天级K线导致 "kline time err" (见 quick_guide)
        return CKLine_Unit({
            DATA_FIELD.FIELD_TIME: CTime(
                dt.year, dt.month, dt.day, dt.hour, dt.minute, auto=False
            ),
            DATA_FIELD.FIELD_OPEN: float(row["open"]),
            DATA_FIELD.FIELD_HIGH: float(row["high"]),
            DATA_FIELD.FIELD_LOW: float(row["low"]),
            DATA_FIELD.FIELD_CLOSE: float(row["close"]),
            DATA_FIELD.FIELD_VOLUME: float(row["volume"]),
        })

    # ------------------------------------------------------------------
    # "当前帧"特征提取: 每喂入一根K线后调用, 只使用此刻及之前的信息
    # ------------------------------------------------------------------
    def _snapshot(self, state: _PairChanState) -> dict:
        out = {k: v for k, v in self._CHAN_COLS.items()}
        kl = state.chan[0]
        bi_list = kl.bi_list
        klc_lst = kl.lst

        out["chan_bi_cnt"] = len(bi_list)
        # 线段方向: +1 上升段 / -1 下降段
        if len(kl.seg_list) > 0:
            out["chan_seg_dir"] = 1 if kl.seg_list[-1].is_up() else -1
        # 高阶结构(线段的线段)方向: 取最后一个"确定"的 segseg, 尾部不确定段继承其方向
        for s in reversed(kl.segseg_list):
            if s.is_sure:
                out["chan_segseg_dir"] = 1 if s.is_up() else -1
                break
        # 最近中枢区间
        if len(kl.zs_list) > 0:
            zs = kl.zs_list[-1]
            out["chan_zs_low"] = float(zs.low)
            out["chan_zs_high"] = float(zs.high)
        # 结构止损参考: 最近一个向下笔的低点及其相对振幅
        # (跌破了低点, 最近结构即被破坏; 振幅用于自适应放止损垫)
        for bi in reversed(bi_list):
            if bi.is_down():
                low = float(bi._low())
                out["chan_stop_ref"] = low
                out["chan_stop_amp"] = abs(
                    float(bi.get_begin_val()) - float(bi.get_end_val())
                ) / max(low, 1e-12)
                break

        if len(bi_list) == 0 or len(klc_lst) < 2:
            return out

        # 最新买卖点(按所在笔 idx)
        flat = kl.bs_point_lst.bsp_store_flat_dict
        if not flat:
            return out
        bsp = flat[max(flat.keys())]
        out["chan_bsp_last"] = ("b" if bsp.is_buy else "s") + bsp.type2str()
        # 只有 1/1p 类买卖点携带 divergence_rate 特征
        div = dict(bsp.features.items()).get("divergence_rate")
        if div is not None:
            out["chan_bsp_div"] = float(div)

        # ---- 入场/离场判定: 官方 demo 同款"分型确认"时机 ----
        # 买卖点必须挂在倒数第二根合并K(分型中间元素)上, 且分型已识别
        n = len(klc_lst)
        if bsp.klu.klc.idx != n - 2:
            return out
        if len(bi_list) < self.min_bi_cnt:
            return out
        if self.require_sure_seg and not any(s.is_sure for s in kl.seg_list):
            return out

        types = {t.value for t in bsp.type}
        div = out["chan_bsp_div"]
        if bsp.is_buy and klc_lst[-2].fx == FX_TYPE.BOTTOM:
            if types & self.buy_bsp_types:
                # 级别共振闸门(缠论"不接飞刀"原则):
                # 高阶结构(segseg)确认向下 -> 不做多; 3a(趋势延续)必须确认向上
                segseg_dir = out["chan_segseg_dir"]
                if segseg_dir == -1:
                    return out
                if "3a" in types and "1" not in types and "1p" not in types and segseg_dir != 1:
                    return out
                # 背驰强度分级: 1类只做强背驰, 1p(盘整背驰)略宽
                if "1" in types and div > self.max_div_1:
                    return out
                if "1p" in types and "1" not in types and div > self.max_div_1p:
                    return out
                out["chan_bsp_buy"] = 1
                out["chan_bsp_buy_tag"] = f"chan_{bsp.type2str()}"
                out["chan_bsp_amp"] = out["chan_stop_amp"]
                ref, amp = out["chan_stop_ref"], out["chan_stop_amp"]
                if ref == ref and amp == amp:  # 非 NaN
                    buf = max(self.min_stop_buffer, self.entry_stop_amp_k * amp)
                    out["chan_entry_stop"] = ref * (1 - buf)
        elif (not bsp.is_buy) and klc_lst[-2].fx == FX_TYPE.TOP:
            if types & self.sell_bsp_types:
                out["chan_bsp_sell"] = 1
                out["chan_bsp_sell_tag"] = f"chan_{bsp.type2str()}"
        return out

    # ------------------------------------------------------------------
    # 主入口: 增量喂K + 逐帧记录
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        for col, default in self._CHAN_COLS.items():
            dataframe[col] = default

        state = self._states.get(pair)
        rebuild = state is None or state.broken
        if not rebuild:
            assert state is not None
            head_t = self._tstr(dataframe["date"].iloc[0])
            if state.fed_idx == 0 or head_t not in state.time2idx:
                # 数据起点晚于已喂范围(重启/更换历史) -> 重建
                rebuild = True
            elif dataframe["date"].iloc[-1] <= state.last_ts:
                # 无新K线
                return dataframe

        if rebuild:
            state = _PairChanState(self._new_chan(), self._lv)
            self._states[pair] = state
            offset = 0
            start_row = 0
        else:
            # live 模式 dataframe 左端被裁剪: 偏移 = 当前首行对应的 chan klu idx
            offset = state.time2idx[self._tstr(dataframe["date"].iloc[0])]
            start_row = int(dataframe["date"].searchsorted(state.last_ts, side="right"))

        # 逐根回放: 每根K线收盘时刻记录当时缠论状态(回测无未来函数的关键)
        feat_rows: list[dict] = []
        for i in range(start_row, len(dataframe)):
            row = dataframe.iloc[i]
            try:
                state.chan.trigger_load({state.lv: [self._make_klu(row)]})
            except Exception as e:
                logger.error(f"[{pair}] chan trigger_load failed at {row['date']}: {e}")
                state.broken = True
                break
            chan_idx = state.fed_idx
            state.fed_idx += 1
            state.time2idx[self._tstr(row["date"])] = chan_idx
            state.last_ts = row["date"]
            feat_rows.append(self._snapshot(state))

        if feat_rows:
            pos = dataframe.columns.get_loc
            for col in self._CHAN_COLS:
                dataframe.iloc[start_row:start_row + len(feat_rows), pos(col)] = [
                    f[col] for f in feat_rows
                ]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = dataframe["chan_bsp_buy"]
        dataframe.loc[dataframe["chan_bsp_buy"] == 1, "enter_tag"] = dataframe[
            "chan_bsp_buy_tag"
        ]
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = dataframe["chan_bsp_sell"]
        dataframe.loc[dataframe["chan_bsp_sell"] == 1, "exit_tag"] = dataframe[
            "chan_bsp_sell_tag"
        ]
        return dataframe

    # ------------------------------------------------------------------
    # 缠论结构止损: 跟踪最近一个下笔的低点
    # 初始即入场笔低点(买点结构失效位); freqtrade 止损只升不降 -> 移动止损
    # ------------------------------------------------------------------
    def _be_threshold(self, pair: str, trade: Trade) -> float:
        """保本触发阈值(子类可覆写为 1R 动态阈值), 返回利润比例"""
        return self.be_trigger

    def _special_entry_stop(self, pair: str, trade: Trade, row) -> Optional[float]:
        """子类钩子: 按入场标签定制初始结构止损价(返回None走默认笔低点逻辑)"""
        return None

    def _entry_stop_price(self, pair: str, trade: Trade) -> Optional[float]:
        """入场结构止损: 入场时刻结构位 + 宽垫, 开仓时一次性确定并缓存"""
        cached = trade.get_custom_data("chan_entry_stop")
        if cached is not None:
            return float(cached)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None
        # 取开仓K线那一行(行上记录的是入场时刻的结构状态, 不会追溯变化)
        idx = int(dataframe["date"].searchsorted(trade.open_date_utc, side="right")) - 1
        row = dataframe.iloc[max(idx, 0)]
        price = self._special_entry_stop(pair, trade, row)
        if price is None:
            ref, amp = row.get("chan_stop_ref"), row.get("chan_stop_amp")
            if ref is None or pd.isna(ref) or ref <= 0:
                return None
            amp = 0.0 if (amp is None or pd.isna(amp)) else float(amp)
            buf = max(self.min_stop_buffer, self.entry_stop_amp_k * amp)
            price = float(ref) * (1 - buf)
        trade.set_custom_data("chan_entry_stop", price)
        return price

    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
        current_profit: float, after_fill: bool, **kwargs
    ) -> Optional[float]:
        # 收集所有位于"现价下方"的有效结构止损候选, 取最高者;
        # 注意: 入场时价格往往在前中枢下方(背驰买点的形成条件), 此时中枢位
        # 不构成有效止损候选, 不能参与 max, 否则会误判为结构破坏
        cands: list = []
        has_structure = False
        entry_stop = self._entry_stop_price(pair, trade)
        if entry_stop is not None:
            has_structure = True
            if entry_stop < current_rate:
                cands.append(entry_stop)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is not None and len(dataframe) > 0:
            row = dataframe.iloc[-1]
            # 跟踪1: 开仓价上方且仍低于现价的中枢, 跌破其下沿=最近中枢被破坏
            zs_low = row.get("chan_zs_low")
            if zs_low is not None and not pd.isna(zs_low) and zs_low > 0:
                has_structure = True
                if zs_low > trade.open_rate and zs_low < current_rate:
                    cands.append(float(zs_low) * (1 - self.zs_trail_buffer))
            # 跟踪2: 最近下笔低点(窄垫)
            ref, amp = row.get("chan_stop_ref"), row.get("chan_stop_amp")
            if ref is not None and not pd.isna(ref) and ref > 0:
                has_structure = True
                a = 0.0 if (amp is None or pd.isna(amp)) else float(amp)
                buf = max(self.min_stop_buffer, self.trail_stop_amp_k * a)
                trail = float(ref) * (1 - buf)
                if trail < current_rate:
                    cands.append(trail)
        # 保本位: 浮盈达标后锁定开仓价上方一点
        if current_profit > self.be_trigger:
            be_price = trade.open_rate * (1 + self.be_lock)
            if be_price < current_rate:
                cands.append(be_price)
        if cands:
            return stoploss_from_absolute(
                max(cands), current_rate, is_short=trade.is_short,
                leverage=trade.leverage,
            )
        if not has_structure:
            return None  # 无任何结构信息, 维持现状
        # 有结构信息但所有参考位均在价格上方: 买点结构已被破坏, 贴身立即离场
        return -0.001

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float,
        time_in_force: str, current_time: datetime, entry_tag: Optional[str],
        side: str, **kwargs
    ) -> bool:
        # 入场价格显著偏离结构参考位(追高>3%)时放弃本次进场
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return True
        ref = dataframe.iloc[-1].get("chan_stop_ref")
        if ref is not None and not pd.isna(ref) and float(ref) > 0:
            if rate > float(ref) * 1.03:
                return False
        return True
