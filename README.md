# freqtradeX — 缠论结构 × Smart Money × ML 的 freqtrade 策略库

基于 [chan.py](https://github.com/Vespa314/chan.py)（缠论计算引擎，已 vendor 到本仓库）+
Smart Money Concepts 流动性过滤 + BTC 市场锚 + 机器学习概率门控 + 5 分钟精细执行，
构建在 [freqtrade](https://github.com/freqtrade/freqtrade) 之上的加密货币交易策略。

> **风险声明**：本项目仅供研究学习。加密货币交易风险极高，请先在 dry-run 模式长期验证，
> 并自行承担一切交易后果。回测收益不代表未来表现。

## 策略与实测成绩

回测口径：Binance/OKX 永续 1h，10 个主流币（BTC/ETH/SOL/BNB/XRP/DOGE/ADA/AVAX/LINK/LTC），
2024-03 ~ 2026-09（2.55 年），单笔 3000 USDT，1x 杠杆，`--timeframe-detail 5m`。

| 策略 | 市场 | 全期收益 | 年化 | 胜率 | 特点 |
|---|---|---|---|---|---|
| **ChanSmartMoneyStrategy** | USDT 永续（多+空） | **+79.8%** | ~27% | 55.6%（高置信子集 68-78%） | 五层完整栈，主力策略 |
| ChanFuturesStrategy (v8默认) | USDT 永续（多+空） | +22.8%（1500U仓） | ~8.4% | 41.5% | 双半期为正的最稳健基座，无ML依赖 |
| ChanStructStrategy | 现货（仅做多） | +0.7% | — | — | 教学版，演示最小缠论接入 |

### 五层策略栈（ChanSmartMoneyStrategy）

```
1. 缠论信号引擎    chan.py 逐K回放(trigger_step) → 笔/段/中枢 → 四通道信号
                   空头3a / 空头1p / 多头1p / 多头3a（纯"1"类只作离场事件）
2. SMC流动性过滤   猎杀回收(sweep&reclaim)/FVG失衡/Killzone 共振评分
                   —— 仅门控空头趋势通道（背驰通道直通）
3. BTC市场锚       BTC 1h MA20 regime 选择性闸门
                   β依赖通道需市场同向确认, 特质通道直通
4. ML概率门控      HistGradientBoosting, 47因子, AUC 0.655(留出集),
                   仅多头, proba>=0.65 放行（空头AUC不足, 不部署）
5. 5m精细执行      --timeframe-detail 5m: 止损/跟踪在5m粒度评估(与实盘一致)
────────────────────────────────────────────────────────────────
止损体系:  3a类=中枢边界(跌回中枢即失效) / 1p类=入场笔低点+振幅垫
           持仓跟踪=中枢下沿阶梯+笔低点+保本锁定 / 硬顶-20%
出场体系:  段级背驰卖点(笔进场段离场) + 浮盈>=4%遇强卖点止盈 + 平空信号
```

## 目录结构

```
freqtradeX/
├── docker-compose.yml            # freqtrade 官方镜像, 一键 dry-run
├── user_data/
│   ├── chan/                     # vendored chan.py 核心源码(纯标准库, 零依赖)
│   ├── strategies/
│   │   ├── ChanStructStrategy.py      # 现货策略(教学/基线)
│   │   ├── ChanFuturesStrategy.py     # 期货策略基座(SMC策略继承自它)
│   │   └── ChanSmartMoneyStrategy.py  # 主力策略(含ML门控)
│   ├── configs/
│   │   ├── config_spot.json           # 现货配置(dry-run)
│   │   └── config_futures.json        # 永续配置(dry-run, 3000U仓位)
│   └── ml/
│       ├── train_ml.py                # ML训练脚本(时间切分+置换重要性)
│       └── ml_best_long.joblib        # 预训练多头模型(AUC 0.655)
└── README.md
```

## 依赖处理（重要）

**无需安装任何额外依赖。**

- chan.py 核心是**纯 Python 标准库**实现（无 pandas/numpy/talib 依赖），直接 vendor 使用
- 官方镜像 `freqtradeorg/freqtrade:stable` 已内置策略所需全部依赖
  （pandas/numpy/sklearn/joblib——sklearn 经由镜像内的 hyperopt 依赖链提供）
- 不需要自建镜像、不需要 pip install、不需要挂载任何外部路径

## 快速开始

前置：Docker（含 docker compose 插件）。

```bash
git clone https://github.com/beatyman/freqtradeX && cd freqtradeX

# 1) 下载数据(推荐 OKX: 全历史分页无限制; Binance 历史接口可能限频)
docker compose run --rm freqtrade download-data --exchange okx \
    --trading-mode futures -t 1h 5m --timerange 20220101- \
    --pairs BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT BNB/USDT:USDT \
            XRP/USDT:USDT DOGE/USDT:USDT ADA/USDT:USDT AVAX/USDT:USDT \
            LINK/USDT:USDT LTC/USDT:USDT

# 2) 回测主力策略
docker compose run --rm freqtrade backtesting \
    --config user_data/configs/config_futures.json \
    --strategy ChanSmartMoneyStrategy \
    --timerange 20240301-20260920 --cache none --timeframe-detail 5m

# 3) dry-run 挂机(FreqUI: http://localhost:8080)
docker compose up -d
```

现货策略（需先下载现货数据）：

```bash
docker compose run --rm freqtrade download-data --config user_data/configs/config_spot.json \
    --timerange 20240101- -t 1h
docker compose run --rm freqtrade backtesting --config user_data/configs/config_spot.json \
    --strategy ChanStructStrategy --timerange 20240301-
```

> 提示：给已有数据补更早历史需先删除旧文件（或加 `--prepend`），freqtrade 默认只向前追加。

## ML 训练管线（可选，重新训练模型）

策略内置训练数据导出开关：设 `CHAN_ML_EXPORT=1` 跑一次回测，即在每个信号行
导出 47 个因果因子 + 双重屏障标签（固定 +3%/-2% 与 ATR 自适应）到
`user_data/ml_signals.csv`（该文件已 gitignore）：

```bash
# 导出(约40分钟, 10币×4.5年)
docker compose run --rm -e CHAN_ML_EXPORT=1 freqtrade backtesting \
    --config user_data/configs/config_futures.json \
    --strategy ChanSmartMoneyStrategy \
    --timerange 20220301-20260920 --cache none \
    --pairs BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT BNB/USDT:USDT \
            XRP/USDT:USDT DOGE/USDT:USDT ADA/USDT:USDT AVAX/USDT:USDT \
            LINK/USDT:USDT LTC/USDT:USDT

# 训练+验证(时间切分 2025-11-01, 输出 AUC 矩阵与置换重要性)
docker compose run --rm --entrypoint python freqtrade user_data/ml/train_ml.py
# 产物 ml_best_long.joblib 自动落位, 策略下次加载即生效
```

训练要点（实测经验）：
- **数据分层决定成败**：混入二线山寨会毒化训练（AUC 0.55→0.44），只用主流币
- **ATR 自适应标签**对多头显著有效（0.51→0.55），对空头无效
- **因子按留出集 AUC 增量准入**：Alpha158 回归族14因子实测使 AUC 0.655→0.570
  （样本1203时的维度灾难），已回滚——不要按数量堆因子

## 关键设计决策（为什么这样架构）

1. **逐K回放无未来函数**：chan 计算走 `trigger_step` 逐根喂入，每根K线记录"当前帧"
   状态；回测列 ≡ 实盘当下可见信息。`--timeframe-detail 5m` 让止损模拟与实盘的
   5秒循环行为一致（实测 +4.6pt）
2. **训练/实盘同一份因子代码**：`_ml_derived()` 同时服务数据导出与实盘打分，
   杜绝特征漂移
3. **通道差异化处理**：同一信号类型在不同通道（趋势延续/背驰反转 × 多/空 × β/特质）
   的最优过滤、止损、离场配置完全不同——这是20+轮回测实验的结论，全部记录在
   代码注释中
4. **失败实验同样入档**：Route A（1R分批止盈/动态保本）的负结果、Alpha158 因子
   负结果、20币扩容负结果等均在代码注释中留有数据，防止重复踩坑

## 已知限制（诚实声明）

- 模型训练期(2022~2025-11)与回测期(2024-03起)存在重叠，+79.8% 偏乐观；
  上线前应实现 walk-forward 滚动重训（AUC 0.655 本身是在真实留出集上测得）
- 空头侧 ML 无优势（AUC<0.53），空头的边际全部来自结构+SMC+BTC锚
- 策略在 10 个主流币上验证；二线山寨实测为负（-1.18%），勿直接扩币

## 致谢与许可

- [Vespa314/chan.py](https://github.com/Vespa314/chan.py)：缠论计算引擎，
  其核心源码（见 `user_data/chan/`，含原 LICENSE）为本仓库的信号基座
- [freqtrade](https://github.com/freqtrade/freqtrade)：交易框架
- Smart Money Concepts 因子定义参考了
  [smart-money-concepts](https://github.com/joshyattridge/smart-money-concepts)
  的语义（因其实现在回放场景含未来信息，本仓库为因果重实现）
