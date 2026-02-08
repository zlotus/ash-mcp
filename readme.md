# ash-mcp

基于 `baostock + akshare` 的 MCP 服务，面向 A 股长期持有与低频交易分析。

## 设计原则

1. 单一工具入口：每个功能只保留一个 tool。
2. 数据源优先级：在线查询以 `baostock` 为主；夜间预取允许使用 `akshare` 补齐缺失字段（如总股本）。
3. 低频优先：围绕周/月级决策，不追求高频撮合。
4. LLM 友好：输出尽量为精简表格和摘要，控制 token。

## 功能地图

### A. 基础数据层（baostock）

- `get_current_time`：服务器时间
- `get_stock_basic`：证券基础信息
- `get_stock_kdata`：股票历史 K 线（d/w/m/分钟）
- `get_industry_info`：行业分类映射
- `get_dividend_info`：分红派息数据
- `get_profit_info`：季度盈利能力
- `get_operation_info`：季度营运能力
- `get_growth_info`：季度成长能力
- `get_index_data`：指数历史 K 线
- `get_valuation_info`：PE/PB/PS/PCF 时序

### B. 市场与个股分析层

- `get_market_indices`：三大指数市场概览（baostock -> akshare）
- `get_sector_fund_flow`：行业资金流（akshare）
- `get_north_fund_flow`：北向资金汇总（akshare）
- `get_stock_spot`：个股最新行情快照（baostock -> akshare）
- `get_financial_analysis`：财务指标摘要（baostock -> akshare）
- `get_valuation_status`：5 年 PE 分位估值结论（baostock -> akshare）
- `get_stock_news`：个股新闻（akshare）
- `get_dragon_tiger_list`：龙虎榜（akshare）

### C. 长期持有量化层

- `get_grid_strategy`：网格交易计划（baostock -> akshare）
- `get_long_term_factor_score`：长期多因子评分（估值/质量/成长/分红）
- `get_low_freq_backtest`：低频回测（月度定投 + 手续费）
- `get_portfolio_rebalance_plan`：组合再平衡（等权/逆波动）
- `get_value_candidates_and_grid`：围绕锚点股票筛选同类低估标的并输出各自网格计划（支持硬约束：估值分位/ROE/负债率/分红年限/市值相似度/波动率；优先使用预取缓存）
  - 返回包含：约束配置、候选评分表、入选原因、每只股票的网格计划

## 典型工作流

### 工作流 1：长期选股初筛

1. `get_long_term_factor_score`
2. `get_financial_analysis`
3. `get_valuation_status`

### 工作流 2：单票长期持有验证

1. `get_stock_kdata`
2. `get_low_freq_backtest`
3. `get_stock_news`（解释近期异常波动）

### 工作流 3：组合月度调仓

1. `get_portfolio_rebalance_plan`
2. `get_market_indices`
3. `get_sector_fund_flow`

## 运行方式

### 克隆仓库（含子模块）

```bash
git clone --recurse-submodules https://github.com/zlotus/ash-mcp.git
```

若已克隆主仓再补拉子模块：

```bash
git submodule update --init --recursive
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 查看工具列表

```bash
python a_share_value_mcp.py -h
```

### Gemini JSON 配置（MCP）

将服务按 JSON 方式配置到 Gemini（示例）：

```json
{
  "mcpServers": {
    "ash-mcp": {
      "command": "python",
      "args": ["a_share_value_mcp.py"],
      "cwd": "/home/wsl/quant/ash-mcp"
    }
  }
}
```

若你使用固定环境，也可把 `command` 改为绝对路径（例如 `/home/wsl/miniconda3/envs/quant/bin/python`）。

### 夜间预取（推荐）

```bash
python prefetch_strategy.py
```

说明：

- 预取内容：`industry_map`、`metrics`、`financial_cache`、`cap_map`。
- `financial_cache.total_share` 优先来自 `baostock query_profit_data`，缺失时夜间再用 `akshare` 兜底。
- 运行中会按固定进度 checkpoint，持续写入 `prefetch-file-server/data/strategy_cache.json`。

### 缓存发布服务（可选）

- 本仓只负责生成缓存文件：`prefetch-file-server/data/strategy_cache.json`。
- 生产环境可使用独立服务（例如 Rust 实现的 `prefetch-file-server`）对该文件提供下载接口。
- 该发布服务不属于本仓核心运行依赖，可以独立仓库维护并单独部署。

### 预取调试日志（临时）

```bash
PREFETCH_DEBUG=1 python prefetch_strategy.py
```

说明：仅该次进程启用 DEBUG 级别，默认运行不受影响。

### 预取进度日志速查

预取脚本会输出类似：

`[HS300] Progress: 120/300 | added=95 | cache_hit=20 | fetched=95 | skipped=5 | share_unavailable=3 | failed=0`

字段含义：

- `count/total`：已处理股票数 / 总股票数。
- `added`：本次运行新增写入 `metrics` 的股票数量。
- `cache_hit`：命中既有 `metrics` 缓存的股票数量。
- `fetched`：本次实际完成计算并写入的股票数量（通常与 `added` 接近）。
- `skipped`：因关键数据缺失（如估值序列为空）跳过的数量。
- `share_unavailable`：无法得到有效总股本（`total_share`）的数量，会写 `unavailable: true`。
- `failed`：请求或计算异常数量（可配合 DEBUG 日志排查具体 symbol）。

## 项目结构

```text
ash-mcp/
├─ a_share_value_mcp.py      # MCP 入口与工具注册（轻量 wrapper）
├─ prefetch_strategy.py       # 夜间缓存预取脚本（低频任务）
├─ models/
│  └─ inputs.py              # 枚举与全部输入模型
├─ core/
│  └─ utils.py               # 通用工具函数（格式化、打分、权重处理等）
├─ data/
│  ├─ baostock_client.py     # baostock 访问封装
│  └─ market_data.py         # 价格序列获取（baostock 优先 + akshare fallback）
├─ tools/
│  ├─ stock.py               # 基础证券与行情类工具实现
│  ├─ market.py              # 市场概览与新闻类工具实现
│  ├─ fundamental.py         # 基本面与估值类工具实现
│  └─ strategy.py            # 策略/回测/再平衡类工具实现
└─ readme.md
```

## 参数规范

- 股票代码建议使用 6 位数字（如 `600519`），内部自动规范化。
- 日期支持 `YYYYMMDD` 或 `YYYY-MM-DD`。
- 长期分析默认使用较长回看窗口；如需更快响应可缩小范围。

## 错误处理

- 工具统一通过 `handle_error` 返回可读错误信息。
- 当主数据源失败时，优先尝试 fallback；若仍失败，返回明确错误原因。
- 对 `financial_cache` 中不可用总股本会写 `unavailable: true`，避免重复无效请求。

## 后续重构建议

建议下一阶段将单文件拆分为多模块（`models/`, `services/`, `tools/`），降低耦合并提升可维护性。

## 定时任务示例

```cron
0 1 * * * cd /home/wsl/quant/ash-mcp && /home/wsl/miniconda3/envs/quant/bin/python3 prefetch_strategy.py >> prefetch.log 2>&1
```
