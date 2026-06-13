# SupportIQ Web

SupportIQ Web 是一个基于 Streamlit 构建的综合性客户支持分析仪表板。它提供交互式可视化和机器学习驱动的支持工单数据分析洞察。

**重要提示：** 这是一个使用公共客户支持数据集的案例研究，不隶属于任何特定公司。

---

## 🎯 项目概述

本项目展示了 Decision Scientist（决策科学家）在客户支持领域的数据分析能力，包括：

- **数据驱动决策**: 开发指标框架追踪产品绩效，提供可操作的洞察
- **实验设计**: 设计、执行和分析 A/B 测试验证产品变化的影响
- **统计分析**: 应用统计方法进行假设检验和因果推断
- **跨职能协作**: 与产品、工程和运营团队合作推动决策

**参考职位**: Vinted Support Agent Tooling Team - Decision Scientist

---

## 📂 项目结构

```
supportiq_web/
├── README.md                         # 项目文档
├── CHANGELOG.md                     # 变更日志
├── requirements.txt                  # Python 依赖
├── main.py                          # Python CLI 管道运行器
├── app.py                           # Streamlit 主应用（10个分析页面）
├── __init__.py                      # 核心模块初始化（PathManager, Logger）
├── .streamlit/
│   └── config.toml                  # Streamlit 配置
├── data/                            # 数据存储
│   ├── raw/                         # 原始CSV数据
│   │   └── customer_support_tickets.csv
│   ├── processed/                   # 处理后的数据
│   │   ├── tickets_cleaned.csv
│   │   ├── daily_metrics.csv
│   │   ├── issue_type_metrics.csv
│   │   ├── priority_metrics.csv
│   │   ├── agent_performance.csv
│   │   ├── channel_metrics.csv
│   │   ├── ab_test_data.csv         # A/B测试数据
│   │   └── statistical_analysis_results.json
│   └── experiments/                 # 实验追踪数据
├── models/                          # ML模型（pickle格式）
│   ├── priority_model.pkl
│   ├── routing_model.pkl
│   ├── csat_model.pkl
│   └── sla_model.pkl
├── scripts/                         # ETL、ML和统计分析脚本
│   ├── download_kaggle_data.py     # 数据下载/生成
│   ├── etl_pipeline.py             # ETL管道
│   ├── train_models.py             # ML模型训练
│   ├── compute_metrics.py          # DuckDB指标计算
│   ├── ab_test_analysis.py         # A/B测试分析框架
│   ├── statistical_tests.py        # 统计检验套件
│   ├── regression_analysis.py      # 回归分析
│   ├── causal_inference.py         # 因果推断
│   ├── time_series_forecasting.py  # 时间序列预测
│   ├── cate_meta_learners.py      # CATE/异质性效果分析
│   ├── data_quality.py            # 数据质量监控
│   └── experiment_tracker.py        # 实验追踪框架
├── sql/                             # SQL脚本
│   ├── 01_clean_tickets.sql       # 数据清洗
│   ├── 02_standardize_issue_taxonomy.sql  # 问题分类标准化
│   ├── 03_ticket_metrics.sql      # 指标计算
│   └── 04_query_optimization.sql   # 查询优化
├── tests/                           # 单元测试
│   └── __init__.py                 # 测试框架
└── doc/                             # 文档
    ├── files_overview.md            # 文件详细说明
    └── statistical_methods.md       # 统计方法文档
```

---

## ✨ 功能特性

### 📊 执行仪表盘 (Executive Dashboard)
- 关键绩效指标 (KPIs) 及业务解读
- 实时支持健康状态监控
- 工单量趋势分析
- CSAT评分分析
- 首次响应时间分析
- 最近工单列表

### 👥 代理绩效 (Agent Performance)
- 按CSAT和处理工单量排名的顶级代理
- 代理指标对比
- 绩效趋势

### 🔍 问题分析 (Issue Analytics)
- 问题类型对比
- 按问题类型的解决时间
- SLA违约分析

### ⏱️ SLA监控 (SLA Monitoring)
- 每日SLA违约率
- 基于优先级的SLA分析
- 违约工单详情

### 😊 CSAT洞察 (CSAT Insights)
- CSAT趋势分析
- 低CSAT率监控
- 基于问题和优先级的CSAT分析

### 📈 趋势分析 (Trend Analysis)
- 多指标趋势可视化
- 星期模式分析
- 工作时间与非工作时间对比

### 🧪 实验分析 (Experiment Analysis)
- **样本量计算器**: 实验设计前的样本量规划
- **A/B测试分析**: 完整的统计检验框架
- **统计方法支持**:
  - t检验 (独立样本、配对样本)
  - 卡方检验 (比例比较)
  - Mann-Whitney U检验 (非参数)
  - ANOVA (多组比较)
- **功效分析**: 统计功效计算和评估
- **最佳实践指南**: 实验设计、分析和报告规范

### ⏩ 时间序列预测 (Time Series Forecasting)
- 趋势检测与预测
- 季节性分析
- ARIMA/ETS forecasting
- 异常检测 (IQR, CUSUM)
- 预测可视化与置信区间

### 🎯 CATE分析 (Heterogeneous Treatment Effects)
- S-Learner (单模型方法)
- T-Learner (双模型方法)
- X-Learner (交叉拟合) - 推荐
- DR-Learner (双重稳健)
- 异质性效果分析与可视化

---

## 🛠️ 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 前端/可视化 | Streamlit + Plotly | 1.28+ / 5.15+ |
| 数据存储 | CSV + DuckDB | - |
| 机器学习 | scikit-learn | 1.2+ |
| 统计分析 | SciPy | 1.10+ |
| 数据处理 | Pandas + NumPy | 2.0+ / 1.24+ |
| 实验框架 | 自定义 A/B 测试分析模块 | - |

---

## 📊 统计方法论

本项目实现了 Decision Scientist 职位所需的完整统计分析框架：

### 假设检验方法

| 方法 | 用途 | 实现位置 |
|------|------|----------|
| **t检验 (Welch's)** | 两组均值比较 | `ab_test_analysis.py`, `statistical_tests.py` |
| **配对t检验** | 前后对比 | `statistical_tests.py` |
| **卡方检验** | 比例/率比较 | `ab_test_analysis.py`, `statistical_tests.py` |
| **Mann-Whitney U** | 非参数两组比较 | `ab_test_analysis.py`, `statistical_tests.py` |
| **Wilcoxon** | 非参数配对比较 | `statistical_tests.py` |
| **ANOVA** | 多组均值比较 | `statistical_tests.py` |
| **Kruskal-Wallis** | 非参数多组比较 | `statistical_tests.py` |

### 效应量指标

- **Cohen's d**: t检验的效应量 (0.2=小, 0.5=中, 0.8=大)
- **Cramer's V**: 卡方检验的效应量
- **Pearson r / Spearman ρ**: 相关系数
- **Eta-squared (η²)**: ANOVA效应量

### 置信区间

- 95% 置信水平 (α = 0.05)
- 均值差的置信区间
- 比例差的置信区间
- 相关系数的Fisher z变换置信区间

### 因果推断方法

| 方法 | 用途 | 实现位置 |
|------|------|----------|
| **Difference-in-Differences (DiD)** | 面板数据因果效应 | `causal_inference.py` |
| **Propensity Score Matching (PSM)** | 选择偏差校正 | `causal_inference.py` |
| **Instrumental Variables (IV)** | 内生性处理 | `causal_inference.py` |
| **Regression Discontinuity (RDD)** | 断点回归 | `causal_inference.py` |
| **S-Learner** | 异质性处理效应 | `cate_meta_learners.py` |
| **T-Learner** | 异质性处理效应 | `cate_meta_learners.py` |
| **X-Learner** | 异质性处理效应 | `cate_meta_learners.py` |
| **DR-Learner** | 双重稳健估计 | `cate_meta_learners.py` |

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip (Python包管理器)
- 虚拟环境位于项目根目录的 `.venv`

### 安装依赖

```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 运行完整管道

**使用 main.py (推荐):**
```bash
# 完整管道
python main.py

# 清理并重新运行
python main.py --rerun

# 仅运行特定步骤
python main.py --step 2    # ETL
python main.py --step 3    # 训练
python main.py --launch-only  # 仅启动应用

# 仅数据处理（下载 + ETL）
python main.py --data-only
```

**CLI 帮助:**
```bash
python main.py --help
```

### 手动运行各步骤

```bash
# 步骤1: 下载/生成数据
python scripts/download_kaggle_data.py

# 步骤2: 运行ETL管道
python scripts/etl_pipeline.py

# 步骤3: 训练ML模型
python scripts/train_models.py

# 步骤4: 启动Streamlit应用
streamlit run app.py
```

### 数据来源

本项目使用模拟数据生成，基于 Kaggle Customer Support Ticket Dataset 的真实结构：
- 数据集参考: https://www.kaggle.com/suraj520/customer-support-ticket-dataset

---

## 🔄 管道流程

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           main.py (统一入口)                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
          ▼                          ▼                          ▼
   ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
   │ Step 1      │           │ Step 2      │           │ Step 3      │
   │ 数据下载    │──────────▶│ ETL管道     │──────────▶│ 模型训练    │
   └─────────────┘           └─────────────┘           └─────────────┘
          │                          │                          │
          ▼                          ▼                          ▼
     data/raw/              data/processed/                models/

                              ┌─────────────────────────────────┐
                              │ Step 4 (自动或手动)              │
                              │ Streamlit 应用启动               │
                              └─────────────────────────────────┘
                                          │
                                          ▼
                                 http://localhost:8501
```

---

## 🤖 机器学习模型

| 模型 | 用途 | 算法 |
|------|------|------|
| Priority Prediction | 预测工单优先级 | Random Forest |
| Routing Model | 分类问题类型 | Random Forest |
| CSAT Prediction | 预测CSAT评分 | Random Forest (回归) |
| SLA Breach Prediction | 预测SLA违约 | Random Forest |

---

## 📈 关键指标

### 量度指标
- total_tickets, ticket_volume_by_channel, ticket_volume_by_priority

### 效率指标
- median_first_response_min, p75_resolution_min, solved_rate

### 体验指标
- avg_csat, low_csat_rate

### 风险指标
- sla_breach_rate, escalation_rate, reopened_count

---

## 📁 数据文件说明

### 原始数据 (data/raw/)
- `customer_support_tickets.csv` - 原始工单数据（5000条记录）

### 处理后数据 (data/processed/)
- `tickets_cleaned.csv` - 清洗后的工单数据
- `daily_metrics.csv` - 每日汇总指标
- `issue_type_metrics.csv` - 按问题类型汇总
- `priority_metrics.csv` - 按优先级汇总
- `agent_performance.csv` - 代理绩效指标
- `channel_metrics.csv` - 按渠道汇总
- `ab_test_data.csv` - A/B测试实验数据
- `statistical_analysis_results.json` - 统计分析结果

---

## 🧪 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试类
python -m pytest tests/ -k "TestABTestAnalysis" -v
```

---

## 📜 许可证

MIT License

---

## 📝 免责声明

本项目仅用于教育和作品集展示目的。
