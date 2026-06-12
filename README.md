# SupportIQ Web

SupportIQ Web 是一个基于 Streamlit 构建的综合性客户支持分析仪表板。它提供交互式可视化和机器学习驱动的支持工单数据分析洞察。

**重要提示：** 这是一个使用公共客户支持数据集的案例研究，不隶属于任何特定公司。

## 📂 项目结构

```
supportiq-web/
├── README.md                      # 项目文档
├── requirements.txt               # Python 依赖
├── run_pipeline.bat               # Windows 管道运行脚本
├── app.py                         # Streamlit 主应用
├── .streamlit/
│   └── config.toml                # Streamlit 配置
├── data/                          # 数据存储
│   ├── raw/                       # 原始CSV数据
│   │   └── customer_support_tickets.csv
│   └── processed/                 # 处理后的数据
│       ├── tickets_cleaned.csv
│       ├── daily_metrics.csv
│       ├── issue_type_metrics.csv
│       ├── priority_metrics.csv
│       ├── agent_performance.csv
│       └── channel_metrics.csv
├── models/                        # ML模型（pickle格式）
│   ├── priority_model.pkl
│   ├── routing_model.pkl
│   ├── csat_model.pkl
│   └── sla_model.pkl
├── scripts/                       # ETL和ML脚本
│   ├── 01_download_kaggle_data.py # 下载Kaggle数据集（无需登录）
│   ├── 02_etl_pipeline.py        # ETL管道
│   └── 03_train_models.py        # ML模型训练
└── doc/                           # 文档
    └── files_overview.md          # 文件详细说明
```

## ✨ 功能特性

### 📊 执行仪表盘 (Executive Dashboard)
- 关键绩效指标 (KPIs)
- 实时支持健康状态监控
- 工单量趋势分析
- CSAT评分分析

### 🎫 工单分析 (Ticket Analysis)
- 按渠道、优先级、问题类型筛选
- 解决时间分布
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

## 🛠️ 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 前端 | Streamlit | 1.28+ |
| 数据存储 | CSV Files | - |
| 机器学习框架 | scikit-learn | 1.2+ |
| 可视化 | Plotly | 5.15+ |
| 数据处理 | Pandas | 2.0+ |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip (Python包管理器)
- 虚拟环境: `D:\kw\vt\.venv`

### 安装依赖

```bash
# 激活虚拟环境
D:\kw\vt\.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 运行完整管道

**Windows批处理（推荐）:**
```batch
run_pipeline.bat
```

**手动步骤:**
```bash
# 步骤1: 下载/生成数据
D:\kw\vt\.venv\Scripts\python.exe scripts/01_download_kaggle_data.py

# 步骤2: 运行ETL管道
D:\kw\vt\.venv\Scripts\python.exe scripts/02_etl_pipeline.py

# 步骤3: 训练ML模型
D:\kw\vt\.venv\Scripts\python.exe scripts/03_train_models.py

# 步骤4: 启动Streamlit应用
D:\kw\vt\.venv\Scripts\streamlit.exe run app.py
```

### 数据来源

本项目使用模拟数据生成，基于 Kaggle Customer Support Ticket Dataset 的真实结构：
- 数据集参考: https://www.kaggle.com/suraj520/customer-support-ticket-dataset

由于环境限制无法安装 `kagglehub`，脚本会自动生成5000条真实结构的模拟数据。

## 🔄 管道流程

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 1. 数据下载     │ -> │ 2. ETL管道      │ -> │ 3. 模型训练     │ -> │ 4. Streamlit    │
│   /生成         │    │   (清洗/转换)   │    │   (ML模型)      │    │   仪表板        │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                     │                      │                      │
         ▼                     ▼                      ▼                      ▼
    data/raw/            data/processed/          models/*.pkl        http://localhost:8501
```

## 🤖 机器学习模型

| 模型 | 用途 | 算法 |
|------|------|------|
| Priority Prediction | 预测工单优先级 | Random Forest |
| Routing Model | 分类问题类型 | Random Forest |
| CSAT Prediction | 预测CSAT评分 | Random Forest (回归) |
| SLA Breach Prediction | 预测SLA违约 | Random Forest |

## 📈 关键指标

### 量度指标
- total_tickets, ticket_volume_by_channel, ticket_volume_by_priority

### 效率指标
- median_first_response_min, p75_resolution_min, solved_rate

### 体验指标
- avg_csat, low_csat_rate

### 风险指标
- sla_breach_rate, escalation_rate, reopened_count

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

## 📜 许可证

MIT License

## 📝 免责声明

本项目仅用于教育和作品集展示目的。
