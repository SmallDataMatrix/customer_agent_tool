# SupportIQ Web - 文件功能说明

## 概述

本文件详细说明 SupportIQ Web 项目中每个文件的功能和作用。

---

## 📄 根目录文件

### 1. README.md
**功能**: 项目主文档，包含项目概述、功能特性、技术栈、安装和运行说明。

### 2. requirements.txt
**功能**: 项目依赖清单，包含所有需要安装的 Python 包及其版本要求。

**主要依赖**:
- `streamlit>=1.28.0` - 前端框架
- `pandas>=2.0.0` - 数据处理
- `scikit-learn>=1.2.0` - 机器学习
- `plotly>=5.15.0` - 可视化
- `numpy>=1.24.0` - 数值计算

### 3. run_pipeline.bat
**功能**: Windows 批处理脚本，用于自动执行完整的数据管道流程：
1. 下载/生成数据
2. 运行ETL管道
3. 训练ML模型
4. 启动Streamlit应用

---

## 📁 配置文件

### .streamlit/config.toml
**功能**: Streamlit 应用配置文件

**配置项**:
- `browser.gatherUsageStats = false` - 禁用使用统计收集
- `server.port = 8501` - 服务端口

---

## 📊 Streamlit 应用

### app.py
**功能**: Streamlit 主应用，包含7个分析页面：

| 页面 | 功能 |
|------|------|
| Executive Dashboard | 执行仪表盘，展示关键KPIs |
| Ticket Analysis | 工单分析，支持多维度筛选 |
| Agent Performance | 代理绩效分析 |
| Issue Analytics | 问题类型分析 |
| SLA Monitoring | SLA违约监控 |
| CSAT Insights | CSAT评分洞察 |
| Trend Analysis | 趋势分析 |

**数据加载**: 从 `data/processed/` 目录加载CSV文件

---

## 📁 scripts/ - ETL和机器学习脚本

### 1. scripts/01_download_kaggle_data.py
**功能**: 数据下载/生成脚本

**处理流程**:
1. 创建数据目录 (`data/raw/`, `data/processed/`, `models/`)
2. 尝试使用 kagglehub 下载真实Kaggle数据集
3. 若失败则生成模拟数据（5000条记录）

**输出文件**:
- `data/raw/customer_support_tickets.csv`

**生成的字段**:
- ticket_id, ticket_created_at, channel, priority, issue_type
- sub_issue, customer_id, customer_name, customer_email
- agent_id, agent_name, status, first_response_time_min
- resolution_time_min, csat_score, ticket_text, tags
- solved_by_agent, reopened, queue_name, language
- timezone, escalated, handling_time_min, wait_time_min

### 2. scripts/02_etl_pipeline.py
**功能**: ETL管道脚本，负责数据清洗和转换

**处理流程**:
1. **Extract**: 从 `data/raw/` 读取原始数据
2. **Transform**: 
   - 数据清洗（缺失值处理）
   - 标准化字段格式
   - 添加衍生字段（日期、小时、星期、是否周末、是否工作时间）
   - CSAT分类（Positive/Negative）
3. **Load**: 将处理后的数据保存到 `data/processed/`

**输出文件**:
| 文件 | 内容 |
|------|------|
| `tickets_cleaned.csv` | 清洗后的工单明细数据 |
| `daily_metrics.csv` | 每日汇总指标 |
| `issue_type_metrics.csv` | 按问题类型汇总 |
| `priority_metrics.csv` | 按优先级汇总 |
| `agent_performance.csv` | 代理绩效指标 |
| `channel_metrics.csv` | 按渠道汇总 |

**关键转换逻辑**:
- 将时间字段转换为datetime格式
- 创建 `created_date`, `created_hour`, `created_dayofweek` 字段
- 根据CSAT评分创建 `csat_category` 字段（>=4为Positive）
- 计算 `is_weekend`, `is_business_hours` 标志

### 3. scripts/03_train_models.py
**功能**: 机器学习模型训练脚本

**训练的模型**:

| 模型 | 目标变量 | 算法 | 评估指标 |
|------|----------|------|----------|
| Priority Model | priority | RandomForestClassifier | Accuracy, F1-score |
| Routing Model | issue_type | RandomForestClassifier | Accuracy, F1-score |
| CSAT Model | csat_score | RandomForestRegressor | MAE, R2 |
| SLA Model | sla_breach | RandomForestClassifier | Accuracy, F1-score |

**模型保存位置**: `models/*.pkl`

**特征工程**:
- 类别特征: channel, issue_type, sub_issue, language, priority
- 数值特征: is_weekend, is_business_hours, first_response_time_min, resolution_time_min, handling_time_min, wait_time_min

---

## 📁 data/ - 数据目录

### data/raw/
**功能**: 存储原始数据

- `customer_support_tickets.csv` - 原始工单数据（5000条记录）

### data/processed/
**功能**: 存储处理后的数据

#### tickets_cleaned.csv
**说明**: 清洗后的工单明细数据

**关键字段**:
- ticket_id: 工单唯一标识
- ticket_created_at: 创建时间
- channel: 渠道（Email/Chat/Phone/Social Media）
- priority: 优先级（Low/Medium/High）
- issue_type: 问题类型（Account/Payment/Shipping/Technical/Returns）
- csat_score: CSAT评分（1-5）
- sla_breach: 是否违约（True/False）
- resolution_time_min: 解决时间（分钟）

#### daily_metrics.csv
**说明**: 每日汇总指标

**关键字段**:
- created_date: 日期
- total_tickets: 总工单数
- solved_tickets: 已解决工单数
- median_first_response_min: 中位数首次响应时间
- avg_csat: 平均CSAT评分
- sla_breach_rate: SLA违约率
- solved_rate: 解决率

#### issue_type_metrics.csv
**说明**: 按问题类型汇总的指标

**关键字段**:
- issue_type: 问题类型
- total_tickets: 总工单数
- avg_resolution_min: 平均解决时间
- avg_csat: 平均CSAT评分
- sla_breach_rate: SLA违约率

#### priority_metrics.csv
**说明**: 按优先级汇总的指标

**关键字段**:
- priority: 优先级
- total_tickets: 总工单数
- avg_resolution_min: 平均解决时间
- avg_csat: 平均CSAT评分
- sla_breach_rate: SLA违约率

#### agent_performance.csv
**说明**: 代理绩效指标

**关键字段**:
- agent_id: 代理ID
- agent_name: 代理姓名
- tickets_handled: 处理工单数
- avg_first_response_min: 平均首次响应时间
- avg_resolution_min: 平均解决时间
- avg_csat: 平均CSAT评分
- solved_rate: 解决率

#### channel_metrics.csv
**说明**: 按渠道汇总的指标

**关键字段**:
- channel: 渠道
- total_tickets: 总工单数
- avg_resolution_min: 平均解决时间
- avg_csat: 平均CSAT评分
- sla_breach_rate: SLA违约率

---

## 📁 models/ - 模型目录

**功能**: 存储训练好的机器学习模型（pickle格式）

| 文件 | 模型名称 | 用途 |
|------|----------|------|
| `priority_model.pkl` | Priority Prediction Model | 预测工单优先级 |
| `routing_model.pkl` | Routing Model | 分类问题类型 |
| `csat_model.pkl` | CSAT Prediction Model | 预测CSAT评分 |
| `sla_model.pkl` | SLA Breach Prediction Model | 预测SLA违约 |

**模型结构**:
```python
{
    'model': sklearn.pipeline.Pipeline,  # 完整的预处理+模型管道
    'label_encoder': sklearn.preprocessing.LabelEncoder,  # 类别编码器（分类模型）
    'accuracy': float,  # 准确率（分类模型）
    'f1_score': float,  # F1分数（分类模型）
    'mae': float,  # MAE（回归模型）
    'r2_score': float  # R2分数（回归模型）
}
```

---

## 🔄 数据管道流程

```
scripts/01_download_kaggle_data.py
        ↓
    data/raw/customer_support_tickets.csv
        ↓
scripts/02_etl_pipeline.py
        ↓
    data/processed/*.csv
        ↓
scripts/03_train_models.py
        ↓
    models/*.pkl
        ↓
    app.py (Streamlit Dashboard)
```

---

## 📊 关键指标说明

### 量度指标
| 指标 | 说明 | 计算方式 |
|------|------|----------|
| total_tickets | 总工单数 | COUNT(ticket_id) |
| ticket_volume_by_channel | 各渠道工单量 | COUNT按channel分组 |
| ticket_volume_by_priority | 各优先级工单量 | COUNT按priority分组 |

### 效率指标
| 指标 | 说明 | 计算方式 |
|------|------|----------|
| median_first_response_min | 中位数首次响应时间 | MEDIAN(first_response_time_min) |
| p75_resolution_min | 75百分位解决时间 | PERCENTILE(resolution_time_min, 75) |
| solved_rate | 解决率 | solved_tickets / total_tickets * 100 |

### 体验指标
| 指标 | 说明 | 计算方式 |
|------|------|----------|
| avg_csat | 平均CSAT评分 | AVG(csat_score) |
| low_csat_rate | 低CSAT率 | COUNT(csat_score < 4) / total * 100 |

### 风险指标
| 指标 | 说明 | 计算方式 |
|------|------|----------|
| sla_breach_rate | SLA违约率 | COUNT(sla_breach=True) / total * 100 |
| escalation_rate | 升级率 | COUNT(escalated=True) / total * 100 |
| reopened_count | 重开工单数 | COUNT(reopened=True) |

---

## 👥 数据实体关系

```
Customers ─── creates ─── Tickets ─── assigned to ─── Agents
    │                              │
    │                              └── categorized as ─── Issue Types
    └── contacted via ─── Channels
```

**实体说明**:
- **Customers**: 客户信息（customer_id, customer_name, customer_email）
- **Agents**: 代理信息（agent_id, agent_name）
- **Tickets**: 工单信息（核心实体）
- **Channels**: 渠道（Email, Chat, Phone, Social Media）
- **Issue Types**: 问题类型（Account, Payment, Shipping, Technical, Returns）
