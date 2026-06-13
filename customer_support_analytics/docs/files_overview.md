# Customer Support Analytics - File Overview

## Overview

This document describes the function, processing flow, and relationships of each file in the Customer Support Analytics project. This project is a customer support intelligence analysis platform that helps businesses gain insights into customer support operations through data pipeline processing, AI model training, statistical analysis, and interactive dashboards.

---

## Core Processing Flow

```
[Data Collection] → [Data Cleaning] → [Feature Engineering] → [Model Training] → [Visualization]
     ↓                  ↓                  ↓                  ↓                ↓
  Step 1            Step 2             Step 3             Step 4            Step 5
```

---

## 📄 Root Directory Files

### 1. README.md
**Function**: Main project documentation serving as the entry point for the project, containing:

- **Project Introduction**: Background, goals, and value proposition
- **Features**: List of main supported functionalities
- **Tech Stack**: Technology frameworks and tools used
- **Installation Guide**: Environment configuration and dependency installation
- **Usage Instructions**: How to start and access the application
- **Directory Structure**: Project file organization

**Use Cases**:
- First read for new team members joining the project
- Quick overview of overall project architecture
- Finding basic execution commands

---

## 📊 Streamlit Application

### app/streamlit_app.py
**Function**: Streamlit main application providing a complete data analysis dashboard with multiple analysis pages.

**Technical Architecture**:
```
Streamlit Framework
    ├── Page routing system (via selectbox)
    ├── Data caching mechanism (@st.cache_data)
    ├── Plotly interactive charts
    └── Custom CSS styling
```

**Page Details**:

#### 1. Executive Dashboard
**Purpose**: Provide at-a-glance KPI overview for senior management.

**Content**:
- Key KPI cards (total tickets, resolution rate, average response time, CSAT score)
- Real-time data statistics
- Quick trend indicators

**Data Source**: `daily_metrics.csv`

#### 2. Ticket Analysis
**Purpose**: Support multi-dimensional ticket data deep analysis.

**Features**:
- Multi-field filtering (channel, priority, status, date range)
- Dynamic data table display
- Detailed field information view

#### 3. Agent Performance
**Purpose**: Evaluate and compare customer support agent performance.

**Analysis Dimensions**:
- Personal ticket volume
- First response time ranking
- Resolution rate comparison
- CSAT score distribution
- Escalation rate analysis

#### 4. Issue Analytics
**Purpose**: Identify high-frequency issues and hotspot areas.

**Analysis Content**:
- Issue type distribution
- Average resolution time by type
- CSAT score comparison
- SLA breach rate analysis

#### 5. SLA Monitoring
**Purpose**: Monitor Service Level Agreement (SLA) compliance.

**Monitoring Metrics**:
- Overall SLA compliance rate
- Priority-based breach rate
- Breach ticket details
- Trend change charts

#### 6. CSAT Insights
**Purpose**: Analyze customer satisfaction distribution and influencing factors.

**Analysis Dimensions**:
- CSAT score distribution
- CSAT comparison by channel, issue type, agent
- Low score ticket tracking
- Improvement suggestions

#### 7. Trend Analysis
**Purpose**: Display key metrics changes over time.

**Time Dimension Analysis**:
- Daily/Weekly/Monthly trends
- Year-over-year/Month-over-month analysis
- Seasonal pattern recognition
- Anomaly point annotation

---

## 📁 steps/ - Pipeline Step Scripts

The pipeline step scripts define the sequential execution flow from data collection to metrics computation.

### 1. download_kaggle_data.py
**Function**: Data collection script responsible for obtaining or generating raw ticket data.

**Processing Flow**:

```
┌──────────────────────────────────────────────────────────────┐
│                    Data Collection Flow                       │
├──────────────────────────────────────────────────────────────┤
│  1. Initialization                                           │
│     ├── Create directory structure                            │
│     └── Import required libraries                            │
│                                                              │
│  2. Data Source Selection                                    │
│     ├── Primary: kagglehub download real dataset             │
│     ├── Secondary: Public data source (GitHub)               │
│     └── Fallback: Generate synthetic dataset                 │
│                                                              │
│  3. Synthetic Data Generation                                │
│     ├── Set random seed (seed=42) for reproducibility       │
│     ├── Generate 5000 ticket records                         │
│     └── Use real distribution probability distributions      │
│                                                              │
│  4. Data Persistence                                        │
│     └── Save to data/raw/customer_support_tickets.csv        │
└──────────────────────────────────────────────────────────────┘
```

### 2. etl_pipeline.py
**Function**: ETL (Extract-Transform-Load) pipeline for data cleaning, transformation, and loading.

**ETL Three-Stage Details**:

#### Stage 1: Extract
Input: `data/raw/customer_support_tickets.csv`
Output: `pd.DataFrame` (raw data)

#### Stage 2: Transform
- Column Name Standardization
- Data Type Conversion
- Categorical Field Standardization
- Missing Value Handling
- Derived Field Creation
- Business Rule Application (SLA breach determination)

#### Stage 3: Load
- `tickets_cleaned.csv`: Complete cleaned ticket details
- `daily_metrics.csv`: Daily aggregated metrics
- `issue_type_metrics.csv`: Issue type aggregated metrics
- `agent_performance.csv`: Agent performance metrics
- `priority_metrics.csv`: Priority aggregated metrics
- `channel_metrics.csv`: Channel aggregated metrics

### 3. train_models.py
**Function**: Machine learning model training, training 4 prediction models for intelligent ticket processing.

**Model Training Flow**:

```
┌──────────────────────────────────────────────────────────────┐
│                    Model Training Flow                        │
├──────────────────────────────────────────────────────────────┤
│  1. Data Preparation                                         │
│     ├── Load tickets_cleaned.csv                             │
│     ├── Select features and target variables                  │
│     └── Data split (80% train / 20% test)                    │
│                                                              │
│  2. Feature Engineering                                      │
│     ├── Categorical features: One-Hot Encoding               │
│     ├── Numeric features: Standardization                    │
│     └── Missing values: Median/constant fill                  │
│                                                              │
│  3. Model Training (RandomForest)                            │
│     ├── Priority Prediction Model                             │
│     ├── Routing Model                                       │
│     ├── CSAT Prediction Model                               │
│     └── SLA Breach Prediction Model                          │
│                                                              │
│  4. Model Evaluation & Persistence                           │
└──────────────────────────────────────────────────────────────┘
```

### 4. compute_metrics.py
**Function**: Calculate and compute business metrics from processed data.

**Metrics Calculated**:
- Volume metrics (total, by channel, by priority)
- Efficiency metrics (response time, resolution time)
- Quality metrics (CSAT, resolution rate)
- SLA compliance metrics

---

## 📁 scripts/analytics/ - Analytical Modules

### ab_test_analysis.py
**Purpose**: Design, execute, and analyze experiments to validate product changes and tooling enhancements.

**Key Features**:

| Feature | Description |
|---------|-------------|
| Sample Size Calculator | Determine required sample size before running experiments |
| Effect Size Calculation | Compute Cohen's d and other effect size metrics |
| Hypothesis Testing | t-test, chi-square, Mann-Whitney U tests |
| Confidence Intervals | Mean and proportion difference intervals |
| Power Analysis | Calculate and evaluate statistical power |
| Results Visualization | Interactive charts for experiment results |

**Integration Point**: Called from the **Experiment Analysis** page in the Streamlit dashboard.

**Practical Application Scenarios**:

1. **Tooling Evaluation**: Test whether new support tooling improves agent efficiency
2. **Process Changes**: Validate if workflow modifications reduce resolution time
3. **Training Impact**: Measure whether training programs improve CSAT scores
4. **Routing Optimization**: Compare different ticket routing algorithms

**Usage Flow**:
```
┌────────────────────────────────────────────────────────────┐
│  Experiment Design → Data Collection → Analysis → Insights  │
└────────────────────────────────────────────────────────────┘
     ↓                  ↓               ↓            ↓
  Sample Size      Experiment      Statistical    Business
  Calculation      Data CSV        Testing        Decision
```

**Output Files**:
- `data/processed/ab_test_data.csv`: Generated experiment data
- Displayed in Streamlit dashboard under Experiment Analysis page

---

## 📈 Causal Inference & Treatment Effects

### causal_inference.py
**Purpose**: Estimate causal effects from observational data, moving beyond correlation to understand true treatment effects.

**Key Methods**:

| Method | Use Case | Practical Application |
|--------|----------|----------------------|
| **Difference-in-Differences (DiD)** | Panel data with pre/post periods | Measure effect of policy change on support metrics |
| **Propensity Score Matching (PSM)** | Selection bias correction | Match treated/untreated tickets for fair comparison |
| **Instrumental Variables (IV)** | Endogeneity treatment | Use instrument to estimate causal effect of response time |
| **Regression Discontinuity (RDD)** | Discontinuity design | Analyze effect at SLA boundary thresholds |
| **Synthetic Control** | Comparative case studies | Create counterfactual for policy evaluation |
| **Sensitivity Analysis** | Robustness checks | Assess how results change under different assumptions |

**Integration Point**: Called from the **CATE Analysis** page in the Streamlit dashboard.

**Practical Application Scenarios**:

1. **Policy Evaluation**: Did implementing new response time targets actually improve CSAT?
2. **Resource Allocation**: What is the causal effect of agent experience on resolution time?
3. **Process Optimization**: Does dedicated queue routing reduce escalation rates?
4. **Training Effectiveness**: Causal impact of agent training programs on ticket outcomes

**Analysis Flow**:
```
Observational Data → Method Selection → Effect Estimation → Sensitivity Check → Business Recommendations
```

### cate_meta_learners.py
**Purpose**: Estimate Conditional Average Treatment Effects (CATE) to understand how effects vary across different subgroups.

**Meta-Learner Approaches**:

| Learner | Approach | Best Use Case |
|---------|----------|---------------|
| **S-Learner** | Single model with treatment indicator | Quick baseline estimation |
| **T-Learner** | Separate models for treatment/control | When treatment effects may differ substantially |
| **X-Learner** | Cross-fitting approach | Recommended for most practical applications |
| **DR-Learner** | Doubly robust estimation | When both outcome and propensity models are accurate |

**Practical Application Scenarios**:

1. **Heterogeneous Treatment Effects**: Which agent segments benefit most from additional training?
2. **Personalized Routing**: How does routing strategy effect vary by ticket priority?
3. **Targeted Interventions**: Which customer segments show largest CSAT improvement from proactive support?
4. **Resource Prioritization**: Where should we allocate support resources for maximum impact?

**Integration Point**: Called from the **CATE Analysis** page in the Streamlit dashboard.

**Output**:
- Treatment effect estimates per subgroup
- Confidence intervals for effect heterogeneity
- Visualization of effect distribution across features

---

## 📊 Statistical Analysis

### statistical_tests.py
**Purpose**: Comprehensive statistical testing framework for support metrics analysis.

**Supported Tests**:
- t-tests (independent, paired)
- Chi-square tests for proportions
- Mann-Whitney U (non-parametric)
- Wilcoxon signed-rank test
- ANOVA and Kruskal-Wallis
- Correlation analysis (Pearson, Spearman)

### regression_analysis.py
**Purpose**: Regression modeling for understanding relationships between support metrics.

**Models**:
- Linear regression for continuous outcomes
- Logistic regression for binary outcomes
- Interpretation of coefficients and marginal effects

---

## ⏩ Time Series Analysis

### time_series_forecasting.py
**Purpose**: Forecast future ticket volumes and identify temporal patterns.

**Features**:
- Trend detection and forecasting
- Seasonality analysis
- ARIMA/ETS forecasting
- Anomaly detection (IQR, CUSUM methods)
- Forecast visualization with confidence intervals

---

## 🔄 Quality & Tracking

### data_quality.py
**Purpose**: Data quality checks and validation for support metrics.

### experiment_tracker.py
**Purpose**: Track and manage experiment results over time.

---

## 📁 SQL Scripts

### sql/*.sql
**Function**: Data cleaning, standardization, and metric calculation using DuckDB.

**Scripts**:
- `clean_tickets.sql`: Data cleaning and standardization
- `standardize_taxonomy.sql`: Issue type taxonomy standardization
- `ticket_metrics.sql`: Metric calculations
- `query_optimization.sql`: Query performance optimization

---

## 📊 Data Files

### data/raw/
- `customer_support_tickets.csv`: Raw ticket data (5000 records)

### data/processed/
- `tickets_cleaned.csv`: Cleaned ticket data
- `daily_metrics.csv`: Daily aggregated metrics
- `issue_type_metrics.csv`: Issue type aggregated metrics
- `priority_metrics.csv`: Priority aggregated metrics
- `agent_performance.csv`: Agent performance metrics
- `channel_metrics.csv`: Channel aggregated metrics
- `ab_test_data.csv`: A/B test experimental data

---

## 🎯 Analytical Capability Summary

| Component | Purpose | Integration |
|-----------|---------|-------------|
| **A/B Testing** | Experiment design and analysis | Streamlit pages |
| **Causal Inference** | Estimate true causal effects | CATE Analysis page |
| **CATE Meta-Learners** | Heterogeneous treatment effects | CATE Analysis page |
| **Statistical Tests** | Hypothesis testing framework | Multiple pages |
| **Regression Analysis** | Relationship modeling | Multiple pages |
| **Time Series** | Forecasting and trend analysis | Trend Analysis page |
| **ML Models** | Predictive features | Core application |

---

## Pipeline Flow with Analytics

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           main.py (Unified Entry)                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
          ▼                          ▼                          ▼
   ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
   │ Step 1      │           │ Step 2      │           │ Step 3      │
   │ Data Download│──────────▶│ ETL Pipeline│──────────▶│ ML Training │
   └─────────────┘           └─────────────┘           └─────────────┘
          │                          │                          │
          ▼                          ▼                          ▼
     data/raw/              data/processed/                models/

                              ┌─────────────────────────────────┐
                              │ Step 4 (auto or manual)          │
                              │ Streamlit App Launch             │
                              └─────────────────────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
          ▼                               ▼                               ▼
   ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
   │ A/B Testing     │         │ Causal Inference│         │ Statistical    │
   │ Analysis        │         │ & CATE          │         │ Tests          │
   └─────────────────┘         └─────────────────┘         └─────────────────┘
          │                               │                               │
          └───────────────────────────────┼───────────────────────────────┘
                                          ▼
                                 Interactive Dashboard
                                    (10 Pages)
```
