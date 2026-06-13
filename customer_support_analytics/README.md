# Customer Support Analytics

A comprehensive customer support analysis platform built with Streamlit. It provides interactive visualizations and machine learning-driven insights for support ticket data analysis.

**Important Note:** This is a case study using a public customer support dataset, not affiliated with any specific company.

---

## 🎯 Project Overview

This project demonstrates Decision Scientist capabilities in the customer support domain:

- **Data-Driven Decision Making**: Develop metric frameworks to track product performance and provide actionable insights
- **Experiment Design**: Design, execute, and analyze A/B tests to validate the impact of product changes
- **Statistical Analysis**: Apply statistical methods for hypothesis testing and causal inference
- **Cross-Functional Collaboration**: Work with product, engineering, and operations teams to drive decisions

**Reference Position**: Vinted Support Agent Tooling Team - Decision Scientist

---

## 📂 Project Structure

```
customer_support_analytics/              # Project root
├── .venv/                              # Python virtual environment
├── .gitignore                          # Git ignore configuration
├── README.md                           # This file
├── LICENSE                             # MIT License
├── CHANGELOG.md                        # Change log
├── requirements.txt                     # Python dependencies
├── setup.py                            # Package setup
├── pyproject.toml                       # Project configuration
│
├── trained_model/                      # Trained ML models
│   ├── csat_model.pkl
│   ├── priority_model.pkl
│   ├── routing_model.pkl
│   └── sla_model.pkl
│
├── data/                              # Data directory
│   ├── raw/                           # Raw data
│   └── processed/                     # Processed data
│
├── src/
│   ├── supportiq_web.py               # Compatibility shim
│   └── customer_support_analytics/    # Main package
│       ├── __init__.py                # Package init
│       ├── __version__.py             # Version info
│       │
│       ├── app/                       # Web application
│       │   ├── __init__.py
│       │   ├── main.py                # CLI pipeline runner
│       │   └── streamlit_app.py       # Streamlit main app
│       │
│       ├── steps/                     # Pipeline step scripts
│       │   ├── __init__.py
│       │   ├── download_kaggle_data.py    # Step 1: Download data
│       │   ├── etl_pipeline.py            # Step 2: Extract/Transform/Load
│       │   ├── train_models.py             # Step 3: Train ML models
│       │   └── compute_metrics.py          # Step 4: Compute metrics
│       │
│       ├── scripts/                   # Analysis scripts
│       │   ├── __init__.py
│       │   └── analytics/             # Analytics modules
│       │       ├── __init__.py
│       │       ├── ab_test_analysis.py
│       │       ├── statistical_tests.py
│       │       ├── regression_analysis.py
│       │       ├── causal_inference.py
│       │       ├── time_series_forecasting.py
│       │       ├── cate_meta_learners.py
│       │       ├── data_quality.py
│       │       └── experiment_tracker.py
│       │
│       ├── sql/                       # SQL scripts
│       │   ├── clean_tickets.sql
│       │   ├── standardize_taxonomy.sql
│       │   ├── ticket_metrics.sql
│       │   └── query_optimization.sql
│       │
│       ├── core/                      # Core shared modules
│       │   ├── __init__.py
│       │   ├── path.py               # Path management
│       │   └── logger.py             # Logging utilities
│       │
│       └── utils/                     # Utility functions
│           ├── __init__.py
│           ├── validators.py
│           └── formatters.py
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── unit/                          # Unit tests
│   └── integration/                   # Integration tests
│
├── docs/                              # Documentation
│   └── files_overview.md
│
├── configs/                           # Configuration files
│   ├── streamlit.toml
│   └── logging.yaml
│
├── data/                              # Data directory
│   ├── raw/                           # Raw data
│   └── processed/                     # Processed data
│
├── notebooks/                         # Jupyter notebooks
│
└── scripts/                           # DevOps scripts
    ├── setup_environment.sh
    ├── run_tests.sh
    └── deploy.sh
```

---

## ✨ Features

### 📊 Executive Dashboard
- Key Performance Indicators (KPIs) with business context
- Real-time support health monitoring
- Ticket volume trends
- CSAT score analysis
- First response time analysis
- Recent tickets list

### 👥 Agent Performance
- Top agents by CSAT and ticket volume
- Agent metrics comparison
- Performance trends

### 🔍 Issue Analytics
- Issue type comparison
- Resolution time by issue type
- SLA breach analysis

### ⏱️ SLA Monitoring
- Daily SLA breach rate
- Priority-based SLA analysis
- Breach ticket details

### 😊 CSAT Insights
- CSAT trend analysis
- Low CSAT rate monitoring
- CSAT by issue and priority

### 📈 Trend Analysis
- Multi-metric trend visualization
- Day-of-week patterns
- Business hours vs off-hours comparison

### 🧪 Experiment Analysis
- **Sample Size Calculator**: Plan experiments before running
- **A/B Test Analysis**: Complete statistical testing framework
- **Statistical Methods**:
  - t-tests (independent, paired)
  - Chi-square tests (proportion comparison)
  - Mann-Whitney U (non-parametric)
  - ANOVA (multiple groups)
- **Power Analysis**: Calculate and evaluate statistical power
- **Best Practices Guide**: Experiment design, analysis, and reporting

### ⏩ Time Series Forecasting
- Trend detection and forecasting
- Seasonality analysis
- ARIMA/ETS forecasting
- Anomaly detection (IQR, CUSUM)
- Forecast visualization with confidence intervals

### 🎯 CATE Analysis (Heterogeneous Treatment Effects)
- S-Learner (single model approach)
- T-Learner (two model approach)
- X-Learner (cross-fitting) - recommended
- DR-Learner (doubly robust)
- Heterogeneous effect analysis and visualization

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Frontend/Visualization | Streamlit + Plotly | 1.28+ / 5.15+ |
| Data Storage | CSV + DuckDB | - |
| Machine Learning | scikit-learn | 1.2+ |
| Statistical Analysis | SciPy | 1.10+ |
| Data Processing | Pandas + NumPy | 2.0+ / 1.24+ |
| Experiment Framework | Custom A/B testing module | - |

---

## 📊 Statistical Methodology

This project implements a complete statistical analysis framework:

### Hypothesis Testing Methods

| Method | Use Case | Implementation |
|--------|----------|----------------|
| **t-test (Welch's)** | Two-group mean comparison | `ab_test_analysis.py`, `statistical_tests.py` |
| **Paired t-test** | Before/after comparison | `statistical_tests.py` |
| **Chi-square test** | Proportion comparison | `ab_test_analysis.py`, `statistical_tests.py` |
| **Mann-Whitney U** | Non-parametric two-group | `ab_test_analysis.py`, `statistical_tests.py` |
| **Wilcoxon** | Non-parametric paired | `statistical_tests.py` |
| **ANOVA** | Multi-group mean comparison | `statistical_tests.py` |
| **Kruskal-Wallis** | Non-parametric multi-group | `statistical_tests.py` |

### Effect Size Metrics

- **Cohen's d**: t-test effect size (0.2=small, 0.5=medium, 0.8=large)
- **Cramer's V**: Chi-square effect size
- **Pearson r / Spearman ρ**: Correlation coefficients
- **Eta-squared (η²)**: ANOVA effect size

### Confidence Intervals

- 95% confidence level (α = 0.05)
- Mean difference intervals
- Proportion difference intervals
- Fisher z-transform intervals for correlations

### Causal Inference Methods

| Method | Use Case | Implementation |
|--------|----------|----------------|
| **Difference-in-Differences (DiD)** | Panel data causal effect | `causal_inference.py` |
| **Propensity Score Matching (PSM)** | Selection bias correction | `causal_inference.py` |
| **Instrumental Variables (IV)** | Endogeneity treatment | `causal_inference.py` |
| **Regression Discontinuity (RDD)** | Discontinuity design | `causal_inference.py` |
| **S-Learner** | Heterogeneous treatment effects | `cate_meta_learners.py` |
| **T-Learner** | Heterogeneous treatment effects | `cate_meta_learners.py` |
| **X-Learner** | Heterogeneous treatment effects | `cate_meta_learners.py` |
| **DR-Learner** | Doubly robust estimation | `cate_meta_learners.py` |

---

## 🚀 Quick Start

### Environment Requirements

- Python 3.8+
- pip (Python package manager)
- Virtual environment at project root `.venv`

### Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

**Using main.py (recommended):**
```bash
# Full pipeline
python main.py

# Clean and rerun
python main.py --rerun

# Run specific steps
python main.py --step 2    # ETL only
python main.py --step 3    # Training only
python main.py --launch-only  # Launch app only

# Data processing only (download + ETL)
python main.py --data-only
```

**CLI Help:**
```bash
python main.py --help
```

### Manual Steps

```bash
# Step 1: Download/generate data
python scripts/analytics/download_kaggle_data.py

# Step 2: Run ETL pipeline
python scripts/analytics/etl_pipeline.py

# Step 3: Train ML models
python scripts/analytics/train_models.py

# Step 4: Launch Streamlit app
streamlit run src/customer_support_analytics/app/streamlit_app.py
```

### Data Source

This project uses simulated data generation based on the real structure of Kaggle's Customer Support Ticket Dataset:
- Dataset reference: https://www.kaggle.com/suraj520/customer-support-ticket-dataset

---

## 🔄 Pipeline Flow

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
                                          ▼
                                 http://localhost:8501
```

---

## 🤖 Machine Learning Models

| Model | Use Case | Algorithm |
|-------|----------|-----------|
| Priority Prediction | Predict ticket priority | Random Forest |
| Routing Model | Classify issue type | Random Forest |
| CSAT Prediction | Predict CSAT score | Random Forest (regression) |
| SLA Breach Prediction | Predict SLA breach | Random Forest |

---

## 📈 Key Metrics

### Volume Metrics
- total_tickets, ticket_volume_by_channel, ticket_volume_by_priority

### Efficiency Metrics
- median_first_response_min, p75_resolution_min, solved_rate

### Experience Metrics
- avg_csat, low_csat_rate

### Risk Metrics
- sla_breach_rate, escalation_rate, reopened_count

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/ -k "TestABTestAnalysis" -v
```

---

## 📜 License

MIT License

---

## 📝 Disclaimer

This project is for educational and portfolio purposes only.
