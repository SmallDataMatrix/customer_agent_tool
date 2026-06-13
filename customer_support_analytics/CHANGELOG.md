# CHANGELOG - SupportIQ Web

## [Unreleased] - 2026-06-13

### Removed
- **run_pipeline.sh, run_pipeline.bat, run_pipeline copy.bat**: These pipeline runner scripts have been removed as they are now redundant. The `main.py` script provides the same and extended functionality with a unified CLI interface.

  **Migration**: Replace any calls to `run_pipeline.sh` with:
  ```bash
  # Full pipeline
  python main.py

  # With clean rerun
  python main.py --rerun

  # Specific step
  python main.py --step 2
  ```

## [Previous Releases]

### 2026-06-12 - Loop 4 Optimization
- Added main.py CLI wrapper with --rerun, --step, --launch-only, --data-only options
- Renamed all scripts to remove numeric prefixes
- Fixed module import issues
- Created comprehensive unit test framework

### 2026-06-12 - Loop 3 Enhancement
- Added CATE meta-learners (S-Learner, T-Learner, X-Learner, DR-Learner)
- Added experiment tracking framework
- Added SQL query optimization with indexes and materialized views
- Enhanced Streamlit with CATE Analysis page

### 2026-06-12 - Loop 2 Gap Analysis
- Added time series forecasting module
- Enhanced HTE implementation
- Added data quality monitoring framework
- Integrated forecasting into Streamlit dashboard

### 2026-06-12 - Loop 1 Initial Analysis
- Established base project structure
- Implemented core ETL pipeline
- Implemented ML model training
- Created 8-page Streamlit dashboard
