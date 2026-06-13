#!/usr/bin/env python3
"""
Customer Support Analytics - Main Pipeline Runner

This script provides a unified interface to run all pipeline steps
with support for clean/rerun functionality.

Usage:
    python main.py                    # Run all steps sequentially
    python main.py --rerun            # Clean previous results and rerun all
    python main.py --step 1           # Run only step 1 (download)
    python main.py --step 2           # Run only step 2 (ETL)
    python main.py --step 3           # Run only step 3 (ML training)
    python main.py --step 4           # Run only step 4 (launch app)
    python main.py --launch-only       # Skip training, launch app only
    python main.py --help             # Show this help message

Steps:
    1. Download data from Kaggle
    2. Run ETL pipeline
    3. Train ML models
    4. Launch Streamlit application
"""

import argparse
import shutil
import sys
from pathlib import Path

# Resolve project root: src/customer_support_analytics/app -> src/customer_support_analytics -> src -> project root
APP_ROOT = Path(__file__).parent  # src/customer_support_analytics/app
PACKAGE_ROOT = APP_ROOT.parent    # src/customer_support_analytics
PROJECT_ROOT = PACKAGE_ROOT.parent.parent  # project root (customer_support_analytics)
sys.path.insert(0, str(PACKAGE_ROOT))

# Data and models directories relative to project root
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "trained_model"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SupportIQ Web Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Run all steps
    python main.py --rerun            # Clean and rerun all steps
    python main.py --step 2           # Run only ETL
    python main.py --launch-only      # Launch app without retraining
        """
    )

    parser.add_argument(
        "--rerun", "-r",
        action="store_true",
        help="Clean previous results before running (removes processed data and models)"
    )

    parser.add_argument(
        "--step", "-s",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run only a specific step (1=download, 2=ETL, 3=train, 4=launch)"
    )

    parser.add_argument(
        "--launch-only", "-l",
        action="store_true",
        help="Skip data download and training, launch app only"
    )

    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Run data steps only (download + ETL), skip training"
    )

    return parser.parse_args()


def clean_previous_results():
    """Remove processed data and models to enable clean rerun."""
    print("\n" + "=" * 60)
    print("CLEANING PREVIOUS RESULTS")
    print("=" * 60)

    processed_dir = DATA_DIR / "processed"
    models_dir = MODELS_DIR

    cleaned = []

    if processed_dir.exists():
        for f in processed_dir.glob("*"):
            if f.is_file():
                f.unlink()
                cleaned.append(str(f))
        print(f"Cleaned: {processed_dir}")

    if models_dir.exists():
        for f in models_dir.glob("*.pkl"):
            f.unlink()
            cleaned.append(str(f))
        print(f"Cleaned: {models_dir}")

    if cleaned:
        print(f"Removed {len(cleaned)} files/directories")
    else:
        print("Nothing to clean")

    print()


def run_step(step_num, python_path):
    """Run a specific pipeline step."""
    steps = {
        1: {
            "name": "Download Kaggle Dataset",
            "script": "steps/download_kaggle_data.py",
            "success": "Kaggle Data Downloader completed successfully",
            "error": "Failed to download data"
        },
        2: {
            "name": "Run ETL Pipeline",
            "script": "steps/etl_pipeline.py",
            "success": "ETL Pipeline completed successfully",
            "error": "ETL Pipeline failed"
        },
        3: {
            "name": "Train ML Models",
            "script": "steps/train_models.py",
            "success": "ML Model Training Pipeline completed successfully",
            "error": "Model training failed"
        }
    }

    if step_num not in steps:
        print(f"Error: Invalid step {step_num}")
        return False

    step = steps[step_num]
    script_path = PACKAGE_ROOT / step["script"]

    print("\n" + "=" * 60)
    print(f"Step {step_num}/3: {step['name']}")
    print("=" * 60)

    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return False

    # Run the script
    result = subprocess_run([python_path, str(script_path)])

    if result.returncode == 0:
        print(f"\n✅ {step['success']}")
        return True
    else:
        print(f"\n❌ {step['error']}")
        return False


def subprocess_run(cmd):
    """Run a command using subprocess."""
    import subprocess
    return subprocess.run(cmd, cwd=str(PACKAGE_ROOT))


def run_download(python_path):
    """Step 1: Download data."""
    return run_step(1, python_path)


def run_etl(python_path):
    """Step 2: Run ETL pipeline."""
    return run_step(2, python_path)


def run_training(python_path):
    """Step 3: Train ML models."""
    return run_step(3, python_path)


def launch_app(streamlit_path):
    """Step 4: Launch Streamlit application."""
    print("\n" + "=" * 60)
    print("Step 4/4: Launching Streamlit App")
    print("=" * 60)

    app_path = APP_ROOT / "streamlit_app.py"

    if not app_path.exists():
        print(f"Error: App not found: {app_path}")
        return False

    print(f"\nLaunching Streamlit app: {app_path}")
    print("Press Ctrl+C to stop\n")

    # Launch streamlit
    result = subprocess_run([streamlit_path, "run", str(app_path)])

    return result.returncode == 0


def find_python():
    """Find the Python executable in virtual environment."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"

    if venv_python.exists():
        return str(venv_python)

    # Fall back to system python
    return sys.executable


def find_streamlit():
    """Find the Streamlit executable."""
    # Primary: Use venv streamlit (at project root level or parent)
    possible_venvs = [
        PROJECT_ROOT / ".venv" / "bin" / "streamlit",
        PROJECT_ROOT.parent / ".venv" / "bin" / "streamlit",  # parent of customer_support_analytics
    ]
    for venv_streamlit in possible_venvs:
        if venv_streamlit.exists():
            return str(venv_streamlit)

    # Secondary: Try to find in PATH
    system_streamlit = shutil.which("streamlit")
    if system_streamlit and Path(system_streamlit).exists():
        return system_streamlit

    # Tertiary: Try common locations
    for p in ["/opt/homebrew/bin/streamlit", "/usr/local/bin/streamlit"]:
        if Path(p).exists():
            return p

    # Last resort
    return "streamlit"


def main():
    """Main entry point."""
    args = parse_args()

    python_path = find_python()
    streamlit_path = find_streamlit()

    print("\n" + "=" * 60)
    print("Customer Support Analytics - Pipeline Runner")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {python_path}")
    print(f"Streamlit: {streamlit_path}")

    # Handle --rerun
    if args.rerun:
        clean_previous_results()

    # Handle --step
    if args.step:
        step = args.step
        if step == 4:
            launch_app(streamlit_path)
        else:
            success = run_step(step, python_path)
            sys.exit(0 if success else 1)
        return

    # Handle --launch-only
    if args.launch_only:
        print("\n" + "=" * 60)
        print("LAUNCH MODE: Skipping download and training")
        print("=" * 60)
        launch_app(streamlit_path)
        return

    # Handle --data-only
    if args.data_only:
        print("\n" + "=" * 60)
        print("DATA MODE: Running download and ETL only")
        print("=" * 60)
        if not run_download(python_path):
            sys.exit(1)
        if not run_etl(python_path):
            sys.exit(1)
        print("\n✅ Data pipeline completed successfully!")
        return

    # Default: Run all steps
    print("\n" + "=" * 60)
    print("FULL PIPELINE: Running all steps")
    print("=" * 60)

    # Step 1: Download
    if not run_download(python_path):
        sys.exit(1)

    # Step 2: ETL
    if not run_etl(python_path):
        sys.exit(1)

    # Step 3: Training
    if not run_training(python_path):
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

    # Step 4: Launch
    launch_app(streamlit_path)


if __name__ == "__main__":
    main()
