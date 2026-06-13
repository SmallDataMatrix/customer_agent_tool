#!/usr/bin/env python3
"""
Customer Support Analytics - Main Pipeline Runner

This is a convenience wrapper that delegates to the actual pipeline runner
located at src/customer_support_analytics/app/main.py

Usage:
    python main.py                    # Run all steps sequentially
    python main.py --rerun           # Clean previous results and rerun all
    python main.py --step 1          # Run only step 1 (download)
    python main.py --step 2          # Run only step 2 (ETL)
    python main.py --step 3          # Run only step 3 (ML training)
    python main.py --step 4          # Run only step 4 (launch app)
    python main.py --launch-only      # Skip training, launch app only
    python main.py --help            # Show this help message
"""

import subprocess
import sys
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# The actual main.py is in the app directory
ACTUAL_MAIN = SCRIPT_DIR / "src" / "customer_support_analytics" / "app" / "main.py"

def main():
    """Run the actual main.py with the same arguments."""
    # Build the command
    cmd = [sys.executable, str(ACTUAL_MAIN)] + sys.argv[1:]
    
    # Run the command
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
