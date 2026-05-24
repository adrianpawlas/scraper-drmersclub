#!/bin/bash
#
# run_scraper.sh — Automated Drmers Club Scraper Runner
#
# This script is designed to be called by launchd (or cron) for
# scheduled runs. It can also be invoked manually at any time.
#
# Usage:
#   ./run_scraper.sh              # Run full pipeline
#   ./run_scraper.sh --limit 10   # Run with product limit (testing)
#   ./run_scraper.sh --skip-embedding  # Skip embeddings (faster)
#

set -euo pipefail

# --- Configuration ---
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG_FILE="$LOG_DIR/run_$TIMESTAMP.log"
DATA_DIR="$PROJECT_DIR/data"

# Ensure directories exist
mkdir -p "$LOG_DIR" "$DATA_DIR"

# --- Run the scraper ---
cd "$PROJECT_DIR"

{
    echo "=========================================="
    echo " Drmers Club Scraper — Automated Run"
    echo " Started at: $(date)"
    echo " Directory:  $PROJECT_DIR"
    echo "=========================================="
    echo ""

    # Check if Python 3 is available
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: python3 not found. Ensure Python is installed."
        exit 1
    fi

    # Log Python version
    echo "Python: $(python3 --version 2>&1)"
    echo ""

    # Run the scraper with all arguments passed through
    python3 main.py "$@"

    EXIT_CODE=$?
    echo ""
    echo "=========================================="
    echo " Run finished at: $(date)"
    echo " Exit code: $EXIT_CODE"
    echo "=========================================="

    exit $EXIT_CODE
} 2>&1 | tee "$LOG_FILE"

# Capture the exit code from the subshell (PIPESTATUS[0] is the exit code of the first command in the pipe)
EXIT_CODE=${PIPESTATUS[0]}

# Cleanup: keep only the last 30 log files
# macOS ls doesn't support the -v flag well, so use a portable approach
log_count=$( (ls -1 "$LOG_DIR"/run_*.log 2>/dev/null) | wc -l | tr -d ' ')
if [ "$log_count" -gt 30 ]; then
    (ls -t "$LOG_DIR"/run_*.log 2>/dev/null) | tail -n +31 | xargs -r rm
fi

exit $EXIT_CODE
