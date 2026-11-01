#!/usr/bin/env bash
# Generate the synthetic data, match up, build the report and keep only its HTML.
# Works in a scratch directory so oceanval's output folders never land in this repo.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-$(mktemp -d)}"
python "$HERE/make_synthetic_data.py"
cd "$WORK"
python "$HERE/matchup.py"
python "$HERE/validate.py"
python "$HERE/collect_report.py" "$WORK"
