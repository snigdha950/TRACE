#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python test_alert_api.py
exec uvicorn trace_alert_api:app --host 0.0.0.0 --port "${PORT:-8000}"
