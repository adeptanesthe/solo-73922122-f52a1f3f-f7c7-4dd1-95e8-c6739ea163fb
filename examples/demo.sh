#!/usr/bin/env bash
# Demo: inspect one INT8 conv layer for accumulator width safety.
# Usage: ./examples/demo.sh [base_url]     (default http://127.0.0.1:8000)
set -euo pipefail

BASE="${1:-http://127.0.0.1:8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pretty() { python3 -m json.tool; }

echo "== health =="
curl -s "$BASE/health" | pretty

echo
echo "== case 1: is a signed 16-bit accumulator safe? =="
curl -s -X POST "$BASE/v1/analyze" \
  -H 'Content-Type: application/json' \
  -d @"$HERE/request_ok.json" | pretty

echo
echo "== case 2: same layer, is 12-bit enough? (expect earliest overflow + repro inputs) =="
curl -s -X POST "$BASE/v1/analyze" \
  -H 'Content-Type: application/json' \
  -d @"$HERE/request_overflow.json" | pretty

echo
echo "== case 3: packed weight layout matches the reference (expect matches=true) =="
curl -s -X POST "$BASE/v1/layout-check" \
  -H 'Content-Type: application/json' \
  -d @"$HERE/layout_match.json" | pretty

echo
echo "== case 4: corrupted packed weight (expect first index + output coords + both sums) =="
curl -s -X POST "$BASE/v1/layout-check" \
  -H 'Content-Type: application/json' \
  -d @"$HERE/layout_mismatch.json" | pretty
