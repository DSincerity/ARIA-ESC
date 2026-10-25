#!/usr/bin/env bash
set -euo pipefail
python3 -m aria_esc.train.sft "$@"
