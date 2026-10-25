#!/usr/bin/env bash
set -euo pipefail
python3 -m aria_esc.process_data.targets "$@"
