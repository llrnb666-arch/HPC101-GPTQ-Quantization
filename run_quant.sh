#!/bin/bash
set -e
cd ~/hpc101/src/lab5
uv pip install -e . --no-deps 2>&1 | tail -3
python3 scripts/quantize.py --config config.yaml --model /checkpoints/gemma-4-12b --output /tmp/quant-gptq --device cuda 2>&1
echo "QUANT_EXIT=$?"
ls -la /tmp/quant-gptq/ 2>&1 | head -10
