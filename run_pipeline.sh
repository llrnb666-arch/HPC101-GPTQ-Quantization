#!/bin/bash
set -x
cd ~/hpc101/src/lab5
uv pip install -e . --no-deps 2>&1 | tail -3

echo "QUANT_START"
python3 scripts/quantize.py \
  --config config.yaml \
  --model /checkpoints/gemma-4-12b \
  --output /tmp/quant-gptq \
  --device cuda \
  --calibration-micro-batch-size 16 \
  --calibration-limit 128 \
  --max-calibration-tokens 2048 \
  2>&1
QUANT_EXIT=$?
echo "QUANT_EXIT=$QUANT_EXIT"

if [ $QUANT_EXIT -ne 0 ]; then
  echo "QUANTIZATION FAILED, exiting"
  exit 1
fi

echo "COPY_TO_PERSISTENT"
rm -rf ~/hpc101/quant-gptq
cp -r /tmp/quant-gptq ~/hpc101/quant-gptq
ls -la ~/hpc101/quant-gptq/

echo "QUALITY_START"
python3 scripts/evaluate_quality.py \
  --model /tmp/quant-gptq \
  --dataset datasets/quality_public.jsonl \
  --output results/gptq-public-quality.json \
  --reference results/bf16-public-quality.json \
  --max-delta-nll 0.16 \
  --linear-backend int4_reference \
  --device cuda \
  --dtype bfloat16 \
  --max-sequence-length 2048 \
  --chunk-size 128 \
  2>&1
echo "QUALITY_EXIT=$?"

echo "PERFORMANCE_START"
python3 scripts/run_generation_queue.py \
  --config config.yaml \
  --model /tmp/quant-gptq \
  --input datasets/performance_public.jsonl \
  --output results/gptq-public-generation.jsonl \
  --summary-output results/gptq-public-summary.json \
  2>&1
echo "PERFORMANCE_EXIT=$?"

echo "PIPELINE_DONE"
cat results/gptq-public-quality.json 2>&1 | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps({k:d[k] for k in ['mean_nll','delta_nll','passed'] if k in d}, indent=2))" 2>/dev/null || true
cat results/gptq-public-summary.json 2>&1 || true
