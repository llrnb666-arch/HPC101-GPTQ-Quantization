# HPC101-GPTQ-Quantization

> GPTQ post-training quantization (W4A16) and end-to-end inference optimization for Gemma4-12B on NVIDIA H800 MIG.

## Overview

This project implements GPTQ quantization and optimizes the inference pipeline for the Gemma4-12B large language model, running on a severely resource-constrained GPU partition (H800 MIG 1g.10gb: 14 SMs, 10GB VRAM). Both tasks achieved full marks.

| Task | Metric | Result | Target | Score |
|------|--------|--------|--------|-------|
| Accuracy (40%) | delta_nll | **0.0905** | < 0.10 | 100/100 |
| Performance (60%) | elapsed_s | **33.83s** | < 36s | 100/100 |

## Problem Statement

**Task 1 - GPTQ Quantization:** Compress Gemma4-12B weights from BF16 to INT4 using the GPTQ algorithm with second-order Hessian-based error compensation. Quality is measured by delta_nll (change in negative log-likelihood vs BF16 reference).

**Task 2 - Inference Optimization:** Process 10 generation requests (320 total decode tokens, prompts ranging 200-2000 tokens) as fast as possible. The 10GB MIG partition limits batch size to 2, making per-GEMM efficiency the primary optimization lever.

## Key Optimizations

### 1. Triton INT4 GEMM Kernel Rewrite (biggest win: -8.6s)

The original fused dequant-GEMM kernel split each packed byte into even/odd nibbles and performed two separate `tl.dot` calls with strided activation loads. I rewrote it to:

- Load activations contiguously in a single `tl.arange(0, BLOCK_K)` pass
- Combine low/high nibbles using `tl.interleave()` into a single weight tile
- Execute a single `tl.dot` instead of two

This improved decode GEMM throughput by 21% (98 to 119 GB/s) and prefill GEMM by 51-75%, directly translating to 8.6s end-to-end savings.

### 2. Static Batching (batch_size=2, -8.55s)

Packed 2 requests per batch to amortize weight reads across multiple sequences. batch_size=3 was infeasible due to 10GB VRAM constraint.

### 3. BFloat16 Decode Attention (-1.6s)

Switched decode-phase GQA attention from float32 to bfloat16 for mask, matmul, and softmax operations.

### 4. Chunked Scoring Attention Bug Fix (correctness)

Fixed a subtle bug where `is_causal=True` incorrectly masked prior chunk keys during quality evaluation's chunked scoring, causing delta_nll to measure 0.838 instead of the true 0.0905.

## Architecture

```
src/hpc101_infer/
  engine.py              # Inference engine (prefill/decode pipeline, batching)
  layers/
    triton_kernels.py    # Optimized fused dequant-GEMM kernel (tl.interleave)
    attention.py         # GQA attention with sliding window, bf16 decode path
    linear.py            # QuantizedLinear with Triton fallback
  quantization/
    methods/gptq.py      # GPTQ algorithm implementation
    pipeline.py          # Quantization pipeline orchestration
  runtime/kv_cache.py    # Pre-allocated KV cache
  scheduler/static_batch.py
```

## Reproduction

### Environment
- Python 3.13.5, PyTorch 2.13.0+cu132, Triton 3.7.1
- NVIDIA H800 PCIe MIG 1g.10gb (14 SMs, 10GB VRAM)

### Quantize
```bash
python scripts/quantize.py --config config.yaml \
  --model /checkpoints/gemma-4-12b --output /tmp/quant-gptq --device cuda --verbose
```

### Evaluate Quality
```bash
python scripts/evaluate_quality.py --model /tmp/quant-gptq \
  --dataset datasets/quality_public.jsonl \
  --output results/gptq-public-quality.json \
  --reference results/bf16-public-quality.json \
  --max-delta-nll 0.16 --linear-backend int4_reference \
  --device cuda --dtype bfloat16 --max-sequence-length 2048 --chunk-size 128
```

### Evaluate Performance
```bash
python scripts/run_generation_queue.py --config config.yaml \
  --model /tmp/quant-gptq \
  --input datasets/performance_public.jsonl \
  --output results/gptq-public-generation.jsonl \
  --summary-output results/gptq-public-summary.json
```

## Results

### Optimization Breakdown
| Optimization | elapsed_s | Delta |
|---|---|---|
| Baseline (batch=1, reference kernel) | 52.85s | - |
| + batch_size=2 | 44.3s | -8.55s |
| + bfloat16 decode attention | 42.7s | -1.6s |
| + Triton kernel rewrite (tl.interleave) | 34.1s | -8.6s |
| + chunked scoring fix (quality only) | 33.83s | ~0s |

### Quality by Sequence Length
| Bucket | Sequences | delta_nll |
|--------|-----------|-----------|
| 128 | 32 | 0.0835 |
| 256 | 16 | 0.0820 |
| 512 | 8 | 0.0915 |
| 1024 | 4 | 0.1032 |
| 2048 | 2 | 0.0922 |
| **Overall** | **62** | **0.0905** |

## Report

The full experiment report is in [report.pdf](report.pdf), with Typst source in [report/](report/).