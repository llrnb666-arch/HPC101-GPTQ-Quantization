"""Triton fused dequantization-GEMM kernel for INT4 quantized weights.

Computes C = A @ dequantize(W)^T without materializing the full BF16 weight.
Packed INT4 weights are unpacked and scaled on-the-fly inside the GEMM loop.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_dequant_gemm_kernel(
    a_ptr, qweight_ptr, scales_ptr, zeros_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_qn, stride_qk,
    stride_sn, stride_sg, stride_zn, stride_zg,
    stride_cm, stride_cn,
    use_bias: tl.constexpr, symmetric: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    HALF_K: tl.constexpr = BLOCK_K // 2
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=a_mask, other=0.0)
        qw_offs = k_start // 2 + tl.arange(0, HALF_K)
        qw_mask = (offs_n[:, None] < N) & (qw_offs[None, :] < K // 2)
        packed = tl.load(qweight_ptr + offs_n[:, None] * stride_qn + qw_offs[None, :] * stride_qk,
                         mask=qw_mask, other=0)
        packed_i32 = packed.to(tl.int32)
        low = (packed_i32 & 0x0F).to(tl.float32)
        high = ((packed_i32 >> 4) & 0x0F).to(tl.float32)
        group_idx = k_start // GROUP_SIZE
        scale = tl.load(scales_ptr + offs_n * stride_sn + group_idx * stride_sg,
                        mask=offs_n < N, other=0.0).to(tl.float32)
        if symmetric:
            low = low - 8.0
            high = high - 8.0
        else:
            zp = tl.load(zeros_ptr + offs_n * stride_zn + group_idx * stride_zg,
                         mask=offs_n < N, other=0.0).to(tl.float32)
            low = low - zp[:, None]
            high = high - zp[:, None]
        w_low = (low * scale[:, None])
        w_high = (high * scale[:, None])
        w = tl.interleave(w_low, w_high).to(a.dtype)
        acc = tl.dot(a, tl.trans(w), acc=acc)
    if use_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(c_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_dequant_gemm(inputs, qweight, scales, group_size, zeros=None, bias=None, symmetric=True):
    M, K = inputs.shape
    N = qweight.shape[0]
    inputs = inputs.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    if zeros is not None:
        zeros = zeros.contiguous()
    output = torch.empty(M, N, device=inputs.device, dtype=inputs.dtype)
    if M <= 16:
        BLOCK_M, BLOCK_N = 16, 128
    else:
        BLOCK_M, BLOCK_N = 64, 128
    BLOCK_K = group_size
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _fused_dequant_gemm_kernel[grid](
        inputs, qweight, scales, zeros if zeros is not None else inputs,
        bias if bias is not None else inputs, output, M, N, K,
        inputs.stride(0), inputs.stride(1), qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0) if zeros is not None else 0, zeros.stride(1) if zeros is not None else 0,
        output.stride(0), output.stride(1),
        use_bias=bias is not None, symmetric=symmetric,
        GROUP_SIZE=group_size, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return output