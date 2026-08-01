from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from hpc101_infer.quantization.packing import pack_int4
from hpc101_infer.quantization.types import (
    LayerContext,
    LayerQuantizationResult,
    QuantizedWeight,
    SCALE_DTYPES,
)


@dataclass(frozen=True)
class GPTQOptions:
    block_size: int
    damp_percent: float


@dataclass(frozen=True)
class GPTQModuleState:
    activations: torch.Tensor
    activation_tokens: int


@dataclass(frozen=True)
class GPTQLayerState:
    modules: Mapping[str, GPTQModuleState]
    options: GPTQOptions


def _calibration_option(
    calibration: Mapping[str, Any],
    name: str,
    default: int | float,
) -> int | float:
    gptq = calibration.get("gptq", calibration)
    if not isinstance(gptq, Mapping):
        raise TypeError("config.calibration['gptq'] must be a mapping")
    return gptq.get(name, default)


def _parse_gptq_options(calibration: Mapping[str, Any]) -> GPTQOptions:
    block_size = _calibration_option(calibration, "block_size", 128)
    damp_percent = _calibration_option(calibration, "damp_percent", 0.01)

    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("GPTQ block_size must be a positive integer")
    if (
        not isinstance(damp_percent, (int, float))
        or isinstance(damp_percent, bool)
        or not 0.0 < float(damp_percent) < 1.0
    ):
        raise ValueError("GPTQ damp_percent must be in (0, 1)")

    gptq = calibration.get("gptq", calibration)
    if isinstance(gptq, Mapping) and gptq.get("desc_act", False):
        raise ValueError("GPTQ desc_act is not supported by this checkpoint format")
    return GPTQOptions(
        block_size=block_size,
        damp_percent=float(damp_percent),
    )


def quantize_weight_gptq(
    weight: torch.Tensor,
    activations: torch.Tensor,
    group_size: int,
    *,
    block_size: int = 128,
    damp_percent: float = 0.01,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[QuantizedWeight, dict[str, float | int]]:
    """
    使用 GPTQ 算法将权重量化为 INT4。

    参数：
        weight: 待量化的权重张量，形状为 (out_features, in_features)。
        activations: 校准数据集的输入激活，形状为 (calibration_tokens, in_features)。
        group_size: 量化粒度，即每个 group 中的列数。
        block_size: 分块计算时每个 block 中的列数。
        damp_percent: 阻尼比例，用于改善 Hessian 的数值稳定性。
        symmetric: 是否使用对称量化。
        scale_dtype: 缩放因子的 dtype。

    返回：
        quantized_weight: 量化后的权重对象，具体详见 QuantizedWeight 类的定义。
        metadatas: 量化过程中的统计信息，不影响评测，用于分析和调试。
    """

    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point matrix")
    if activations.ndim != 2 or not activations.is_floating_point():
        raise ValueError("activations must be a floating-point matrix")
    if activations.shape[1] != weight.shape[1]:
        raise ValueError("activation width does not match weight input size")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0.0 < damp_percent < 1.0:
        raise ValueError("damp_percent must be in (0, 1)")
    if scale_dtype not in SCALE_DTYPES.values():
        raise ValueError("unsupported scale dtype")
    if not torch.isfinite(weight).all():
        raise ValueError("weight must contain only finite values")

    device = weight.device
    compute_dtype = torch.float32
    eps = torch.finfo(compute_dtype).eps

    # Work in float32 for numerical stability.
    W = weight.detach().to(device=device, dtype=compute_dtype).clone()
    X = activations.detach().to(device=device, dtype=compute_dtype)

    out_features, in_features = W.shape
    padded_in = math.ceil(in_features / group_size) * group_size
    if padded_in > in_features:
        W = F.pad(W, (0, padded_in - in_features))
        X = F.pad(X, (0, padded_in - in_features))

    num_groups = padded_in // group_size

    # --- Hessian H = X^T X ---
    H = X.t() @ X

    # Dead columns: input channels with no activation energy.
    dead = torch.diag(H) == 0
    dead_count = int(dead.sum().item())
    if dead.any():
        H[dead, :] = 0.0
        H[:, dead] = 0.0
        H[dead, dead] = 1.0

    # Damping for numerical stability of the Cholesky factorisation.
    diag_abs_mean = float(torch.diag(H).abs().mean())
    damp = damp_percent * diag_abs_mean
    H.diagonal().add_(damp)

    # --- Inverse-Hessian Cholesky factor: H^{-1} = U^T U ---
    # Retry with increasing damping if Cholesky fails (rank-deficient Hessian
    # is common when in_features >> calibration_tokens).
    for attempt in range(10):
        try:
            L = torch.linalg.cholesky(H)
            break
        except RuntimeError:
            extra = damp * (10 ** (attempt + 1))
            H.diagonal().add_(extra)
    else:
        raise RuntimeError("Cholesky failed after maximum damping retries")
    del H
    H_inv = torch.cholesky_inverse(L)
    del L
    # Numerical roundoff can make the explicitly formed inverse very slightly
    # asymmetric/non-PD even when H was factorized successfully. Symmetrize and
    # add a tiny adaptive jitter before the upper Cholesky used by GPTQ.
    H_inv = 0.5 * (H_inv + H_inv.t())
    inv_diag_mean = torch.diag(H_inv).abs().mean().clamp_min(eps)
    for attempt in range(8):
        try:
            U = torch.linalg.cholesky(H_inv, upper=True)
            break
        except RuntimeError:
            jitter = inv_diag_mean * (10.0 ** (attempt - 7))
            H_inv.diagonal().add_(jitter)
    else:
        U = torch.linalg.cholesky(H_inv + inv_diag_mean * 1e-2 * torch.eye(padded_in, device=device, dtype=compute_dtype), upper=True)
    del H_inv

    # --- Per-group quantisation parameters (same strategy as RTN) ---
    W_groups = W.view(out_features, num_groups, group_size)
    if symmetric:
        scales = W_groups.abs().amax(dim=-1) / 7.0
        scales = scales.clamp_min(eps)
        zeros = None
    else:
        minimum = W_groups.amin(dim=-1)
        maximum = W_groups.amax(dim=-1)
        scales = ((maximum - minimum) / 15.0).clamp_min(eps)
        zeros = torch.round(-minimum / scales).clamp(0, 15).to(torch.uint8)

    # --- Column-wise GPTQ with block-batched error propagation ---
    Q_encoded = torch.zeros(out_features, padded_in, dtype=torch.uint8, device=device)
    total_sq_err = torch.zeros((), dtype=compute_dtype, device=device)

    for i in range(0, padded_in, block_size):
        block_end = min(i + block_size, padded_in)
        block_width = block_end - i
        errs = torch.zeros(out_features, block_width, dtype=compute_dtype, device=device)

        for j in range(i, block_end):
            group_idx = j // group_size
            scale = scales[:, group_idx]
            w_col = W[:, j]

            if symmetric:
                q_val = torch.clamp(torch.round(w_col / scale), -8, 7)
                q_dequant = q_val * scale
                Q_encoded[:, j] = (q_val.to(torch.int16) + 8).to(torch.uint8)
            else:
                z = zeros[:, group_idx].to(compute_dtype)
                q_val = torch.clamp(torch.round(w_col / scale) + z, 0, 15)
                q_dequant = scale * (q_val - z)
                Q_encoded[:, j] = q_val.to(torch.uint8)

            total_sq_err += (w_col - q_dequant).square().sum()
            err = (w_col - q_dequant) / U[j, j]
            errs[:, j - i] = err

            # Propagate error to remaining columns inside the block.
            if j + 1 < block_end:
                W[:, j + 1 : block_end] -= err.unsqueeze(1) * U[j, j + 1 : block_end].unsqueeze(0)

        # Batch-update all columns after the block in one matmul.
        if block_end < padded_in:
            W[:, block_end:] -= errs @ U[i:block_end, block_end:]

    quantized = QuantizedWeight(
        qweight=pack_int4(Q_encoded),
        scales=scales.to(scale_dtype),
        zeros=zeros,
        original_shape=(out_features, in_features),
        padded_shape=(out_features, padded_in),
        bits=4,
        group_size=group_size,
        symmetric=symmetric,
        packing="uint8_little_nibble",
    )

    metadata: dict[str, float | int] = {
        "activation_tokens": X.shape[0],
        "block_size": block_size,
        "damp_percent": damp_percent,
        "dead_columns": dead_count,
        "predicted_loss": float((total_sq_err / (out_features * padded_in)).item()),
    }
    return quantized, metadata


class GPTQQuantizationMethod:
    name = "gptq"
    version = "1"

    def calibrate_layer(self, context: LayerContext) -> GPTQLayerState:
        if context.activations is None:
            raise ValueError("GPTQ requires calibration activations")

        options = _parse_gptq_options(context.config.calibration or {})
        modules = dict(context.layer.named_modules())
        states: dict[str, GPTQModuleState] = {}
        for name in context.target_modules:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            if name not in context.activations:
                raise ValueError(f"missing calibration activations for {name}")

            activations = context.activations[name]
            if activations.ndim != 2 or activations.shape[1] != module.in_features:
                raise ValueError(
                    f"invalid calibration activation shape for {name}: "
                    f"expected [tokens, {module.in_features}], "
                    f"got {tuple(activations.shape)}"
                )
            if activations.shape[0] == 0:
                raise ValueError(f"calibration activations for {name} are empty")
            if not activations.is_floating_point():
                raise TypeError(f"calibration activations for {name} must be floating")
            if not torch.isfinite(activations).all():
                raise ValueError(f"calibration activations for {name} are not finite")

            states[name] = GPTQModuleState(
                activations=activations.detach(),
                activation_tokens=activations.shape[0],
            )

        return GPTQLayerState(modules=states, options=options)

    def quantize_layer(
        self, context: LayerContext, state: GPTQLayerState
    ) -> LayerQuantizationResult:
        if not isinstance(state, GPTQLayerState):
            raise TypeError("state must be a GPTQLayerState")

        scale_dtype = SCALE_DTYPES[context.config.scale_dtype]
        modules = dict(context.layer.named_modules())
        weights: dict[str, QuantizedWeight] = {}
        module_metadata: dict[str, dict[str, float | int]] = {}

        for name in context.target_modules:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            module_state = state.modules.get(name)
            if module_state is None:
                raise ValueError(f"missing GPTQ calibration state for {name}")

            try:
                weights[name], module_metadata[name] = quantize_weight_gptq(
                    module.weight,
                    module_state.activations,
                    context.config.group_size,
                    block_size=state.options.block_size,
                    damp_percent=state.options.damp_percent,
                    symmetric=context.config.symmetric,
                    scale_dtype=scale_dtype,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"GPTQ failed for layer {context.layer_index} module {name} "
                    f"with {module_state.activation_tokens} calibration tokens: {error}"
                ) from error

        return LayerQuantizationResult(
            weights=weights,
            metadata={
                "gptq": {
                    "block_size": state.options.block_size,
                    "damp_percent": state.options.damp_percent,
                    "modules": module_metadata,
                }
            },
        )
