"""Quick correctness test for GPTQ implementation.

Run inside the lab5 container after `uv pip install -e . --no-deps`:
    python3 test_gptq.py
"""
import torch

from hpc101_infer.quantization.methods.gptq import quantize_weight_gptq
from hpc101_infer.quantization.methods.rtn import quantize_weight_rtn
from hpc101_infer.quantization.packing import dequantize_weight


def relative_error(original, quantized_weight, activations):
    deq = dequantize_weight(quantized_weight, dtype=torch.float32)
    orig_out = activations @ original.t()
    quant_out = activations @ deq.t()
    return (orig_out - quant_out).norm().item() / orig_out.norm().item()


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_features, in_features = 256, 512
    group_size = 128
    tokens = 512

    weight = torch.randn(out_features, in_features, device=device, dtype=torch.bfloat16) * 0.02
    activations = torch.randn(tokens, in_features, device=device, dtype=torch.bfloat16)

    rtn = quantize_weight_rtn(weight, group_size, symmetric=True, scale_dtype=torch.float16)
    rtn_err = relative_error(weight.float(), rtn, activations.float())

    gptq, meta = quantize_weight_gptq(
        weight, activations, group_size,
        block_size=128, damp_percent=0.01,
        symmetric=True, scale_dtype=torch.float16,
    )
    gptq_err = relative_error(weight.float(), gptq, activations.float())

    print(f"RTN  relative output error: {rtn_err:.6f}")
    print(f"GPTQ relative output error: {gptq_err:.6f}")
    print(f"GPTQ improvement:          {rtn_err / gptq_err:.2f}x")
    print(f"Metadata: {meta}")

    assert gptq.qweight.dtype == torch.uint8
    assert gptq.scales.shape == (out_features, in_features // group_size)
    assert gptq.zeros is None
    assert gptq.bits == 4
    assert gptq.packing == "uint8_little_nibble"
    assert gptq_err < rtn_err, f"GPTQ ({gptq_err}) should beat RTN ({rtn_err})"
    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
