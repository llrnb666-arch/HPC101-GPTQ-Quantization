try:
    import triton
    print("TRITON_VERSION", triton.__version__)
except ImportError:
    print("TRITON_NOT_AVAILABLE")
try:
    from hpc101_infer.layers.triton_kernels import fused_dequant_gemm
    print("KERNEL_IMPORT_OK")
except Exception as e:
    print("KERNEL_IMPORT_FAIL", e)
try:
    from hpc101_infer.layers.linear import _TRITON_AVAILABLE
    print("LINEAR_TRITON_AVAILABLE", _TRITON_AVAILABLE)
except Exception as e:
    print("LINEAR_IMPORT_FAIL", e)
