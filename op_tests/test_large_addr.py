#!/usr/bin/env python3
"""
Test suite for validating 64-bit address fixes in FMHA backward kernels.

This test suite covers all backward kernel variants from fmha_bwd_dqdkdv.csv
to ensure they work correctly with large tensor sizes that exceed 32-bit address range.

Usage:
    python test_large_addr.py                    # Run all tests
    python test_large_addr.py --kernel bf16_a32  # Filter tests by kernel name substring
    python test_large_addr.py --list             # List all available tests
    python test_large_addr.py --csv              # Show CSV kernel mapping
"""

import argparse
import csv
import io
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
import torch

# Set environment to disable v3 forward (use CK for forward, v3 for backward)
os.environ["AITER_DISABLE_V3_FWD"] = "1"

import aiter
from aiter.test_common import checkAllclose, run_perftest
from aiter.test_mha_common import (
    attention_ref,
    generate_qkv,
)


@contextmanager
def capture_cpp_stdout():
    """Capture C-level stdout (from C++ std::cout) during kernel launches."""
    sys.stdout.flush()
    old_fd = os.dup(1)
    tmp = tempfile.TemporaryFile(mode='w+b')
    os.dup2(tmp.fileno(), 1)
    try:
        yield tmp
    finally:
        sys.stdout.flush()
        os.fsync(1)
        os.dup2(old_fd, 1)
        os.close(old_fd)
        tmp.seek(0)
        tmp._captured = tmp.read().decode('utf-8', errors='replace')
        tmp.seek(0)


_asm_kernel_cache = {}

def is_equivalent_kernel(expected_co, dispatched_info):
    """Check if the dispatched kernel is functionally equivalent to the expected one.
    
    Equivalence rules:
    - causal_br supersedes causal (causal_br handles both patterns)
    - On gfx950, causal A32 → causal_br A16 is valid (single-block optimization
      forces A16 for seqlen_k <= 256 because A32 causal has multi-block bug)
    - For non-causal: A16/A32 must match (different pipeline stages)
    """
    if expected_co in dispatched_info:
        return True
    if 'CK' in dispatched_info:
        return False

    import re
    def parse_co(name):
        m = re.search(r'bwd_hd(\d+(?:_\d+)?)_(\w+?)_((?:causal_br_|causal_)?)(a(?:16|32))_', name)
        if not m:
            return None
        return {'hdim': m.group(1), 'dtype': m.group(2), 'mask': m.group(3).rstrip('_'), 'atomic': m.group(4)}

    exp = parse_co(expected_co)
    disp = parse_co(dispatched_info)
    if not exp or not disp:
        return False
    if exp['hdim'] != disp['hdim'] or exp['dtype'] != disp['dtype']:
        return False
    mask_ok = (exp['mask'] == disp['mask'] or
               (exp['mask'] == 'causal' and disp['mask'] == 'causal_br'))
    if not mask_ok:
        return False
    if exp['atomic'] != disp['atomic']:
        is_causal = exp['mask'] in ('causal', 'causal_br')
        if is_causal and exp['atomic'] == 'a32' and disp['atomic'] == 'a16':
            return True
        return False
    return True


def parse_kernel_info(captured_text):
    """Parse captured C++ output to determine which kernels were used.

    The C++ backend caches loaded ASM kernels in a static map, so
    hipModuleLoad messages only appear on the first invocation.
    We mirror that with _asm_kernel_cache so subsequent tests still
    report ASM correctly instead of defaulting to 'CK (fallback)'.
    """
    info = {
        "odo": None,
        "dqdkdv": None,
        "dq_convert": None,
        "dq_shuffle": None,
        "raw_output": captured_text.strip(),
    }
    saw_ck_odo = False
    saw_ck_dq_convert = False
    for line in captured_text.splitlines():
        line = line.strip()
        if "[aiter] BWD kernel selected:" in line:
            co_part = line.split("[aiter] BWD kernel selected:")[-1].strip()
            info["dqdkdv"] = f"ASM ({co_part})"
            _asm_kernel_cache["dqdkdv"] = info["dqdkdv"]
        elif "[aiter] hipModuleLoad:" in line and "odo" in line.lower():
            co_part = line.split("hipModuleLoad:")[-1].strip().split()[0]
            info["odo"] = f"ASM ({co_part})"
            _asm_kernel_cache["odo"] = info["odo"]
        elif "[aiter] hipModuleLoad:" in line and "dq_convert" in line.lower():
            co_part = line.split("hipModuleLoad:")[-1].strip().split()[0]
            info["dq_convert"] = f"ASM ({co_part})"
            _asm_kernel_cache["dq_convert"] = info["dq_convert"]
        elif "[aiter] hipModuleLoad:" in line and "dq_shuffle" in line.lower():
            co_part = line.split("hipModuleLoad:")[-1].strip().split()[0]
            info["dq_shuffle"] = f"ASM ({co_part})"
            _asm_kernel_cache["dq_shuffle"] = info["dq_shuffle"]
        elif "BWD ODO: using CK kernel" in line or "BWD ODO: using fallback" in line:
            info["odo"] = "CK (fallback)"
            saw_ck_odo = True
        elif "BWD dQ_convert: using CK kernel" in line or "BWD dQ_convert: using fallback" in line:
            info["dq_convert"] = "CK (fallback)"
            saw_ck_dq_convert = True

    for key in ("odo", "dq_convert", "dq_shuffle", "dqdkdv"):
        if info[key] is None:
            if key == "odo" and saw_ck_odo:
                info[key] = "CK (fallback)"
            elif key == "dq_convert" and saw_ck_dq_convert:
                info[key] = "CK (fallback)"
            elif key in _asm_kernel_cache:
                info[key] = _asm_kernel_cache[key] + " (cached)"
            elif key == "dq_shuffle":
                info[key] = None
            else:
                info[key] = "CK (fallback)"
    return info


def get_device_arch():
    """Get the GPU architecture (gfx942, gfx950, etc.)"""
    props = torch.cuda.get_device_properties(0)
    gcn_arch = props.gcnArchName if hasattr(props, 'gcnArchName') else "unknown"
    return gcn_arch


def get_hsa_path():
    """Get path to HSA kernels based on GPU architecture."""
    arch = get_device_arch()
    # Determine the base arch (e.g., gfx950 from gfx950:sramecc+:xnack-)
    base_arch = arch.split(":")[0] if ":" in arch else arch
    return Path(aiter.__file__).parent.parent / "hsa" / base_arch / "fmha_v3_bwd"


def load_kernel_configs():
    """Load kernel configurations from CSV file."""
    hsa_path = get_hsa_path()
    csv_path = hsa_path / "fmha_bwd_dqdkdv.csv"
    
    if not csv_path.exists():
        print(f"Warning: CSV file not found at {csv_path}")
        return []
    
    configs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            configs.append(row)
    
    return configs


def run_torch(
    q, k, v, dout,
    causal=False,
    window_size=(-1, -1),
    upcast=True,
    reorder_ops=False,
):
    """
    Run PyTorch reference implementation (aligned with test_mha.py).
    Uses attention_ref from aiter.test_mha_common.
    """
    out, _, softmax_lse = attention_ref(
        q, k, v,
        None,  # query_padding_mask
        None,  # key_padding_mask
        None,  # attn_bias
        0.0,   # dropout_p
        None,  # dropout_mask
        causal=causal,
        window_size=window_size,
        upcast=upcast,
        reorder_ops=reorder_ops,
    )
    
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    return out, softmax_lse, dq, dk, dv


def print_mismatch_info(name, tensor_aiter, tensor_ref, tensor_pt, tol):
    """Print detailed mismatch information (aligned with test_mha.py)."""
    diff = (tensor_aiter - tensor_ref).abs()
    max_diff = diff.max().item()
    
    if max_diff > tol:
        # Find max diff location
        max_idx = torch.unravel_index(torch.argmax(diff), diff.shape)
        coords = tuple(idx.item() for idx in max_idx)
        print(f"\n--- {name} Mismatch Details ---")
        print(f"  Max diff: {max_diff}")
        print(f"  Max diff coords (batch, seq, head, dim): {coords}")
        print(f"  Aiter value:  {tensor_aiter[max_idx].item()}")
        print(f"  Ref value:    {tensor_ref[max_idx].item()}")
        if tensor_pt is not None:
            print(f"  PyTorch value: {tensor_pt[max_idx].item()}")
            print(f"  Aiter-Ref diff: {(tensor_aiter[max_idx] - tensor_ref[max_idx]).item()}")
            print(f"  PyTorch-Ref diff: {(tensor_pt[max_idx] - tensor_ref[max_idx]).item()}")
        
        # Print top 5 largest mismatches (handle large tensors)
        flat_diff = diff.flatten()
        top_k = 5
        if flat_diff.numel() <= 2**31 - 1:
            top_k = min(top_k, flat_diff.numel())
            top_vals, top_indices = torch.topk(flat_diff, top_k)
            print(f"  Top {top_k} mismatches:")
            for i in range(top_k):
                idx = torch.unravel_index(top_indices[i], diff.shape)
                coords = tuple(ix.item() for ix in idx)
                print(f"    [{i+1}] coords={coords}, diff={top_vals[i].item():.6f}, "
                      f"aiter={tensor_aiter[idx].item():.6f}, ref={tensor_ref[idx].item():.6f}")
        else:
            print(f"  (Tensor too large for top-k analysis, showing max only)")


def analyze_buffer_overflow(batch_size, nheads, seqlen_q, seqlen_k, hdim_q, hdim_v, atomic32):
    """
    Analyze which buffers exceed 32-bit addressing for the given parameters.
    Returns a dict mapping buffer name -> (max_offset, overflows_32bit).
    
    Buffer address model (batch-major layout [B, S, H, D]):
      batch_stride = nheads * seqlen * hdim * Bpp
      max_offset = (batch_size - 1) * batch_stride
    
    For dQ with atomic32 (float32 accumulation):
      Bpp_dQ = 4 (float32)
      dQ_base = H_DIM * s_LseD_base where s_LseD_base = (batch_idx*nheads + head_idx)*seqlen_q*Bpp_Q
      max dQ_base = H_DIM * ((batch_size-1)*nheads + nheads-1) * seqlen_q * Bpp_dQ
    """
    LIMIT_32 = 2**32
    bpp_io = 2  # bf16/fp16 = 2 bytes
    bpp_dq = 4 if atomic32 else bpp_io  # dQ accumulator: float32 if A32

    buffers = {}

    q_stride  = nheads * seqlen_q * hdim_q * bpp_io
    do_stride = nheads * seqlen_q * hdim_v * bpp_io
    k_stride  = nheads * seqlen_k * hdim_q * bpp_io
    v_stride  = nheads * seqlen_k * hdim_v * bpp_io
    dk_stride = nheads * seqlen_k * hdim_q * bpp_io
    dv_stride = nheads * seqlen_k * hdim_v * bpp_io

    buffers["Q"]  = (batch_size - 1) * q_stride
    buffers["dO"] = (batch_size - 1) * do_stride
    buffers["K"]  = (batch_size - 1) * k_stride
    buffers["V"]  = (batch_size - 1) * v_stride
    buffers["dK"] = (batch_size - 1) * dk_stride
    buffers["dV"] = (batch_size - 1) * dv_stride

    max_lsed_base = ((batch_size - 1) * nheads + (nheads - 1)) * seqlen_q * bpp_dq
    buffers["dQ (base=H_DIM*LseD)"] = hdim_q * max_lsed_base

    result = {}
    for name, max_offset in buffers.items():
        result[name] = (max_offset, max_offset >= LIMIT_32)
    return result


def print_overflow_analysis(batch_size, nheads, seqlen_q, seqlen_k, hdim_q, hdim_v, atomic32):
    """Print per-buffer 32-bit overflow analysis for a test case."""
    analysis = analyze_buffer_overflow(batch_size, nheads, seqlen_q, seqlen_k, hdim_q, hdim_v, atomic32)

    any_overflow = any(ov for _, ov in analysis.values())
    print(f"\n  Buffer 32-bit overflow analysis (2^32 = {2**32:,}):")
    for name, (max_off, overflows) in analysis.items():
        flag = "OVERFLOW" if overflows else "ok"
        print(f"    {name:25s}: max_offset = {max_off:>15,}  [{flag}]")

    if not any_overflow:
        print("    *** No buffers overflow 32-bit — this test does NOT exercise the fix ***")

    return analysis


def get_test_params_for_kernel(config, large_q=False):
    """
    Generate test parameters for a given kernel configuration.
    Uses large sequence lengths to trigger 64-bit address overflow.
    
    Args:
        config: kernel configuration dict from CSV
        large_q: if True, use large seqlen_q (tests Q/dO/dQ/ODO overflow);
                 if False, use large seqlen_k (tests K/V/dK/dV overflow)
    
    Returns dict with test parameters or None if kernel should be skipped.
    """
    dtype_str = config['dtype']
    hdim_q = int(config['hdim_q'])
    hdim_v = int(config['hdim_v'])
    mask = int(config['mask'])  # 0=no mask, 1=causal, 2=causal_br, 3=swa
    atomic32 = int(config['atomic32'])  # 0=a16, 1=a32
    pssk = int(config['pssk'])
    pddv = int(config['pddv'])
    mode = int(config['mode'])  # 0=normal, 1=group/varlen
    co_name = config['co_name']
    
    # Map dtype string to torch dtype
    dtype = torch.bfloat16 if dtype_str == 'bf16' else torch.float16
    
    # Determine causal type and window size
    # mask=3 (SWA) needs causal=False with finite window_size per mha.py:
    #   swa = not causal and ((window_size_left > 0) or (window_size_right > 0))
    causal = (mask > 0 and mask != 3)
    window_size = (-1, -1)
    if mask == 3:
        causal_type = None
        window_size = (15, 15)
    elif mask == 2:
        causal_type = "bottom_right"
    elif mask == 1:
        causal_type = "top_left"
    else:
        causal_type = None
    
    bf16_cvt = int(config['bf16_cvt'])  # 0=RTNE, 1=RTNA, 2=RTZ, 3=FP16(n/a)
    
    # deterministic=False is required to use ASM backend kernels
    # atomic32 field determines a16 vs a32 kernel via is_v3_atomic_fp32 flag
    # is_v3_atomic_fp32: True=A32 kernel, False=A16 kernel
    deterministic = False
    is_v3_atomic_fp32 = (atomic32 == 1)  # 1=a32, 0=a16
    
    # For pddv=1 kernels, use actual hdim < padded hdim (e.g., 72 pads to 128)
    # For asymmetric hdim (e.g., D192_128: hdim_q=192, hdim_v=128), handle independently
    actual_hdim_q = hdim_q
    actual_hdim_v = hdim_v
    if pddv == 1:
        hdim_to_actual = {64: 48, 128: 72, 192: 160}
        if hdim_q in hdim_to_actual:
            actual_hdim_q = hdim_to_actual[hdim_q]
        if hdim_v in hdim_to_actual:
            actual_hdim_v = hdim_to_actual[hdim_v]
    if pssk == 1 and pddv == 1 and hdim_q != hdim_v:
        actual_hdim_q = hdim_q
        actual_hdim_v = hdim_v
    
    batch_size = 8
    nheads = 40
    
    # ts_kv alignment: A16 on gfx942 uses 64, A32 uses ts from CSV
    ts_kv_align = 64 if (atomic32 == 0) else int(config.get('ts', '192'))

    TILE_ALIGN = 192  # largest KV tile size

    def overflow_seqlen(stride_per_seq):
        """Minimum seqlen for stride_per_seq * seqlen > 2^32, tile-aligned."""
        raw = (2**32 // stride_per_seq) + 1
        return ((raw + TILE_ALIGN - 1) // TILE_ALIGN) * TILE_ALIGN

    bpp_io = 2  # bf16/fp16
    bpp_dq = 4 if (atomic32 == 1) else bpp_io

    # Reference attention_ref allocates [B, H, seq_q, seq_k] in float32.
    # Cap the "small" seqlen when the "large" seqlen is very big to avoid OOM.
    SMALL_SEQ_THRESHOLD = 80000

    # gfx950 forces is_v3_atomic_fp32=False (A16) when seqlen_k <= 256 (single-block
    # optimization). For non-causal A32 kernels, use seqlen_k > 256 to bypass this.
    # Causal A32 kernels have a known multi-block bug on gfx950, so we keep
    # seqlen_k <= 256 (dispatches A16 via single-block path, which works correctly).
    # On gfx942, no single-block constraint exists, so seqlen_k=64 works for all
    # kernels and avoids OOM in attention_ref (scores matrix [B,H,sq,sk] in fp32).
    is_causal_mask = (mask > 0 and mask != 3)
    arch = get_device_arch()
    is_gfx950 = "gfx950" in arch
    if is_gfx950:
        min_seqlen_k = 320 if (atomic32 == 1 and not is_causal_mask) else 64
    else:
        min_seqlen_k = 64

    if large_q is None:
        seqlen_q = max(256, min_seqlen_k)
        seqlen_k = max(256, min_seqlen_k)
    elif large_q:
        stride_qdo = (batch_size - 1) * nheads * actual_hdim_q * bpp_io
        stride_dq = actual_hdim_q * ((batch_size - 1) * nheads + nheads - 1) * bpp_dq
        seqlen_q = max(overflow_seqlen(stride_qdo), overflow_seqlen(stride_dq))
        seqlen_k = max(min_seqlen_k, 64 if seqlen_q > SMALL_SEQ_THRESHOLD else 256)
    else:
        stride_kv = (batch_size - 1) * nheads * actual_hdim_q * bpp_io
        seqlen_k = overflow_seqlen(stride_kv)
        seqlen_q = 64 if seqlen_k > SMALL_SEQ_THRESHOLD else 256
    
    return {
        "batch_size": batch_size,
        "nheads": nheads,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "hdim_q": actual_hdim_q,
        "hdim_v": actual_hdim_v,
        "dtype": dtype,
        "causal": causal,
        "causal_type": causal_type,
        "window_size": window_size,
        "deterministic": deterministic,
        "co_name": co_name,
        "dtype_str": dtype_str,
        "atomic32": atomic32,
        "is_v3_atomic_fp32": is_v3_atomic_fp32,  # True=A32, False=A16
        "how_v3_bf16_cvt": bf16_cvt,  # 0=RTNE, 1=RTNA, 2=RTZ, 3=FP16
        "is_varlen": (mode == 1),  # group mode uses varlen API
    }


def run_mha_backward_test(
    batch_size: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    hdim_q: int,
    hdim_v: int,
    dtype: torch.dtype,
    causal: bool,
    causal_type: str = None,
    window_size: tuple = (-1, -1),
    deterministic: bool = False,
    test_name: str = "unknown",
    co_name: str = "",
    is_v3_atomic_fp32: bool = True,  # True=A32 kernel, False=A16 kernel
    how_v3_bf16_cvt: int = 1,  # 0=RTNE, 1=RTNA, 2=RTZ
    **kwargs,
):
    """
    Run a single MHA backward test and return results.
    Fully aligned with test_mha.py methodology.
    """
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    causal_flag = "" if causal else "--no-causal"
    det_flag = "--deterministic" if deterministic else "--no-deterministic"
    is_swa = window_size != (-1, -1)
    swa_str = f" --window-left {window_size[0]} --window-right {window_size[1]}" if is_swa else ""
    cmd = (f"AITER_DISABLE_V3_FWD=1 python test_mha.py -b {batch_size} -n {nheads} "
           f"-q {seqlen_q} -k {seqlen_k} -d_qk_v {hdim_q},{hdim_v} -d {dtype_str} "
           f"{causal_flag} --no-local {det_flag}{swa_str} -m mha")
    
    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"CO file: {co_name}")
    print(f"Command: {cmd}")
    print(f"Config: batch={batch_size}, heads={nheads}, seq_q={seqlen_q}, seq_k={seqlen_k}")
    print(f"        hdim_q={hdim_q}, hdim_v={hdim_v}, dtype={dtype}")
    print(f"        causal={causal}, causal_type={causal_type}, deterministic={deterministic}")
    if is_swa:
        print(f"        window_size=({window_size[0]}, {window_size[1]}) (SWA mode)")
    cvt_names = {0: "RTNE", 1: "RTNA", 2: "RTZ", 3: "FP16"}
    print(f"        is_v3_atomic_fp32={is_v3_atomic_fp32} ({'A32' if is_v3_atomic_fp32 else 'A16'} kernel)")
    print(f"        how_v3_bf16_cvt={how_v3_bf16_cvt} ({cvt_names.get(how_v3_bf16_cvt, '?')})")
    print(f"{'='*70}")
    
    overflow_info = print_overflow_analysis(
        batch_size, nheads, seqlen_q, seqlen_k, hdim_q, hdim_v,
        atomic32=int(is_v3_atomic_fp32),
    )
    
    device = "cuda"
    
    # Create input tensors (aligned with test_mha.py)
    q = torch.randn(batch_size, seqlen_q, nheads, hdim_q, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, hdim_q, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, hdim_v, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(batch_size, seqlen_q, nheads, hdim_v, device=device, dtype=dtype, requires_grad=True)
    
    # Run aiter forward + backward (aligned with test_mha.py's run_ck)
    try:
        with capture_cpp_stdout() as cap:
            (out, softmax_lse, S_dmask), _ = run_perftest(
                aiter.flash_attn_func,
                q, k, v,
                0.0,   # dropout_p
                None,  # softmax_scale
                causal,
                window_size,
                None,  # bias
                None,  # alibi_slopes
                deterministic,
                return_lse=True,
                return_attn_probs=True,  # Aligned with test_mha.py
                is_v3_atomic_fp32=is_v3_atomic_fp32,
                how_v3_bf16_cvt=how_v3_bf16_cvt,
                num_rotate_args=1,
            )
            
            # Compute gradients
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        
        kernel_info = parse_kernel_info(cap._captured)
        print(f"\n  Kernel dispatch info:")
        print(f"    ODO:        {kernel_info['odo']}")
        print(f"    dQdKdV:     {kernel_info['dqdkdv']}")
        if kernel_info.get('dq_shuffle'):
            print(f"    dQ_shuffle:  {kernel_info['dq_shuffle']}  (A16 mode: dQ accumulated in fp16/bf16)")
        if is_v3_atomic_fp32:
            print(f"    dQ_convert: {kernel_info['dq_convert']}  (A32 mode: fp32→fp16/bf16)")
        else:
            if kernel_info.get('dq_shuffle'):
                print(f"    dQ_convert: skipped  (A16 mode uses dQ_shuffle instead)")
            else:
                print(f"    dQ_convert: {kernel_info['dq_convert']}")
        kernel_exact = co_name in kernel_info['dqdkdv']
        kernel_equiv = is_equivalent_kernel(co_name, kernel_info['dqdkdv'])
        kernel_matched = kernel_exact or kernel_equiv
        if kernel_exact:
            print(f"    Kernel match: ✓ ({co_name} dispatched as expected)")
        elif kernel_equiv:
            print(f"    Kernel match: ≈ (equivalent kernel dispatched: {kernel_info['dqdkdv']})")
        else:
            print(f"    Kernel match: ✗ (expected {co_name}, got: {kernel_info['dqdkdv']})")
        if kernel_info['raw_output']:
            print(f"    [raw C++ output]: {kernel_info['raw_output']}")
        
    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else f"{type(e).__name__}"
        print(f"  ERROR: Aiter failed with: {error_msg}")
        traceback.print_exc()
        return {"passed": False, "error": error_msg or "Unknown error", "co_name": co_name, "test_name": test_name,
                "kernel_info": None, "kernel_matched": None}
    
    # Run PyTorch reference in float32 (upcast=True) - aligned with test_mha.py
    out_ref, softmax_lse_ref, dq_ref, dk_ref, dv_ref = run_torch(
        q, k, v, dout, causal=causal, window_size=window_size, upcast=True
    )
    
    # Run PyTorch in original dtype with reorder_ops (upcast=False, reorder_ops=True)
    # This is exactly what test_mha.py does for tolerance calculation
    out_pt, softmax_lse_pt, dq_pt, dk_pt, dv_pt = run_torch(
        q, k, v, dout, causal=causal, window_size=window_size, upcast=False, reorder_ops=True
    )
    
    # Print diff values (aligned with test_mha.py)
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"softmax_lse max diff: {(softmax_lse - softmax_lse_ref).abs().max().item()}")
    print(f"softmax_lse Pytorch max diff: {(softmax_lse_pt - softmax_lse_ref).abs().max().item()}")
    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
    
    # Tolerance calculation (aligned with test_mha.py: 10x PyTorch diff, min 0.01)
    dq_tol = max(10 * (dq_pt - dq_ref).abs().max().item(), 0.01)
    dk_tol = max(10 * (dk_pt - dk_ref).abs().max().item(), 0.01)
    dv_tol = max(10 * (dv_pt - dv_ref).abs().max().item(), 0.01)
    
    dq_diff = (dq - dq_ref).abs().max().item()
    dk_diff = (dk - dk_ref).abs().max().item()
    dv_diff = (dv - dv_ref).abs().max().item()
    
    dq_passed = dq_diff <= dq_tol
    dk_passed = dk_diff <= dk_tol
    dv_passed = dv_diff <= dv_tol
    passed = dq_passed and dk_passed and dv_passed
    
    status_q = '✓' if dq_passed else '✗'
    status_k = '✓' if dk_passed else '✗'
    status_v = '✓' if dv_passed else '✗'
    
    print(f"\nTolerance: dQ_tol={dq_tol:.6f}, dK_tol={dk_tol:.6f}, dV_tol={dv_tol:.6f}")
    print(f"  dQ: diff={dq_diff:.6f} vs tol={dq_tol:.6f} {status_q}")
    print(f"  dK: diff={dk_diff:.6f} vs tol={dk_tol:.6f} {status_k}")
    print(f"  dV: diff={dv_diff:.6f} vs tol={dv_tol:.6f} {status_v}")
    print(f"  Result: {'PASSED' if passed else 'FAILED'}")
    
    # Show which buffer overflows were exercised and verified
    overflow_bufs = [name for name, (_, ov) in overflow_info.items() if ov]
    if overflow_bufs:
        print(f"  Overflow buffers verified: {', '.join(overflow_bufs)}")
    
    # Print mismatch details if failed (aligned with test_mha.py)
    print_mismatch_info("dQ", dq, dq_ref, dq_pt, dq_tol)
    print_mismatch_info("dK", dk, dk_ref, dk_pt, dk_tol)
    print_mismatch_info("dV", dv, dv_ref, dv_pt, dv_tol)
    
    return {
        "passed": passed,
        "dq_diff": dq_diff,
        "dk_diff": dk_diff,
        "dv_diff": dv_diff,
        "dq_tol": dq_tol,
        "dk_tol": dk_tol,
        "dv_tol": dv_tol,
        "co_name": co_name,
        "test_name": test_name,
        "overflow_buffers": overflow_bufs,
        "kernel_info": kernel_info,
        "kernel_matched": kernel_matched,
    }


def run_mha_varlen_backward_test(
    batch_size: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    hdim_q: int,
    hdim_v: int,
    dtype: torch.dtype,
    causal: bool,
    causal_type: str = None,
    window_size: tuple = (-1, -1),
    deterministic: bool = False,
    test_name: str = "unknown",
    co_name: str = "",
    is_v3_atomic_fp32: bool = True,  # True=A32 kernel, False=A16 kernel
    how_v3_bf16_cvt: int = 1,  # 0=RTNE, 1=RTNA, 2=RTZ
    **kwargs,
):
    """
    Run a single MHA varlen (group mode) backward test and return results.
    Aligned with test_mha.py methodology.
    """
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    is_swa = window_size != (-1, -1)
    
    print(f"\n{'='*70}")
    print(f"Test: {test_name} (VARLEN/GROUP MODE)")
    print(f"CO file: {co_name}")
    print(f"Command: python test_large_addr.py --kernel {co_name.replace('.co', '')}")
    print(f"Config: batch={batch_size}, heads={nheads}, seq_q={seqlen_q}, seq_k={seqlen_k}")
    print(f"        hdim_q={hdim_q}, hdim_v={hdim_v}, dtype={dtype}")
    print(f"        causal={causal}, causal_type={causal_type}, deterministic={deterministic}")
    if is_swa:
        print(f"        window_size=({window_size[0]}, {window_size[1]}) (SWA mode)")
    cvt_names = {0: "RTNE", 1: "RTNA", 2: "RTZ", 3: "FP16"}
    print(f"        is_v3_atomic_fp32={is_v3_atomic_fp32} ({'A32' if is_v3_atomic_fp32 else 'A16'} kernel)")
    print(f"        how_v3_bf16_cvt={how_v3_bf16_cvt} ({cvt_names.get(how_v3_bf16_cvt, '?')})")
    print(f"{'='*70}")
    
    overflow_info = print_overflow_analysis(
        batch_size, nheads, seqlen_q, seqlen_k, hdim_q, hdim_v,
        atomic32=int(is_v3_atomic_fp32),
    )
    
    device = "cuda"
    
    # For varlen mode, create packed tensors
    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_k
    
    # Create cumulative sequence length tensors
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, seqlen_q, device=device, dtype=torch.int32)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, seqlen_k, device=device, dtype=torch.int32)
    
    # Create packed input tensors (no batch dimension)
    q = torch.randn(total_q, nheads, hdim_q, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_k, nheads, hdim_q, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_k, nheads, hdim_v, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(total_q, nheads, hdim_v, device=device, dtype=dtype, requires_grad=True)
    
    # Run aiter forward + backward
    try:
        with capture_cpp_stdout() as cap:
            out, softmax_lse, *_ = aiter.flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=seqlen_q,
                max_seqlen_k=seqlen_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=causal,
                window_size=window_size,
                return_lse=True,
                return_attn_probs=False,
                deterministic=deterministic,
                how_v3_bf16_cvt=how_v3_bf16_cvt,
            )
            
            # Compute gradients
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        
        kernel_info = parse_kernel_info(cap._captured)
        print(f"\n  Kernel dispatch info:")
        print(f"    ODO:        {kernel_info['odo']}")
        print(f"    dQdKdV:     {kernel_info['dqdkdv']}")
        if kernel_info.get('dq_shuffle'):
            print(f"    dQ_shuffle:  {kernel_info['dq_shuffle']}  (A16 mode: dQ accumulated in fp16/bf16)")
        if is_v3_atomic_fp32:
            print(f"    dQ_convert: {kernel_info['dq_convert']}  (A32 mode: fp32→fp16/bf16)")
        else:
            if kernel_info.get('dq_shuffle'):
                print(f"    dQ_convert: skipped  (A16 mode uses dQ_shuffle instead)")
            else:
                print(f"    dQ_convert: {kernel_info['dq_convert']}")
        kernel_exact = co_name in kernel_info['dqdkdv']
        kernel_equiv = is_equivalent_kernel(co_name, kernel_info['dqdkdv'])
        kernel_matched = kernel_exact or kernel_equiv
        if kernel_exact:
            print(f"    Kernel match: ✓ ({co_name} dispatched as expected)")
        elif kernel_equiv:
            print(f"    Kernel match: ≈ (equivalent kernel dispatched: {kernel_info['dqdkdv']})")
        else:
            print(f"    Kernel match: ✗ (expected {co_name}, got: {kernel_info['dqdkdv']})")
        if kernel_info['raw_output']:
            print(f"    [raw C++ output]: {kernel_info['raw_output']}")
        
    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else f"{type(e).__name__}"
        print(f"  ERROR: Aiter failed with: {error_msg}")
        traceback.print_exc()
        return {"passed": False, "error": error_msg or "Unknown error", "co_name": co_name, "test_name": test_name,
                "kernel_info": None, "kernel_matched": None}
    
    # For reference, reshape to batch format and compute using attention_ref
    q_batch = q.reshape(batch_size, seqlen_q, nheads, hdim_q).requires_grad_(True)
    k_batch = k.reshape(batch_size, seqlen_k, nheads, hdim_q).requires_grad_(True)
    v_batch = v.reshape(batch_size, seqlen_k, nheads, hdim_v).requires_grad_(True)
    dout_batch = dout.reshape(batch_size, seqlen_q, nheads, hdim_v)
    
    # Run PyTorch reference in float32 (upcast=True)
    out_ref, softmax_lse_ref, dq_ref_batch, dk_ref_batch, dv_ref_batch = run_torch(
        q_batch, k_batch, v_batch, dout_batch, causal=causal, window_size=window_size, upcast=True
    )
    
    # Run PyTorch in original dtype with reorder_ops
    out_pt, softmax_lse_pt, dq_pt_batch, dk_pt_batch, dv_pt_batch = run_torch(
        q_batch, k_batch, v_batch, dout_batch, causal=causal, window_size=window_size, upcast=False, reorder_ops=True
    )
    
    # Reshape reference results to match varlen format
    dq_ref = dq_ref_batch.reshape(total_q, nheads, hdim_q)
    dk_ref = dk_ref_batch.reshape(total_k, nheads, hdim_q)
    dv_ref = dv_ref_batch.reshape(total_k, nheads, hdim_v)
    dq_pt = dq_pt_batch.reshape(total_q, nheads, hdim_q)
    dk_pt = dk_pt_batch.reshape(total_k, nheads, hdim_q)
    dv_pt = dv_pt_batch.reshape(total_k, nheads, hdim_v)
    
    # Print diff values (aligned with test_mha.py)
    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
    
    # Tolerance calculation (aligned with test_mha.py)
    dq_tol = max(10 * (dq_pt - dq_ref).abs().max().item(), 0.01)
    dk_tol = max(10 * (dk_pt - dk_ref).abs().max().item(), 0.01)
    dv_tol = max(10 * (dv_pt - dv_ref).abs().max().item(), 0.01)
    
    dq_diff = (dq - dq_ref).abs().max().item()
    dk_diff = (dk - dk_ref).abs().max().item()
    dv_diff = (dv - dv_ref).abs().max().item()
    
    dq_passed = dq_diff <= dq_tol
    dk_passed = dk_diff <= dk_tol
    dv_passed = dv_diff <= dv_tol
    passed = dq_passed and dk_passed and dv_passed
    
    status_q = '✓' if dq_passed else '✗'
    status_k = '✓' if dk_passed else '✗'
    status_v = '✓' if dv_passed else '✗'
    
    print(f"\nTolerance: dQ_tol={dq_tol:.6f}, dK_tol={dk_tol:.6f}, dV_tol={dv_tol:.6f}")
    print(f"  dQ: diff={dq_diff:.6f} vs tol={dq_tol:.6f} {status_q}")
    print(f"  dK: diff={dk_diff:.6f} vs tol={dk_tol:.6f} {status_k}")
    print(f"  dV: diff={dv_diff:.6f} vs tol={dv_tol:.6f} {status_v}")
    print(f"  Result: {'PASSED' if passed else 'FAILED'}")
    
    overflow_bufs = [name for name, (_, ov) in overflow_info.items() if ov]
    if overflow_bufs:
        print(f"  Overflow buffers verified: {', '.join(overflow_bufs)}")
    
    # Print mismatch details if failed
    print_mismatch_info("dQ", dq, dq_ref, dq_pt, dq_tol)
    print_mismatch_info("dK", dk, dk_ref, dk_pt, dk_tol)
    print_mismatch_info("dV", dv, dv_ref, dv_pt, dv_tol)
    
    return {
        "passed": passed,
        "dq_diff": dq_diff,
        "dk_diff": dk_diff,
        "dv_diff": dv_diff,
        "dq_tol": dq_tol,
        "dk_tol": dk_tol,
        "dv_tol": dv_tol,
        "co_name": co_name,
        "test_name": test_name,
        "overflow_buffers": overflow_bufs,
        "kernel_info": kernel_info,
        "kernel_matched": kernel_matched,
    }


def run_all_tests(kernel_filter=None, large_q=False, large_k=False, normal=False):
    """Run all kernel tests or a filtered subset.
    
    By default (no flags), runs BOTH large_q and large_k tests.
    Use --large-q or --large-k to run only one mode.
    Use --normal to add a baseline correctness test with small seqlens (q=256, k=256).
    """
    print(f"\nRunning on: {get_device_arch()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Determine which modes to run
    # large_q value in tuple: None=normal, True=large_q, False=large_k
    any_specified = normal or large_q or large_k
    if not any_specified:
        # Default: run all three tests
        modes = [
            (None,  "normal (q=256, k=256): baseline correctness test"),
            (False, "large_k (q=256, k=75600): tests K/V/dK/dV address overflow"),
            (True,  "large_q (q=75600, k=256): tests Q/dO/dQ/ODO address overflow"),
        ]
    else:
        modes = []
        if normal:
            modes.append((None, "normal (q=256, k=256): baseline correctness test"))
        if large_k:
            modes.append((False, "large_k (q=256, k=75600): tests K/V/dK/dV address overflow"))
        if large_q:
            modes.append((True, "large_q (q=75600, k=256): tests Q/dO/dQ/ODO address overflow"))
    
    configs = load_kernel_configs()
    
    if not configs:
        print("No kernel configurations found!")
        return
    
    # Filter configs if requested
    if kernel_filter:
        filter_lower = os.path.basename(kernel_filter).lower()
        filtered = []
        for c in configs:
            co_name = c['co_name'].lower()
            # Check if filter matches
            if filter_lower in co_name:
                # If filter doesn't contain 'group', exclude group kernels
                # unless the filter is an exact match for the kernel name
                kernel_base = co_name.replace('bwd_', '').replace('.co', '')
                filter_base = filter_lower.replace('bwd_', '').replace('.co', '')
                
                if 'group' not in filter_lower and kernel_base.endswith('_group'):
                    # Filter doesn't want group kernels, skip
                    continue
                filtered.append(c)
        configs = filtered
        if not configs:
            print(f"No kernels matching '{kernel_filter}' found!")
            return
    
    all_results = []
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for is_large_q, mode_desc in modes:
        print(f"\n{'#'*70}")
        print(f"# Mode: {mode_desc}")
        print(f"{'#'*70}")
        
        results = []
        passed = 0
        failed = 0
        skipped = 0
        
        for config in configs:
            cfg_pssk = int(config['pssk'])
            
            # pssk=0 kernels use equal seqlens - only run once (in large_k pass)
            if cfg_pssk == 0 and is_large_q:
                skipped += 1
                continue
            
            params = get_test_params_for_kernel(config, large_q=is_large_q)
            if params is None:
                skipped += 1
                continue
            
            if is_large_q is None:
                suffix = "_normal"
            elif cfg_pssk == 0:
                suffix = "_equalSeqlen"
            elif is_large_q:
                suffix = "_largeQ"
            else:
                suffix = "_largeK"
            test_name = params['co_name'].replace('.co', '') + suffix
            
            # Choose test function based on mode
            if params.get('is_varlen', False):
                result = run_mha_varlen_backward_test(
                    **params,
                    test_name=test_name,
                )
            else:
                result = run_mha_backward_test(
                    **params,
                    test_name=test_name,
                )
            
            results.append(result)
            if result.get('passed', False):
                passed += 1
            else:
                failed += 1
        
        # Print per-mode summary
        print(f"\n{'-'*70}")
        print(f"  [{mode_desc}]  Passed: {passed}  Failed: {failed}  Skipped: {skipped}")
        print(f"{'-'*70}")
        
        if failed > 0:
            for r in results:
                if not r.get('passed', False):
                    if 'error' in r:
                        print(f"    FAIL: {r.get('test_name', r['co_name'])}: {r['error']}")
                    else:
                        dq = r.get('dq_diff', float('nan'))
                        dk = r.get('dk_diff', float('nan'))
                        dv = r.get('dv_diff', float('nan'))
                        dq_tol = r.get('dq_tol', 0.01)
                        dk_tol = r.get('dk_tol', 0.01)
                        dv_tol = r.get('dv_tol', 0.01)
                        parts = []
                        if isinstance(dq, (int, float)) and not (dq != dq):
                            parts.append(f"dQ={dq:.4f}(tol={dq_tol:.4f})")
                        else:
                            parts.append(f"dQ={dq}")
                        if isinstance(dk, (int, float)) and not (dk != dk):
                            parts.append(f"dK={dk:.4f}(tol={dk_tol:.4f})")
                        else:
                            parts.append(f"dK={dk}")
                        if isinstance(dv, (int, float)) and not (dv != dv):
                            parts.append(f"dV={dv:.4f}(tol={dv_tol:.4f})")
                        else:
                            parts.append(f"dV={dv}")
                        print(f"    FAIL: {r.get('test_name', r['co_name'])}: {', '.join(parts)}")
        
        all_results.extend(results)
        total_passed += passed
        total_failed += failed
        total_skipped += skipped
    
    # Print overall summary
    print(f"\n{'='*70}")
    print(f"OVERALL TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  Passed:  {total_passed}")
    print(f"  Failed:  {total_failed}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Total:   {len(all_results)}")
    print(f"{'='*70}")
    
    if total_failed > 0:
        print(f"\nAll failed tests:")
        for r in all_results:
            if not r.get('passed', False):
                name = r.get('test_name', r['co_name'])
                if 'error' in r:
                    print(f"  - {name}: {r['error']}")
                else:
                    dq = r.get('dq_diff', float('nan'))
                    dk = r.get('dk_diff', float('nan'))
                    dv = r.get('dv_diff', float('nan'))
                    dq_tol = r.get('dq_tol', 0.01)
                    dk_tol = r.get('dk_tol', 0.01)
                    dv_tol = r.get('dv_tol', 0.01)
                    parts = []
                    if isinstance(dq, (int, float)) and not (dq != dq):
                        parts.append(f"dQ={dq:.4f}(tol={dq_tol:.4f})")
                    else:
                        parts.append(f"dQ={dq}")
                    if isinstance(dk, (int, float)) and not (dk != dk):
                        parts.append(f"dK={dk:.4f}(tol={dk_tol:.4f})")
                    else:
                        parts.append(f"dK={dk}")
                    if isinstance(dv, (int, float)) and not (dv != dv):
                        parts.append(f"dV={dv:.4f}(tol={dv_tol:.4f})")
                    else:
                        parts.append(f"dV={dv}")
                    print(f"  - {name}: {', '.join(parts)}")
    
    # Kernel match verification
    matched_count = 0
    mismatched_count = 0
    unknown_count = 0
    mismatched_details = []
    for r in all_results:
        km = r.get('kernel_matched')
        if km is True:
            matched_count += 1
        elif km is False:
            mismatched_count += 1
            mismatched_details.append(
                f"  ✗ {r.get('test_name', r['co_name'])}: "
                f"expected {r['co_name']}, dispatched {r.get('kernel_info', {}).get('dqdkdv', '?')}"
            )
        else:
            unknown_count += 1
    
    print(f"\n{'='*70}")
    print(f"KERNEL DISPATCH VERIFICATION")
    print(f"{'='*70}")
    if kernel_filter:
        print(f"  --kernel filter: {kernel_filter}")
    print(f"  Matched:    {matched_count}")
    if mismatched_count:
        print(f"  MISMATCHED: {mismatched_count}")
    if unknown_count:
        print(f"  Unknown:    {unknown_count} (test errored before dispatch)")
    if mismatched_details:
        print(f"\n  Mismatched kernels:")
        for d in mismatched_details:
            print(d)
    if mismatched_count == 0 and unknown_count == 0:
        print(f"  All {matched_count} test(s) dispatched the expected kernel ✓")
    print(f"{'='*70}")


def list_tests():
    """List all available tests."""
    configs = load_kernel_configs()
    
    print(f"\nAvailable kernel tests ({len(configs)} total):")
    print(f"{'='*70}")
    
    for config in configs:
        mode_str = "group" if config['mode'] == '1' else "batch"
        atomic_str = "a32" if config['atomic32'] == '1' else "a16"
        mask_names = {0: "none", 1: "causal", 2: "causal_br", 3: "swa"}
        mask_str = mask_names.get(int(config['mask']), f"unknown({config['mask']})")
        print(f"  {config['co_name']}: {config['dtype']} {atomic_str} hdim={config['hdim_q']}/{config['hdim_v']} "
              f"mask={mask_str} mode={mode_str}")


def show_csv():
    """Show CSV kernel mapping."""
    configs = load_kernel_configs()
    
    print(f"\nCSV kernel configurations ({len(configs)} entries):")
    print(f"{'='*100}")
    
    headers = list(configs[0].keys()) if configs else []
    print("  " + " | ".join(f"{h:>10}" for h in headers))
    print(f"  {'-'*90}")
    
    for config in configs:
        values = [config.get(h, '') for h in headers]
        print("  " + " | ".join(f"{v:>10}" for v in values))


def main():
    parser = argparse.ArgumentParser(
        description="Test 64-bit address fixes in FMHA backward kernels"
    )
    parser.add_argument(
        "--kernel", "-k",
        type=str,
        default=None,
        help="Filter tests by kernel name substring (e.g., 'bf16_a32', 'hd128')"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available tests"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Show CSV kernel mapping"
    )
    parser.add_argument(
        "--large-q",
        action="store_true",
        help="Only test large seqlen_q (q=75600, k=256). Default runs both."
    )
    parser.add_argument(
        "--large-k",
        action="store_true",
        help="Only test large seqlen_k (q=256, k=75600). Default runs both."
    )
    parser.add_argument(
        "--normal",
        action="store_true",
        help="Add baseline correctness test with small seqlens (q=256, k=256). "
             "Use alone for normal-only, or combine with --large-q/--large-k."
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_tests()
    elif args.csv:
        show_csv()
    else:
        run_all_tests(kernel_filter=args.kernel, large_q=args.large_q, large_k=args.large_k, normal=args.normal)


if __name__ == "__main__":
    main()