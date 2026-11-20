# Triton kernels: from vector add to FlashAttention

I wrote GPU kernels in [Triton](https://github.com/triton-lang/triton) from scratch, one step at a time:
first elementwise ops, then matrix multiplication, then a full FlashAttention (forward and backward).
Each file is standalone and checks its output against PyTorch.

## Files, in learning order

| File | What it does |
|---|---|
| `vector_addition.py` | First kernel: blocked elementwise add, compared with `torch` |
| `naive_mat_mul.py` | Matmul with one program per output element |
| `block_mat_mul.py` | Tiled matmul: each program computes a `BLOCK_M × BLOCK_N` tile, looping over K |
| `group_mat_mul.py` | Grouped tile ordering (`GROUP_SIZE_M`) for better L2-cache reuse, plus autotuned block sizes and pipeline stages |
| `non_causal_FA_forward_pass.py` | FlashAttention forward for one head: tiled Q·Kᵀ with online softmax, so the N×N matrix is never stored |
| `non_causal_MultiHead_FA_forward_pass.py` | Extended to batch × heads, with masking of padded keys |
| `causal_MultiHead_FA_forward_pass.py` | Adds the causal mask |
| `FLASH_ATTENTION.py` | **Full FlashAttention, forward + backward**: stores log-sum-exp in the forward pass, a preprocessing kernel computes `delta = rowsum(dO ∘ O)`, then separate dQ and dK/dV backward kernels. All kernels are autotuned. |

## Correctness

`FLASH_ATTENTION.py` checks **O, LSE, dQ, dK and dV** against a plain PyTorch causal attention reference
(fp16). It covers several shapes, including sequence lengths that aren't multiples of the block size
(e.g. N = 17, 37, 65), and prints max and mean absolute error for each tensor. The matmul files print
their difference from `A @ B`.

## Benchmark

`benchmark_case(B, H, N, D)` in `FLASH_ATTENTION.py` times forward and backward with
`triton.testing.do_bench` against `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`.

<!-- Re-run and fill in. Results depend on the GPU, so name it. -->
| GPU | B, H, N, D | Triton fwd (ms) | PyTorch SDPA fwd (ms) | Triton bwd (ms) | PyTorch bwd (ms) |
|---|---|---|---|---|---|
| … | … | … | … | … | … |

## Run

Needs an NVIDIA GPU, PyTorch and Triton.

```bash
pip install torch triton
python FLASH_ATTENTION.py      # correctness checks, then forward/backward timings vs SDPA for N = 128…2048
```

Every other file is a standalone script, e.g. `python block_mat_mul.py` or
`python causal_MultiHead_FA_forward_pass.py`. Each prints its difference from the PyTorch result.

## What I learned
- Why attention is memory-bound, not compute-bound, when the N×N score matrix is written to HBM, and how tiling plus online softmax keeps everything in SRAM.
- Why the backward pass recomputes attention probabilities from the saved log-sum-exp instead of storing them.
- How tile order (grouping) changes L2 hit rate in matmul, and what `num_warps` and `num_stages` trade off.

## References
- Dao et al., *FlashAttention* (2022) and *FlashAttention-2* (2023)
- Triton tutorials: vector add, matrix multiplication, fused attention
