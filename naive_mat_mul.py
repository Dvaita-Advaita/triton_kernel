import torch
import triton
import triton.language as tl

@triton.jit
def naive_mat_mul_kernel(
    A_ptr,B_ptr,C_ptr,
    M,N,K,
    stride_am,stride_an,
    stride_bn,stride_bk,
    stride_cm,stride_ck
):
    pid = tl.program_id(axis=0)

    m = pid // N
    n = pid % N
    
    accum = 0.0 # Inside the loop it is floating point so we have to assign floating point number not accum = 0 which integer
    for i in range(0,N):
        a_ptrs = A_ptr + m * stride_am + i * stride_an
        b_ptrs = B_ptr + n * stride_bk + i * stride_bn
        a_val = tl.load(a_ptrs)
        b_val = tl.load(b_ptrs)

        accum += a_val *  b_val

    tl.store(C_ptr + m * stride_cm + n * stride_ck,accum)




def naive_mat_mul(A,B):

    M,N = A.shape
    N1,K = B.shape

    assert N == N1

    C = torch.zeros((M,K), dtype=A.dtype, device=A.device)

    grid = (M * K,)

    naive_mat_mul_kernel[grid](
        A,B,C,
        M,N,K,
        A.stride(0),A.stride(1),
        B.stride(0),B.stride(1),
        C.stride(0),C.stride(1))
    
    return C

A = torch.randn((6,8),device="cuda")
B = torch.rand((8,4),device="cuda")

print(naive_mat_mul(A,B) - A @ B)

