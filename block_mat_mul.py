import torch
import triton
import triton.language as tl

@triton.jit
def block_mat_mul_kernel(
    A_ptr,B_ptr,C_ptr,
    M,N,K,
    stride_am,stride_ak,
    stride_bk,stride_bn,
    stride_cm,stride_cn,
    BLOCK_SIZE_M :  tl.constexpr,
    BLOCK_SIZE_N :  tl.constexpr,
    BLOCK_SIZE_K : tl.constexpr
):
    pid = tl.program(axis=0)

    num_cols_block = tl.cdiv(N,BLOCK_SIZE_N) # How many columns a block have
    
    # This gives block id along rows and colums(Which row and column along C)
    block_id_m = pid // num_cols_block
    block_id_n = pid % num_cols_block

    # Inside the block we need to tell program which rows and columns it will work with
    offs_m = block_id_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    offs_n = block_id_n * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
    offs_k = tl.arange(0,BLOCK_SIZE_K)
    
    # 0ffs_m[:,None] turns into a column vector,for ex:[0,1] -> [[0],[1]] and Offs_m[None,:] truns into row vector.
    # After broadcasting is applied on both column vector and row vector and additon is done to get memory address of the elements
    # For triton (M,1) + (1,N) = (M,N)
    a_ptrs = A_ptr + offs_m[:,None] * stride_am + offs_k[None,:] * stride_ak
    b_ptrs = B_ptr + offs_k[:,None] * stride_bk + offs_n[None,:] * stride_bn


    accumlator = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)
    
    
    # If loop variable k is the block number, then this is used offs_k[None, :] < K - k * BLOCK_SIZE_K:
    # The only caveat is if your loop is written differently:
    # for k in range(0, K, BLOCK_K):
    # Then we should write:
    # offs_k < K - k
    for k in range(0,tl.cdiv(K,BLOCK_SIZE_K)):
        a_mask = offs_m[:,None] < M & (offs_k[None,:] < K - k * BLOCK_SIZE_K)
        a_val = tl.load(a_ptrs,mask=a_mask,other=0.0)

        b_mask = (offs_k[:,None] < K - k) & (BLOCK_SIZE_K  & offs_n[None,:] < N)
        b_val = tl.load(b_ptrs,b_mask,other=0.0)

        accumlator += tl.dot(a_val,b_val)

        a_ptrs += stride_ak * BLOCK_SIZE_K
        b_ptrs += stride_bk * BLOCK_SIZE_K

    c = accumlator.to(tl.float16)

    offs_cm = block_id_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    offs_cn = block_id_n * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
    c_ptrs = C_ptr + offs_cm[:,None] * stride_cm + offs_cn[None,:] * stride_cn
    c_mask = (offs_cm[:,None] < M) & (offs_cn[None,:] < N)

    tl.load(c_ptrs,c,mask=c_mask)

def mat_mul(A,B):
    M,K = A.shape
    K1,N = B.shape

    assert K == K1

    C = torch.empty((M,N),dtype=torch.float16,device=A.device)

    grid = lambda meta : (triton.cdiv(M,meta['BLOCK_SIZE_M']) * triton.cdiv(N,meta['BLOCK_SIZE_N']),)
    block_mat_mul_kernel[grid](
        A,B,C,
        M,N,K,
        A.stride(0),A.stride(1),
        B.stride(0),B.stride(1),
        C.stride(0),C.stride(1)
    )

    return C

A = torch.randn(100,500,device="cuda")
B = torch.randn(500,1000,device="cuda")

print(mat_mul(A,B) - A@B)

    



















