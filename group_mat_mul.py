import torch
import pytest
import triton
import triton.language as tl



def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2)]

@triton.autotune(
    configs=get_cuda_autotune_config(),
     key=['M', 'N', 'K'],
)


@triton.jit
def group_mat_mul_kernel(
    A_ptr,B_ptr,C_ptr,
    M,K,N,
    stride_am,stride_ak,
    stride_bk,stride_bn,
    stride_cm,stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # number of tile/block along rows
    num_pid_m = tl.cdiv(M,BLOCK_SIZE_M)

    # number of tile/block along column
    num_pid_n = tl.cdiv(N,BLOCK_SIZE_N)

    # number of program in group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # group id
    group_id = pid // num_pid_in_group

    # first row of the tile(which is along m) of each group
    first_pid_m = group_id * GROUP_SIZE_M

    # Suppose we have 5 tile and each group is made up of 2 tile,we can make 2 group but there is still 5th rows remaining
    # Now we have to ensure that last group is only made of 1 tile
    group_size_m = min(num_pid_m - first_pid_m,GROUP_SIZE_M)

    # Since each group is made up of multiple of programs,first we have to give each program a local id (Global id of each program already exist)
    # Suppose there are total 16 programs,hence it's global id is (0,15),now each group is made up of 8 programs,then local id can be(0,7)
    local_pid_in_group = pid % num_pid_in_group

    # In blocked mat mul we load program P00->P01->P02->P10->-P11->P12 so on(This is called Row-major Ordering)
    # We can see rate of change of column is much faster than row(when column changes 3(0->1->2) times,row changes 1 time(0->1))
    # In group mat mul we want to load the program in this manner P00->P10->P01->P11->P02->P12 so on 
    # In this case rate of change of row is faster than column(row changes 2 times then column changes 1 times)
    # In blocked mat mul pid_m = pid // BLOCK_SIZE_N and pid_n = pid % BLOCK_SIZE_N,if we want fast change use %(mod),for slow change use //(int divide)
    # We can see the similarity b/w pid_m and local_pid_m,one uses pid other uses local pid,only difference is in blocked mat mul row is changing slowly but here faster
    local_pid_m = local_pid_in_group % group_size_m

    pid_m = first_pid_m + local_pid_m
    pid_n = local_pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
    offs_k = tl.arange(0,BLOCK_SIZE_K)

    a_ptrs = A_ptr + offs_m[:,None] * stride_am + offs_k[None,:] * stride_ak
    b_ptrs = B_ptr + offs_k[:,None] * stride_bk + offs_n[None,:] * stride_bn

    accumlator = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)
    
    # If loop variable k is the block number, then this is used offs_k[None, :] < K - k * BLOCK_SIZE_K:
    # The only caveat is if your loop is written differently:
    # for k in range(0, K, BLOCK_K):
    # Then we should write:
    # offs_k < K - k
    for k in range(0,tl.cdiv(K,BLOCK_SIZE_K)):
        a_mask = (offs_m[:,None] < M) & (offs_k[None,:] < K - k * BLOCK_SIZE_K)
        a_val = tl.load(a_ptrs,mask=a_mask,other=0.0)

        b_mask = (offs_k[:,None] < K - k) * (BLOCK_SIZE_K & offs_n[None,:] < N)
        b_val = tl.load(b_ptrs,mask=b_mask,other=0.0)

        accumlator += tl.dot(a_val,b_val)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Converting float32 into float16
    C = accumlator.to(tl.float16) 
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
    c_ptrs = C_ptr + offs_cm[:,None] * stride_cm + offs_cn[None,:] * stride_cn
    c_mask = (offs_cm[:,None] < M) & (offs_cn[None:,] < N)

    tl.store(c_ptrs,C,mask=c_mask)

def group_mat_mul(A,B):
    M,K = A.shape
    K1,N = B.shape

    assert K == K1
    
    C = torch.empty((M,N),dtype=torch.float16,device=A.device)
    
    # As triton.language as tl is meant for triton kernel(@triton.jit),since this is ordinary python function we can not use tl.cdiv we have to use triton.cdiv
    grid = lambda meta: (triton.cdiv(M,meta['BLOCK_SIZE_M']) * triton.cdiv(N,meta['BLOCK_SIZE_N']),)

    group_mat_mul_kernel[grid](
        A,B,C,
        M,K,N,
        A.stride(0),A.stride(1),
        B.stride(0),B.stride(1),
        C.stride(0),C.stride(1),
    )
    return  C

# PyTEST
@pytest.mark.parametrize("M,K,N",[
    (64,48,128),
    (100,500,100),
    # EDGE CASES
    (33, 65, 17),
    (127, 131, 67),
    (129, 63, 130),
    
])

# Pytorch test ∣actual−expected∣ ≤ atol + rtol×∣expected∣
# Absolute Tolerance(atol) is especially important when the expected value is close to zero
# Relative Tolerance(rtol) scales with the size of the expected value.
def test_mat_mul(M,K,N):
    a = torch.randn((M,K),dtype=torch.float16,device="cuda")
    b = torch.randn((K,N),dtype=torch.float16,device="cuda")

    pytorch_mat_mul = torch.matmul(a,b) # Actual Value of Pytorch
    grouped_mat_mul = group_mat_mul(a,b) # Expected Value from triton

    assert pytorch_mat_mul.shape == (M,N)

    torch.testing.assert_close(
    pytorch_mat_mul,
    grouped_mat_mul,
    rtol=1e-2, # Relative Tolerance
    atol=1e-2, # Absolute Tolerance
    )

    max_diff = float(torch.max(torch.abs(grouped_mat_mul - pytorch_mat_mul)))
    print(f"Matmul Sucess: M={M},K={K},N={N} | max_diff {max_diff:.4f}")




    
