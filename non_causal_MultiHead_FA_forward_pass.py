import math
import torch
import triton
import triton.language as tl

@triton.jit
def multihead_FA_kernel(

    Q_ptr,K_ptr,V_ptr,O_ptr,

    B,H,N,D:tl.constexpr,

    stride_qb,stride_qh,
    stride_qn,stride_qd,

    stride_kb,stride_kh,
    stride_kn,stride_kd,

    stride_vb,stride_vh,
    stride_vn,stride_vd,

    stride_ob,stride_oh,
    stride_on,stride_od,

    sm_scale,

    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
   # How many rows in block query martrix,then that number of program  
    pid_m =  tl.program_id(0)

    # How many total B x H program exist if B = 2 and H = 2 then 4 programs (0,1,2,3)
    pid_bh = tl.program_id(1)
    
    # Which pid_bh program is resonsible for which batch and head
    batch = pid_bh // H
    head = pid_bh % H
    
    # Since the shape the of the tensor is (B,H,N,D) in order to get the pointers for element we first need to know which batch and head it belongs to  
    Q_base = (Q_ptr + batch * stride_qb + head * stride_qh)
    K_base = (K_ptr + batch * stride_kb + head * stride_kh)
    V_base = (V_ptr + batch * stride_vb + head * stride_vh)
    O_base = (O_ptr + batch * stride_ob + head * stride_oh)
    
    # Offset and pointers
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    offs_d = tl.arange(0,BLOCK_SIZE_D)

    q_ptrs = (Q_base + offs_m[:,None] * stride_qn + offs_d[None,:] * stride_qd)

    q_mask = ((offs_m[:,None] < N) & (offs_d[None,:] < D))

    q = tl.load(q_ptrs,mask=q_mask,other=0.0)

    m = tl.full((BLOCK_SIZE_M,),
                -1.0e6,
               dtype=tl.float32 )
    
    l = tl.zeros((BLOCK_SIZE_M,),
                 dtype=tl.float32)
    
    acc = tl.zeros((BLOCK_SIZE_M,D),
                   dtype=tl.float32)
    
    for kv_start in range(0,N,BLOCK_SIZE_N):

        offs_n = kv_start + tl.arange(0,BLOCK_SIZE_N)

        k_ptrs = (K_base + offs_d[:,None] * stride_kd + offs_n[None,:] * stride_kn)

        k_mask = ((offs_d[:,None] < D) & (offs_n[None,:] < N))

        k = tl.load(k_ptrs,mask=k_mask,other=0.0)

        scores = tl.dot(q,k)

        scores = scores * sm_scale

        # Prevent padded keys from participating in online softmax

        valid_keys = offs_n < N

        scores = tl.where(
            valid_keys[None,:],
            scores,
            -1.0e6
        )
        
        ## Onlie Softmax ##

        block_max = tl.max(scores,axis=1)

        m_new = tl.maximum(m,block_max)
        
        # Rescaling factor
        alpha = tl.exp(m - m_new)

        p = tl.exp(scores-m_new[:,None])

        l = alpha * l + tl.sum(p,axis=1)

        # load v

        v_ptrs = (V_base + offs_n[:,None] * stride_vn + offs_d[None,:] * stride_vd)

        v_mask = ((offs_n[:,None] < N) & (offs_d[None,:] < D))

        v = tl.load(v_ptrs,mask=v_mask,other=0.0)
        
        acc = acc * alpha[:,None] + tl.dot(p.to(tl.float16),v)

        m = m_new

        output = acc / l[:,None]

        o_ptrs = (O_base + offs_m[:,None] * stride_on + offs_d[None,:] * stride_od)

        o_mask = ((offs_m[:,None] < N) & (offs_d[None,:] < D))

        tl.store(o_ptrs,output,mask=o_mask)

    
def multi_head_flash_attention(q,k,v):

    assert q.ndim == 4
    assert k.ndim == 4
    assert v.ndim == 4

    assert q.shape == k.shape
    assert q.shape == v.shape

    assert q.is_cuda
    assert k.is_cuda
    assert v.is_cuda

    assert q.dtype == torch.float16
    assert k.dtype == torch.float16
    assert v.dtype == torch.float16

    B,H,N,D = q.shape

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_D = max(16,triton.next_power_of_2(D))

    output = torch.empty_like(q)

    sm_scale = 1 / math.sqrt(D)

    grid = (
        triton.cdiv(N,BLOCK_SIZE_M), 
        B * H)

    multihead_FA_kernel[grid](
        q,k,v,output,

        B,H,N,D,

        q.stride(0),q.stride(1),
        q.stride(2),q.stride(3),

        k.stride(0),k.stride(1),
        k.stride(2),k.stride(3),

        v.stride(0),v.stride(1),
        v.stride(2),v.stride(3),

        output.stride(0),output.stride(1),
        output.stride(2),output.stride(3),

        sm_scale,

        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_D,

        num_warps = 4
    )

    return output


B = 2
H = 2
N = 32
D = 32

q = torch.randn(
    (B, H, N, D),
    device="cuda",
    dtype=torch.float16,
)

k = torch.randn(
    (B, H, N, D),
    device="cuda",
    dtype=torch.float16,
)

v = torch.randn(
    (B, H, N, D),
    device="cuda",
    dtype=torch.float16,
)

actual = multi_head_flash_attention(q, k, v)

# Against Pytorch

sm_scale = 1/ math.sqrt(D)

scores = (q.float() @ k.float().transpose(-2,-1))

scores = scores * sm_scale

p = torch.softmax(scores,dim = -1)

expected = (p @ v.float()).to(torch.float16)

print("actual shape:", actual.shape)
print("expected shape:", expected.shape)

print(
    "NaNs:",
    torch.isnan(actual).sum().item()
)

print(
    "max difference:",
    torch.max(
        torch.abs(actual - expected)
    ).item()
)

torch.testing.assert_close(
    actual,
    expected,
    atol=1e-2,
    rtol=1e-2,
)

print("Multi-head FlashAttention passed!")
    



        




