import math
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr,K_ptr,V_ptr,O_ptr,

    N,
    D: tl.constexpr,

    stride_qn,
    stride_qd,

    stride_kn,
    stride_kd,

    stride_vn,
    stride_vd,

    stride_on,
    stride_od,

    sm_scale,

    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr
):

    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)

    offs_d = tl.arange(0,D)

    q_ptrs = (Q_ptr + offs_m[:,None] * stride_qn + offs_d[None,:] * stride_qd)

    q_mask = ((offs_m[:,None] < N) & (offs_d[None,:] < D))

    q = tl.load(q_ptrs,mask=q_mask,other=0.0)

    m = tl.full((BLOCK_SIZE_M,),
                -1.0e6,  # Here if use float("-inf") it will produce NaN so we use -1.0e6 
                dtype=tl.float32)
    
    l = tl.zeros((BLOCK_SIZE_M,),
                    dtype=tl.float32)
    
    acc = tl.zeros((BLOCK_SIZE_M,D),
                   dtype=tl.float32)
    
    for kv_start in range(0,N,BLOCK_SIZE_N):

        offs_n = kv_start + tl.arange(0,BLOCK_SIZE_N)

        k_ptrs = (K_ptr + offs_d[:,None] * stride_kd + offs_n[None,:] * stride_kn)

        k_mask = ((offs_d[:,None] < D) & (offs_n[None,:] < N))

        k = tl.load(k_ptrs,mask=k_mask,other=0.0)

        scores = tl.dot(q,k)

        scores = scores * sm_scale

        # tl.where(condition,A,B) means if condition is True use A if it's false use B
        valid_keys = offs_n < N
        scores = tl.where(valid_keys[None,:],
                          scores,
                          -1.0e6) # Here if use float("-inf") it will produce NaN so we use -1.0e6 
        
        block_max = tl.max(scores,axis=1)

        m_new = tl.maximum(m,block_max)

        alpha = tl.exp(m - m_new)
        
        # softmax star
        p = tl.exp(scores - m_new[:,None])

        block_sum = tl.sum(p,axis=1)

        l = alpha * l + block_sum

        v_ptrs = (V_ptr + offs_n[:,None] * stride_vn + offs_d[None,:] * stride_vd)
        v_mask = ((offs_n[:,None] < N) & (offs_d[None,:] < D))

        v = tl.load(v_ptrs,mask=v_mask,other=0.0)

        acc = acc * alpha[:,None] + tl.dot(p.to(tl.float16),v)

        m = m_new

        output = (acc / l[:,None])

        o_ptrs = O_ptr + offs_m[:,None] * stride_on + offs_d[None,:] * stride_od

        o_mask = ((offs_m[:,None] < N) & (offs_d[None,:] < D))

        tl.store(o_ptrs,output,mask=o_mask)


def flash_attention(q,k,v):

    assert q.ndim == 2
    assert k.ndim == 2
    assert v.ndim == 2

    assert q.shape == k.shape
    assert q.shape == v.shape

    assert q.is_cuda
    assert k.is_cuda
    assert v.is_cuda
    
    assert q.dtype == torch.float16
    assert k.dtype == torch.float16
    assert v.dtype == torch.float16

    N,D = q.shape

    assert N > 0

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_D = max(16,triton.next_power_of_2(D),)

    output = torch.empty_like(q)

    sm_scale = 1.0/math.sqrt(D)

    grid = (triton.cdiv(N,BLOCK_SIZE_M),)

    flash_attention_fwd_kernel[grid](
        q,k,v,output,

        N,D,

        q.stride(0),q.stride(1),
        k.stride(0),k.stride(1),
        v.stride(0),v.stride(1),
        output.stride(0),output.stride(1),

        sm_scale,

        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_D,

        num_warps = 4
    )

    return output

# Test against Pytorch
N = 37
D = 32

q = torch.randn((N,D),dtype=torch.float16,device="cuda")
k = torch.randn((N,D),dtype=torch.float16,device="cuda")
v = torch.randn((N,D),dtype=torch.float16,device="cuda")

actual = flash_attention(q,k,v)

scale = 1/math.sqrt(D)

scores = (q.float() @ k.float().T)

scores *= scale

prob = torch.softmax(scores,dim=1)


expected = (
    prob @ v.float()).to(torch.float16)

torch.testing.assert_close(
    actual,expected,
    atol=1e-2,
    rtol=1e-2,
)

print("actual NaNs:", torch.isnan(actual).sum().item())
print("expected NaNs:", torch.isnan(expected).sum().item())

print("actual first row:", actual[0, :8])
print("expected first row:", expected[0, :8])




