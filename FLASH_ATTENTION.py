import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
            },
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
            },
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
            },
            num_warps=4,
            num_stages=3,
        ),

        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
            },
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["N", "D"],
)


@triton.jit
def flash_attention_forward_kernel(
    Q_ptr,K_ptr,V_ptr,O_ptr,LSE_ptr,

    B,H,N,D:tl.constexpr,

    stride_qb,stride_qh,
    stride_qn,stride_qd,

    stride_kb,stride_kh,
    stride_kn,stride_kd,

    stride_vb,stride_vh,
    stride_vn,stride_vd,

    stride_ob,stride_oh,
    stride_on,stride_od,
    
    stride_lseb,stride_lseh,stride_lsen,

    sm_scale,

    BLOCK_SIZE_M:tl.constexpr,
    BLOCK_SIZE_N:tl.constexpr,
    BLOCK_SIZE_D:tl.constexpr,
):
    pid_m = tl.program_id(0)

    pid_bh = tl.program_id(1)

    batch = pid_bh // H
    head = pid_bh % H

    Q_base = Q_ptr + batch * stride_qb + head * stride_qh
    K_base = K_ptr + batch * stride_kb + head * stride_kh
    V_base = V_ptr + batch * stride_vb + head * stride_vh
    O_base = O_ptr + batch * stride_ob + head * stride_oh
    LSE_base = LSE_ptr + batch * stride_lseb + head * stride_lseh

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    offs_d = tl.arange(0,BLOCK_SIZE_D)

    q_ptrs = (Q_base + offs_m[:,None] * stride_qn + offs_d[None,:] * stride_qd)

    q_mask = ((offs_m[:,None] < N) & (offs_d[None,:] < D))

    q = tl.load(q_ptrs,mask=q_mask,other=0.0)

    m = tl.full((BLOCK_SIZE_M,),
                -1.0e6,
                dtype=tl.float32)
    
    l = tl.zeros((BLOCK_SIZE_M,),
                 dtype=tl.float32)
    
    acc = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_D),
                   dtype=tl.float32)
    
    end_n = tl.minimum((pid_m + 1) * BLOCK_SIZE_M,N)

    for kv_start in tl.range(0,end_n,BLOCK_SIZE_N):
        offs_n = kv_start + tl.arange(0,BLOCK_SIZE_N)

        k_ptrs = (K_base + offs_d[:,None] * stride_kd + offs_n[None,:] * stride_kn)

        k_mask = ((offs_d[:,None] < D) & (offs_n[None,:] < N))

        k = tl.load(k_ptrs,mask=k_mask,other=0.0)
        
        scores = tl.dot(q,k)

        scores = sm_scale * scores

        valid_keys = offs_n < N

        scores = tl.where(valid_keys[None,:],
                          scores,
                          -1.0e6)

        causal_mask = (offs_m[:,None] >= offs_n[None,:])

        scores = tl.where(causal_mask,
                          scores,
                          -1.0e6)
        
        block_max = tl.max(scores,axis=1)

        m_new = tl.maximum(m,block_max)

        alpha = tl.exp(m - m_new)

        p = tl.exp(scores - m_new[:,None])

        l = l * alpha + tl.sum(p,axis=1)

        v_ptrs = (V_base + offs_n[:,None] * stride_vn + offs_d[None,:] * stride_vd)

        v_mask = ((offs_n[:,None] < N) & (offs_d[None,:] < D))
        
        v = tl.load(v_ptrs,mask=v_mask,other=0.0)

        acc = acc * alpha[:,None] + tl.dot(p.to(tl.float16),v)

        m = m_new
    
    output = acc/ l[:,None]

    o_ptrs = (O_base + offs_m[:,None] * stride_on + offs_d[None,:] * stride_od)

    o_mask = ((offs_m[:,None] < N) & (offs_d[None,:] < D))

    lse = m + tl.log(l)

    lse_ptrs = LSE_base + offs_m * stride_lsen

    lse_mask = offs_m < N

    tl.store(o_ptrs,output,mask=o_mask)
    tl.store(lse_ptrs,lse,mask=lse_mask)

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                
            },
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
              
            },
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
               
            },
            num_warps=4,
            num_stages=3,
        ),

        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                
            },
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["N", "D"],
)

@triton.jit
def preprocess_backward_kernel(
    O_ptr,dO_ptr,Delta_ptr,

    B:tl.constexpr,
    H:tl.constexpr,
    N:tl.constexpr,
    D:tl.constexpr,

    stride_ob,stride_oh,
    stride_on,stride_od,

    stride_dob,stride_doh,
    stride_don,stride_dod,

    stride_deltab,stride_deltah,stride_deltan,

    BLOCK_SIZE_M:tl.constexpr,
    BLOCK_SIZE_D:tl.constexpr,

):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch = pid_bh // H
    head = pid_bh % H

    O_base = O_ptr + batch * stride_ob + head * stride_oh
    dO_base = dO_ptr + batch * stride_dob + head * stride_doh
    Delta_base = Delta_ptr + batch * stride_deltab + head * stride_deltah

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    offs_d = tl.arange(0,BLOCK_SIZE_D)
    
    o_ptrs = (O_base + offs_m[:,None] * stride_on + offs_d[None,:] * stride_od)

    o_mask = ((offs_m[:,None] < N) & (offs_d[None,:] < D))

    o = tl.load(o_ptrs,mask=o_mask,other=0.0).to(tl.float32)

    do_ptrs = (dO_base + offs_m[:,None] * stride_don + offs_d[None,:] * stride_dod)
    
    do_mask = ((offs_m[:,None] < N) & (offs_d[None,:] < D))

    do = tl.load(do_ptrs,mask=do_mask,other=0.0).to(tl.float32)

    delta = tl.sum( o * do ,axis=1)

    delta_ptrs = Delta_base + offs_m * stride_deltan

    delta_mask = offs_m < N

    tl.store(delta_ptrs,delta,mask=delta_mask)

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
            },
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
            },
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
            },
            num_warps=4,
            num_stages=3,
        ),

        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
            },
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["N", "D"],
)

@triton.jit
def flash_backward_dq_kernel(
    Q_ptr,K_ptr,V_ptr,dO_ptr,LSE_ptr,Delta_ptr,dQ_ptr,

    B:tl.constexpr,
    H:tl.constexpr,
    N:tl.constexpr,
    D:tl.constexpr,

    stride_qb,stride_qh,
    stride_qn,stride_qd,

    stride_kb,stride_kh,
    stride_kn,stride_kd,

    stride_vb,stride_vh,
    stride_vn,stride_vd,

    stride_dob,stride_doh,
    stride_don,stride_dod,

    stride_lseb,stride_lseh,stride_lsen,

    stride_deltab,stride_deltah,stride_deltan,

    stride_dqb,stride_dqh,
    stride_dqn,stride_dqd,

    sm_scale,

    BLOCK_SIZE_M:tl.constexpr,
    BLOCK_SIZE_N:tl.constexpr,
    BLOCK_SIZE_D:tl.constexpr

):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch = pid_bh // H
    head = pid_bh % H

    Q_base = Q_ptr + batch * stride_qb + head * stride_qh
    K_base = K_ptr + batch * stride_kb + head * stride_kh
    V_base = V_ptr + batch * stride_vb + head * stride_vh
    dO_base = dO_ptr + batch * stride_dob + head * stride_doh
    LSE_base = LSE_ptr + batch * stride_lseb + head * stride_lseh
    Delta_base = Delta_ptr + batch * stride_deltab + head * stride_deltah

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    offs_d = tl.arange(0,BLOCK_SIZE_D)

    q_ptrs = (Q_base + offs_m[:,None] * stride_qn + offs_d[None,:] * stride_qd)

    q_mask = ((offs_m[:,None] < N) & (offs_d[None,:] < D))

    q = tl.load(q_ptrs,mask=q_mask,other=0.0)

    do_ptrs = (dO_base + offs_m[:,None] * stride_don + offs_d[None,:] * stride_dod)

    do_mask = ((offs_m[:,None] < N) & (offs_d[None,:] < D))

    do = tl.load(do_ptrs,mask=do_mask,other=0.0)

    lse_ptrs = (LSE_base + offs_m * stride_lsen)

    lse_mask = (offs_m < N)

    lse = tl.load(lse_ptrs,mask=lse_mask,other=0.0)

    delta_ptrs = (Delta_base + offs_m * stride_deltan)

    delta_mask = (offs_m < N)

    delta = tl.load(delta_ptrs,mask=delta_mask,other=0.0)

    acc_dq = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_D),
                    dtype=tl.float32)

    end_n = tl.minimum((pid_m + 1) * BLOCK_SIZE_M,N)

    for kv_start in tl.range(0,end_n,BLOCK_SIZE_N):
        offs_n = kv_start + tl.arange(0,BLOCK_SIZE_N)

        k_ptrs = (K_base + offs_n[:,None] * stride_kn + offs_d[None,:] * stride_kd)

        k_mask = ((offs_n[:,None] < N) & (offs_d[None,:] < D))

        k = tl.load(k_ptrs,mask=k_mask,other=0.0)

        scores = tl.dot(q,tl.trans(k))

        scores = sm_scale * scores

        valid_queries = offs_m[:,None] < N

        valid_keys = offs_n[None,:] < N

        causal_mask = offs_m[:,None] >= offs_n[None,:]

        attention_mask = (
            valid_queries
                &
            valid_keys
                &
            causal_mask
        )

        scores = tl.where(attention_mask,
                        scores,
                        -1.0e6)
        
        p = tl.exp(scores - lse[:,None])

        v_ptrs = (V_base + offs_n[:,None] * stride_vn + offs_d[None,:] * stride_vd )

        v_mask = ((offs_n[:,None] < N) & (offs_d[None,:] < D))

        v = tl.load(v_ptrs,mask=v_mask,other=0.0)

        dp = tl.dot(do, tl.trans(v))

        ds = p * (dp - delta[:,None])

        acc_dq += tl.dot(ds.to(tl.float16),k) * sm_scale
    
    dQ_base = dQ_ptr + batch * stride_dqb + head * stride_dqh

    dq_ptrs = (dQ_base + offs_m[:,None] * stride_dqn + offs_d[None,:] * stride_dqd)

    dq_mask = ((offs_m[:,None] < N) & (offs_d[None,:] < D))

    tl.store(dq_ptrs,acc_dq,mask=dq_mask)

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
            },
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
            },
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
            },
            num_warps=4,
            num_stages=3,
        ),

        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
            },
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["N", "D"],
)

@triton.jit
def flash_backward_dk_dv_kernel(
    Q_ptr,K_ptr,V_ptr,
    dO_ptr,dK_ptr,dV_ptr,
    LSE_ptr,Delta_ptr,

    B:tl.constexpr,
    H:tl.constexpr,
    N:tl.constexpr,
    D:tl.constexpr,

    stride_qb,stride_qh,
    stride_qn,stride_qd,

    stride_kb,stride_kh,
    stride_kn,stride_kd,

    stride_vb,stride_vh,
    stride_vn,stride_vd,

    stride_dob,stride_doh,
    stride_don,stride_dod,

    stride_dkb,stride_dkh,
    stride_dkn,stride_dkd,

    stride_dvb,stride_dvh,
    stride_dvn,stride_dvd,

    stride_lseb,stride_lseh,stride_lsen,

    stride_deltab,stride_deltah,stride_deltan,

    sm_scale,

    BLOCK_SIZE_M:tl.constexpr,
    BLOCK_SIZE_N:tl.constexpr,
    BLOCK_SIZE_D:tl.constexpr,
):

    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch = pid_bh // H
    head = pid_bh % H

    Q_base = Q_ptr + batch * stride_qb + head * stride_qh
    K_base = K_ptr + batch * stride_kb + head * stride_kh
    V_base = V_ptr + batch * stride_vb + head * stride_vh
    dO_base = dO_ptr + batch * stride_dob + head * stride_doh
    LSE_base = LSE_ptr + batch * stride_lseb + head * stride_lseh
    Delta_base = Delta_ptr + batch * stride_deltab + head * stride_deltah

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
    offs_d = tl.arange(0,BLOCK_SIZE_D)

    k_ptrs = (K_base + offs_n[:,None] * stride_kn + offs_d[None,:] * stride_kd)

    k_mask = ((offs_n[:,None] < N) & (offs_d[None,:] < D))

    k = tl.load(k_ptrs,mask=k_mask,other=0.0)

    v_ptrs = (V_base + offs_n[:,None] * stride_vn + offs_d[None,:] * stride_vd)

    v_mask = ((offs_n[:,None] < N) & (offs_d[None,:] < D))

    v = tl.load(v_ptrs,mask=v_mask,other=0.0)

    acc_dk = tl.zeros((BLOCK_SIZE_N,BLOCK_SIZE_D),
                    dtype=tl.float32)
    
    acc_dv = tl.zeros((BLOCK_SIZE_N,BLOCK_SIZE_D),
                    dtype=tl.float32)
    
    # suppose there are three K and Q blocks and BLOCK_SIZE_M != BLOCK_SIZE_N,since K0 will interact with all query blocks (Q0,Q1,Q2)
    #K1 will interact with (Q1,Q2) and K2 will interact with only Q2
    # For BLOCK_SIZE_M == BLOCK_SIZE_N first_q = pid_m * BLOCK_SIZE_M
    # Suppose BLOCK_SIZE_M = BLOCK_SIZE_N = 16,K0 has 16(0,15) rows and these rows need for all (0 to 47) QUERY rows but K1 has (16,31) rows then query blocks Q0 don't need this
     
    first_q = ((pid_n * BLOCK_SIZE_N)//BLOCK_SIZE_M) * BLOCK_SIZE_M

    for q_start in tl.range(first_q,N,BLOCK_SIZE_M):
        offs_m = q_start + tl.arange(0,BLOCK_SIZE_M)

        q_ptrs = (Q_base + offs_m[:,None] * stride_qn + offs_d[None,:] * stride_qd)

        q_mask = ((offs_m[:,None] < N) & (offs_d[None,:] < D))

        q = tl.load(q_ptrs,mask=q_mask,other=0.0)

        lse_ptrs = (LSE_base + offs_m * stride_lsen)

        lse_mask = (offs_m < N)

        lse = tl.load(lse_ptrs,mask=lse_mask,other=0.0)

        do_ptrs = (dO_base + offs_m[:,None] * stride_don + offs_d[None,:] * stride_dod)

        do_mask = ((offs_m[:,None] < N) & (offs_d[None,:] < D))

        do = tl.load(do_ptrs,mask=do_mask,other=0.0)

        delta_ptrs = Delta_base + offs_m * stride_deltan

        delta_mask = (offs_m < N)

        delta = tl.load(delta_ptrs,mask=delta_mask,other=0.0)

        scores = tl.dot(q,tl.trans(k))

        scores *= sm_scale

        valid_queries = offs_m[:,None] < N

        valid_keys = offs_n[None,:] < N

        causal = offs_m[:,None] >= offs_n[None,:]

        attention_mask = (
            valid_queries
                &
            valid_keys
                &
            causal
        )

        scores = tl.where(attention_mask,
                        scores,
                        -1.0e6)
        
        p = tl.exp(scores - lse[:,None])

        dp = tl.dot(do,tl.trans(v))

        ds = p * (dp - delta[:,None])

        acc_dk += tl.dot(tl.trans(ds.to(tl.float16)),q) * sm_scale

        acc_dv += tl.dot(tl.trans(p.to(tl.float16)),do)

    dK_base = dK_ptr + batch * stride_dkb + head * stride_dkh
    dV_base = dV_ptr + batch * stride_dvb + head * stride_dvh

    dk_ptrs = (dK_base + offs_n[:,None] * stride_dkn + offs_d[None,:] * stride_dkd)
    dk_mask = ((offs_n[:,None] < N) & (offs_d[None,:] < D))

    dv_ptrs = (dV_base + offs_n[:,None] * stride_dvn + offs_d[None,:] * stride_dvd)
    dv_mask = ((offs_n[:,None] < N) & (offs_d[None,:] < D))

    tl.store(dk_ptrs,acc_dk,mask=dk_mask)
    tl.store(dv_ptrs,acc_dv,mask=dv_mask)

def flash_attention_forward(q,k,v):
     
    assert q.ndim == 4

    assert q.shape == k.shape
    assert q.shape == v.shape

    assert q.is_cuda
    assert k.is_cuda
    assert v.is_cuda

    B,H,N,D = q.shape

    # output has same shape as q
    o = torch.empty_like(q)

    sm_scale = 1/ math.sqrt(D)

    # lse has shape (B,H,N)
    lse = torch.empty((B,H,N),
                    device=q.device,
                    dtype=torch.float32)
    
    
    BLOCK_SIZE_D = triton.next_power_of_2(D)

    forward_grid = lambda META: (
    triton.cdiv(N, META["BLOCK_SIZE_M"]),
    B * H,)

    flash_attention_forward_kernel[forward_grid](
        q,k,v,o,lse,

        B,H,N,D,

        q.stride(0),q.stride(1),
        q.stride(2),q.stride(3),

        k.stride(0),k.stride(1),
        k.stride(2),k.stride(3),

        v.stride(0),v.stride(1),
        v.stride(2),v.stride(3),

        o.stride(0),o.stride(1),
        o.stride(2),o.stride(3),

        lse.stride(0),lse.stride(1),lse.stride(2),

        sm_scale,

        
        BLOCK_SIZE_D = BLOCK_SIZE_D,

        
    )

    return o,lse

def flash_attention_backward(q,k,v,o,lse,do):

    B,H,N,D = q.shape

    sm_scale = 1 / math.sqrt(D)

    BLOCK_SIZE_D = triton.next_power_of_2(D)

    # Allocate Final Gradient
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    delta = torch.empty((B,H,N),
                        device=q.device,
                        dtype=torch.float32)
    
    preprocess_grid = lambda META: (
    triton.cdiv(N, META["BLOCK_SIZE_M"]),
    B * H,)
    preprocess_backward_kernel[preprocess_grid](
        o,do,delta,

        B,H,N,D,

        o.stride(0),o.stride(1),
        o.stride(2),o.stride(3),
        
        do.stride(0),do.stride(1),
        do.stride(2),do.stride(3),

        delta.stride(0),delta.stride(1),delta.stride(2),

        BLOCK_SIZE_D = BLOCK_SIZE_D,

        

    )

    dq_grid = lambda META: (
    triton.cdiv(N, META["BLOCK_SIZE_M"]),
    B * H,)

    flash_backward_dq_kernel[dq_grid](
        q,k,v,do,lse,delta,dq,

        B,H,N,D,

        q.stride(0),q.stride(1),
        q.stride(2),q.stride(3),

        k.stride(0),k.stride(1),
        k.stride(2),k.stride(3),

        v.stride(0),v.stride(1),
        v.stride(2),v.stride(3),

        do.stride(0),do.stride(1),
        do.stride(2),do.stride(3),

        lse.stride(0),lse.stride(1),lse.stride(2),

        delta.stride(0),delta.stride(1),delta.stride(2),

        dq.stride(0),dq.stride(1),
        dq.stride(2),dq.stride(3),

        sm_scale,

        BLOCK_SIZE_D = BLOCK_SIZE_D

       

    )

    dk_dv_grid = lambda META: (
    triton.cdiv(N, META["BLOCK_SIZE_N"]),
    B * H,
)
    flash_backward_dk_dv_kernel[dk_dv_grid](
        q,k,v,do,dk,dv,lse,delta,

        B,H,N,D,

        q.stride(0),q.stride(1),
        q.stride(2),q.stride(3),

        k.stride(0),k.stride(1),
        k.stride(2),k.stride(3),

        v.stride(0),v.stride(1),
        v.stride(2),v.stride(3),

        do.stride(0),do.stride(1),
        do.stride(2),do.stride(3),

        dk.stride(0),dk.stride(1),
        dk.stride(2),dk.stride(3),

        dv.stride(0),dv.stride(1),
        dv.stride(2),dv.stride(3),

        lse.stride(0),lse.stride(1),lse.stride(2),

        delta.stride(0),delta.stride(1), delta.stride(2),

        sm_scale,

        
        BLOCK_SIZE_D = BLOCK_SIZE_D
      
    )

    return dq,dk,dv



def test_flash_attention_case(B, H, N, D, seed=0):
    torch.manual_seed(seed)

    # ---------------------------------------------------------
    # Inputs
    # ---------------------------------------------------------

    q = torch.randn(
        (B, H, N, D),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )

    k = torch.randn(
        (B, H, N, D),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )

    v = torch.randn(
        (B, H, N, D),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )

    # Incoming gradient from next layer
    do = torch.randn_like(q)

    sm_scale = 1.0 / math.sqrt(D)

    # =========================================================
    # PyTorch reference
    # =========================================================

    scores_ref = (
        q.float()
        @ k.float().transpose(-2, -1)
    ) * sm_scale

    causal_mask = torch.tril(
        torch.ones(
            (N, N),
            device=q.device,
            dtype=torch.bool,
        )
    )

    scores_ref = scores_ref.masked_fill(
        ~causal_mask,
        float("-inf"),
    )

    p_ref = torch.softmax(
        scores_ref,
        dim=-1,
    )

    o_ref = (
        p_ref
        @ v.float()
    ).to(torch.float16)

    # Reference LSE
    lse_ref = torch.logsumexp(
        scores_ref,
        dim=-1,
    )

    # Backpropagate supplied dO
    o_ref.backward(do)

    dq_ref = q.grad.detach().clone()
    dk_ref = k.grad.detach().clone()
    dv_ref = v.grad.detach().clone()

    # =========================================================
    # Triton implementation
    # =========================================================

    with torch.no_grad():

        o, lse = flash_attention_forward(
            q.detach(),
            k.detach(),
            v.detach(),
        )

        dq, dk, dv = flash_attention_backward(
            q.detach(),
            k.detach(),
            v.detach(),
            o,
            lse,
            do,
        )

    # =========================================================
    # Errors
    # =========================================================

    errors = {
        "O_max":  (o - o_ref).abs().max().item(),
        "O_mean": (o - o_ref).abs().mean().item(),

        "LSE_max":  (lse.float() - lse_ref).abs().max().item(),
        "LSE_mean": (lse.float() - lse_ref).abs().mean().item(),

        "dQ_max":  (dq - dq_ref).abs().max().item(),
        "dQ_mean": (dq - dq_ref).abs().mean().item(),

        "dK_max":  (dk - dk_ref).abs().max().item(),
        "dK_mean": (dk - dk_ref).abs().mean().item(),

        "dV_max":  (dv - dv_ref).abs().max().item(),
        "dV_mean": (dv - dv_ref).abs().mean().item(),
    }

    return errors

test_cases = [
    # B, H, N, D
    (1, 1, 17, 16),
    (2, 2, 32, 32),
    (2, 2, 37, 16),
    (2, 2, 37, 32),
    (2, 4, 48, 64),
    (1, 4, 65, 32),
]


for B, H, N, D in test_cases:

    print("=" * 70)
    print(
        f"B={B}, H={H}, N={N}, D={D}"
    )

    try:
        err = test_flash_attention_case(
            B, H, N, D
        )

        print(
            f"O   | max={err['O_max']:.8f} "
            f"| mean={err['O_mean']:.8f}"
        )

        print(
            f"LSE | max={err['LSE_max']:.8f} "
            f"| mean={err['LSE_mean']:.8f}"
        )

        print(
            f"dQ  | max={err['dQ_max']:.8f} "
            f"| mean={err['dQ_mean']:.8f}"
        )

        print(
            f"dK  | max={err['dK_max']:.8f} "
            f"| mean={err['dK_mean']:.8f}"
        )

        print(
            f"dV  | max={err['dV_max']:.8f} "
            f"| mean={err['dV_mean']:.8f}"
        )
        print("Forward:",
              flash_attention_forward_kernel.best_config)

        print("dQ:",
             flash_backward_dq_kernel.best_config)

        print("dK/dV:",
                flash_backward_dk_dv_kernel.best_config) 

    except Exception as e:

        print("FAILED:")
        print(e)


def benchmark_case(B, H, N, D):

    q = torch.randn(
        B, H, N, D,
        device="cuda",
        dtype=torch.float16,
    )

    k = torch.randn_like(q)
    v = torch.randn_like(q)
    do = torch.randn_like(q)

    # -----------------------------
    # Triton forward
    # -----------------------------

    triton_fwd_ms = triton.testing.do_bench(
        lambda: flash_attention_forward(q, k, v)
    )

    o, lse = flash_attention_forward(q, k, v)

    # -----------------------------
    # Triton backward
    # -----------------------------

    triton_bwd_ms = triton.testing.do_bench(
        lambda: flash_attention_backward(
            q, k, v, o, lse, do
        )
    )

    # -----------------------------
    # PyTorch forward
    # -----------------------------

    torch_fwd_ms = triton.testing.do_bench(
        lambda: F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
    )

    # -----------------------------
    # PyTorch backward
    # -----------------------------

    q_ref = q.detach().requires_grad_(True)
    k_ref = k.detach().requires_grad_(True)
    v_ref = v.detach().requires_grad_(True)

    out_ref = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        is_causal=True,
    )

    def torch_backward():
        torch.autograd.grad(
            out_ref,
            (q_ref, k_ref, v_ref),
            do,
            retain_graph=True,
        )

    torch_bwd_ms = triton.testing.do_bench(
        torch_backward
    )

    print(f"B={B}, H={H}, N={N}, D={D}")
    print(f"Triton forward : {triton_fwd_ms:.4f} ms")
    print(f"PyTorch forward : {torch_fwd_ms:.4f} ms")
    print(f"Triton backward: {triton_bwd_ms:.4f} ms")
    print(f"PyTorch backward: {torch_bwd_ms:.4f} ms")
    print()


# IMPORTANT: this is OUTSIDE benchmark_case
for N in [128, 256, 512, 1024, 2048]:
    benchmark_case(
        B=2,
        H=8,
        N=N,
        D=64,
    )









