import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def vector_add_kernel(a_ptr,b_ptr,output_ptr,num_elements,BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    
    offsets = pid*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offsets < num_elements

    a = tl.load(a_ptr + offsets,mask=mask)
    b = tl.load(b_ptr + offsets,mask=mask)
    
    output = a + b

    tl.store(output_ptr + offsets,output,mask=mask)

def add(a:torch.tensor, b:torch.tensor):
    output = torch.empty_like(a)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    
    num_elements = output.numel()

    grid = lambda meta: (tl.cdiv(num_elements,meta['BLOCK_SIZE ']),)

    vector_add_kernel[grid](a,b,output,num_elements,BLOCK_SIZE=1024)

    return output

torch.manual_seed(0)
size = 98432
x = torch.rand(size,)
y = torch.rand(size,)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')


