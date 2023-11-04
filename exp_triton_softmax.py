# from SOTA Deep Learning Tutorials - https://youtube.com/playlist?list=PLSXcJOyFhmS-qb_CF-GLhkWxSmi-ftbPO&si=8NLQ7FSeCaDN9_lO

# Code eager softmax in PyTorch, Triton

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """ eager mode Softmax """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0] # max returns (values, indices), but we just want to print value
    # print(f"{x_max=}")

    # We subtract the maximum element in order to avoid overflows. Softmax is invariant to this shift.
    # read MN + M elements ; write MN elements
    safe_x = x - x_max[:, None]
    # print(f"{safe_x=}")

    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1)
    sm_out = numerator/denominator[:, None]
    return sm_out

# Make a triton kernel
@triton.jit
def _softmax_fwd_kernel(
    output_ptr,
    stride_output_row,
    input_ptr,
    stride_input_row,
    num_cols,
    block_size: tl.constexpr,
    # sm_out: triton.Tensor, sm_out_stride: int,
    # x: triton.Tensor, x_stride: int,
    # cols: int,
    # *,
    # block_size: int,
    # num_warps: int
):
    # pass
    """ Triton impl of Softmax, fwd pass only """
    # setup input ptrs
    row_index = tl.program_id(0)
    row_start_ptr = input_ptr + (row_index * stride_input_row)

    # setup column offsets
    col_offset = tl.arange(0, block_size)
    input_pointers = row_start_ptr + col_offset
    # So, now that we got full access to the entire row (?) pointers, we can pick up the row from HBM to SRAM

    # Setup generic mask
    row_mask = col_offset < num_cols

    # move to SRAM, and apply mask so dont inadvertently read from other 
    row = tl.load(input_pointers, mask = row_mask, other=float("-inf"))

    # Now start the softmax process
    # row_max = tl.max(row, axis=0)
    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    sm_out = numerator/denominator

    # Pointer arithmeitc to get to the right output row back to HBM
    output_row_ptr = output_ptr + (row_index * stride_output_row)
    ouptut_pointers = output_row_ptr + col_offset
    tl.store(ouptut_pointers, sm_out, mask=row_mask)

def softmax(x:torch.Tensor) -> torch.Tensor:
    """ Triton impl of Softmax, fwd pass only """

    # Each row to become its own kernel isntance ==> if 1000 rows, then, 1000 kernel instances launched, each processing a row
    rows, cols = x.shape
    assert x.dim() == 2, f"Currently, softmax only supports 2D tensors, got {x.dim()}"

    # block_size related to chunks of data in each row (cols) that each kernel instance will process
    # so example, if we have 1000 columns, the block size will be 1024; last chunk will be 1000 - 1024 = 24 additional space to handle and masking needs to be done appropriately
    block_size = triton.next_power_of_2(cols)
    num_warps = 4  # 32 # Thread density
    if block_size > 2047: # 2048
        num_warps = 8 # double
    if block_size > 4095: # 4096
        num_warps = 16

    grid = (rows, )

    # Allocate output buffer
    sm_out = torch.empty_like(x)

    # Launch kernel
    _softmax_fwd_kernel[grid](
        sm_out,
        sm_out.stride(0),
        x,
        x.stride(0),
        cols,
        block_size=block_size,
        num_warps=num_warps
    )
    # _softmax_fwd_kernel[grid, num_warps, block_size](x, sm_out, rows, cols, block_size)

    return sm_out

sample = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32, device='cuda')
# sample = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float32, device='cuda')
ref_out = F.softmax(sample, dim=1)
print(f"{ref_out=}")

eager_out = naive_softmax(sample)
print(f"{eager_out=}")

triton_out = softmax(sample)
print(f"{triton_out=}")