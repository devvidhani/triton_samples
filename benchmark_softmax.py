import torch
import triton
import triton.language as tl

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """ eager mode Softmax """
    x_max = x.max(dim=1)[0] # max returns (values, indices), but we just want to print value
    # print(f"{x_max=}")
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

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 100)
        ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'triton',
            'torch-native',
            'torch-jit',
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch (native)",
            "Torch (jit)",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('green', '--')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    if provider == 'torch-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


from pathlib import Path
benchmark.run(show_plots=True, print_data=True, save_path=Path.cwd())
