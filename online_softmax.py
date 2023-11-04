
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time

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

def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """ online softmax, 2.x faster than eager mode algo """
    row_count, col_count = x.shape
    assert x.dim()==2, f"Input must be 2D tensor, got {x.dim()}"

    # In triton, we have to have output buffer ready
    output = torch.zeros_like(x)

    # Setting up 2 loops here for row and col
    for r in range(row_count):
        row_max = 0
        normalizer = 0 # for flash attention, online softmax is used
        for c in range(col_count):
            curr = x[r, c]
            prev_old_max = row_max
            row_max = max(row_max, curr)
            if row_max > prev_old_max:
                print(f"row_max: {row_max} updated as prev_old_max: {prev_old_max} for row: {r}, col: {c}")

            # if difference, then we need to normalize
            normalizer = normalizer * torch.exp(prev_old_max - row_max) + torch.exp(curr - row_max)

        # Effectively doing safe_x here
        output[r,:] = torch.exp(x[r,:] - row_max) / normalizer

        return output

#### Sample unit test ####

sample = torch.tensor([[1,2,3,4,5],[5, 4,3,2,1]], dtype=torch.float32, device='cuda')
start = time.perf_counter()
eager_out = naive_softmax(sample)
stop = time.perf_counter()
eager_time = stop-start
start = time.perf_counter()
online_out = online_softmax(sample)
stop = time.perf_counter()
online_time = stop-start
ref_out = torch.softmax(sample, dim=1)
print(f"{eager_out=}\n{online_out=}\n{ref_out=}\n")
print(f"{eager_time=}, {online_time=}")
