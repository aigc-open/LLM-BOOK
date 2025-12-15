import torch
import torch.fx as fx



@torch.compile
def forward(self, x):
    if x.sum() > 0:  # [√] 完全支持！
        return x * 2
    else:
        return x + 1

output = forward(torch.randn(10))  # 工作正常

from torch._dynamo.convert_frame import convert_frame
callback.convert_frame_assert = convert_frame_assert