# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
import triton_gcu.triton
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_gcu._C import _gcu_getCurrentRawStream as get_raw_stream
from torch_gcu._C import _gcu_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/x6/cx66nil2rmjz74tu777dbbttcltjle4nv3gzw5sts6gspkwdtu3k.py
# Topologically Sorted Source Nodes: [linear, x], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   linear => add_tensor_2
#   x => relu
# Graph fragment:
#   %add_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %arg1_1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_2,), kwargs = {})
triton_poi_fused_addmm_relu_0 = async_compile.triton('triton_poi_fused_addmm_relu_0', '''
import torch_gcu
import triton
import triton_gcu.triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, AttrsDescriptorWrapper, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='gcu', index=0, multi_processor_count=3, cc=(4, 0), major=4, regs_per_multiprocessor=None, max_threads_per_multi_processor=6, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '001D65F11414053AF2F4D82627D97ED080D0018A2CB2662EE975BE4193927DDA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    # GCU loop support pointwise and persistent_reduction among XBLOCK
    gridsize = tl.num_programs(0)
    xloop_num = (xnumel + XBLOCK - 1) // XBLOCK
    for xloop in range(tl.program_id(0), xloop_num, gridsize):
        xoffset = xloop * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        x2 = xindex
        x0 = (xindex % 256)
        tmp0 = tl.load(in_out_ptr0 + (x2), None)
        tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1], 0, tl.int32)
        tmp4 = triton_helpers.maximum(tmp3, tmp2)
        tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='gcu')


# kernel path: /tmp/torchinductor_root/at/catjfczfwogu5lt3iobqbhnhu6qfzlqigudwagnu473mdxnywusw.py
# Topologically Sorted Source Nodes: [linear_2, x_2], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   linear_2 => add_tensor
#   x_2 => relu_2
# Graph fragment:
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg6_1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
triton_poi_fused_addmm_relu_1 = async_compile.triton('triton_poi_fused_addmm_relu_1', '''
import torch_gcu
import triton
import triton_gcu.triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, AttrsDescriptorWrapper, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='gcu', index=0, multi_processor_count=3, cc=(4, 0), major=4, regs_per_multiprocessor=None, max_threads_per_multi_processor=6, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '001D65F11414053AF2F4D82627D97ED080D0018A2CB2662EE975BE4193927DDA', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    # GCU loop support pointwise and persistent_reduction among XBLOCK
    gridsize = tl.num_programs(0)
    xloop_num = (xnumel + XBLOCK - 1) // XBLOCK
    for xloop in range(tl.program_id(0), xloop_num, gridsize):
        xoffset = xloop * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 128)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1], 0, tl.int32)
        tmp4 = triton_helpers.maximum(tmp3, tmp2)
        tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='gcu')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
    args.clear()
    assert_size_stride(arg0_1, (256, 256), (256, 1))
    assert_size_stride(arg1_1, (256, ), (1, ))
    assert_size_stride(arg2_1, (16, 256), (256, 1))
    assert_size_stride(arg3_1, (256, 256), (256, 1))
    assert_size_stride(arg4_1, (256, ), (1, ))
    assert_size_stride(arg5_1, (128, 256), (256, 1))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (10, 128), (128, 1))
    assert_size_stride(arg8_1, (10, ), (1, ))
    with torch.gcu._DeviceGuard(0):
        torch.gcu.set_device(0)
        buf0 = empty_strided((16, 256), (256, 1), device='gcu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(arg2_1, reinterpret_tensor(arg0_1, (256, 256), (1, 256), 0), out=buf0)
        del arg0_1
        del arg2_1
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [linear, x], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf1, arg1_1, 4096, stream=stream0)
        del arg1_1
        buf2 = empty_strided((16, 256), (256, 1), device='gcu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [linear, x, linear_1], Original ATen: [aten.addmm, aten.relu]
        extern_kernels.mm(buf1, reinterpret_tensor(arg3_1, (256, 256), (1, 256), 0), out=buf2)
        del arg3_1
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [linear_1, x_1], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf3, arg4_1, 4096, stream=stream0)
        del arg4_1
        buf4 = empty_strided((16, 128), (128, 1), device='gcu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [linear_1, x_1, linear_2], Original ATen: [aten.addmm, aten.relu]
        extern_kernels.mm(buf3, reinterpret_tensor(arg5_1, (256, 128), (1, 256), 0), out=buf4)
        del arg5_1
        del buf3
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [linear_2, x_2], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_1.run(buf5, arg6_1, 2048, stream=stream0)
        del arg6_1
        buf6 = empty_strided((16, 10), (10, 1), device='gcu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [linear_2, x_2, x_3], Original ATen: [aten.addmm, aten.relu]
        extern_kernels.addmm(arg8_1, buf5, reinterpret_tensor(arg7_1, (128, 10), (1, 128), 0), alpha=1, beta=1, out=buf6)
        del arg7_1
        del arg8_1
        del buf5
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((256, 256), (256, 1), device='gcu:0', dtype=torch.float32)
    arg1_1 = rand_strided((256, ), (1, ), device='gcu:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, 256), (256, 1), device='gcu:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, 256), (256, 1), device='gcu:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, ), (1, ), device='gcu:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, 256), (256, 1), device='gcu:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='gcu:0', dtype=torch.float32)
    arg7_1 = rand_strided((10, 128), (128, 1), device='gcu:0', dtype=torch.float32)
    arg8_1 = rand_strided((10, ), (1, ), device='gcu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
