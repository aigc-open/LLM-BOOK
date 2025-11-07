# NPU 适配实践指南 - 保持客户使用方式不变

## 场景说明

**问题**：NPU 已经支持 Triton，但某些算子在 NPU 上可能需要不同的实现（性能优化、内存布局差异等），同时希望保持客户使用方式完全一致。

**解决方案**：使用条件 Monkey Patch，根据设备类型自动选择最优实现。

---

## 核心思路

```python
# 客户代码保持不变
from liger_kernel.transformers import apply_liger_kernel_to_llama

apply_liger_kernel_to_llama()  # 自动检测设备并应用最优实现
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
```

**内部实现**：
1. 检测当前设备类型（CUDA / NPU / AMD）
2. 为不同设备提供特化实现
3. 通过 Monkey Patch 统一接口

---

## 实践步骤

### 步骤 1：创建设备检测工具

```python
# src/liger_kernel/utils.py 或创建新文件

import torch

def infer_device():
    """
    推断当前设备类型
    """
    if torch.cuda.is_available():
        # 检查是否是 NPU（假设 NPU 有特定属性）
        try:
            device_name = torch.cuda.get_device_name(0)
            if "NPU" in device_name or "Ascend" in device_name:
                return "npu"
            elif "AMD" in device_name:
                return "amd"
            else:
                return "cuda"
        except:
            return "cuda"
    return "cpu"


def is_npu_available():
    """检查是否有 NPU 可用"""
    return infer_device() == "npu"
```

### 步骤 2：为 NPU 创建特化算子

假设您需要为 NPU 优化 RMSNorm 算子：

```python
# src/liger_kernel/ops/rms_norm_npu.py

import torch
import triton
import triton.language as tl

# NPU 特化的 Triton 内核
@triton.jit
def _rms_norm_forward_kernel_npu(
    Y_ptr,
    X_ptr,
    W_ptr,
    RSTD_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    针对 NPU 优化的 RMSNorm 前向内核
    
    优化点：
    1. 调整 BLOCK_SIZE 适配 NPU 内存层次
    2. 优化内存访问模式
    3. 使用 NPU 特定指令
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 加载数据 - NPU 优化：使用更大的向量化宽度
    X_ptr += row_idx * n_cols
    Y_ptr += row_idx * n_cols
    
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    
    # 计算 - NPU 优化：可能使用不同的精度策略
    X_row = X_row.to(tl.float32)  # NPU 可能在 FP32 计算更快
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_square + eps)
    
    # 存储 rstd
    tl.store(RSTD_ptr + row_idx, rstd)
    
    # 归一化
    Y_row = X_row * rstd * W_row
    
    # 存储结果 - NPU 优化：使用连续内存写入
    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_norm_backward_kernel_npu(
    dY_ptr,
    X_ptr,
    W_ptr,
    RSTD_ptr,
    dX_ptr,
    dW_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """NPU 优化的反向传播内核"""
    # NPU 特化的反向传播实现
    # 类似的优化策略
    pass


class LigerRMSNormFunctionNPU(torch.autograd.Function):
    """NPU 优化的 RMSNorm Autograd Function"""
    
    @staticmethod
    def forward(ctx, X, W, eps, offset, casting_mode):
        # 使用 NPU 优化的内核
        n_rows, n_cols = X.shape
        Y = torch.empty_like(X)
        RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        
        # NPU 特定的 BLOCK_SIZE 选择
        BLOCK_SIZE = calculate_npu_block_size(n_cols)
        
        # 启动 NPU 优化的内核
        grid = (n_rows,)
        _rms_norm_forward_kernel_npu[grid](
            Y, X, W, RSTD,
            n_cols, eps, BLOCK_SIZE
        )
        
        ctx.save_for_backward(X, W, RSTD)
        ctx.n_cols = n_cols
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return Y
    
    @staticmethod
    def backward(ctx, dY):
        X, W, RSTD = ctx.saved_tensors
        dX = torch.empty_like(X)
        dW = torch.empty_like(W)
        
        # NPU 优化的反向传播
        _rms_norm_backward_kernel_npu[(ctx.n_rows,)](
            dY, X, W, RSTD, dX, dW,
            ctx.n_cols, ctx.BLOCK_SIZE
        )
        
        return dX, dW, None, None, None


def calculate_npu_block_size(n_cols):
    """
    根据 NPU 特性计算最优 BLOCK_SIZE
    
    NPU 可能有不同的：
    - 向量寄存器宽度
    - L1 cache 大小
    - 内存带宽特性
    """
    # 示例：NPU 在 4096 块大小时性能最佳
    if n_cols <= 2048:
        return 2048
    elif n_cols <= 4096:
        return 4096
    else:
        return 8192
```

### 步骤 3：创建设备感知的 Module

```python
# src/liger_kernel/transformers/rms_norm.py

import torch.nn as nn
from liger_kernel.utils import is_npu_available
from liger_kernel.ops.rms_norm import LigerRMSNormFunction  # CUDA 版本
from liger_kernel.ops.rms_norm_npu import LigerRMSNormFunctionNPU  # NPU 版本


class LigerRMSNorm(nn.Module):
    """
    设备感知的 RMSNorm 模块
    自动选择最优实现，对用户透明
    """
    
    def __init__(self, hidden_size, eps=1e-6, offset=0.0, 
                 casting_mode="llama", init_fn="ones"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.offset = offset
        self.casting_mode = casting_mode
        
        # 根据设备选择实现
        self._select_implementation()
    
    def _select_implementation(self):
        """根据当前设备选择最优实现"""
        if is_npu_available():
            self._forward_impl = LigerRMSNormFunctionNPU.apply
            self._device_type = "npu"
        else:
            self._forward_impl = LigerRMSNormFunction.apply
            self._device_type = "cuda"
    
    def forward(self, hidden_states):
        """
        前向传播 - 自动使用最优实现
        用户无需关心底层设备
        """
        return self._forward_impl(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode
        )
    
    def extra_repr(self):
        """显示当前使用的设备实现"""
        return f"device={self._device_type}, hidden_size={self.weight.shape[0]}, eps={self.variance_epsilon}"
```

### 步骤 4：更新 Monkey Patch 函数（保持接口不变）

```python
# src/liger_kernel/transformers/monkey_patch.py

from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.utils import infer_device


def apply_liger_kernel_to_llama(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    为 LLaMA 应用 Liger 内核
    
    自动检测设备并应用最优实现：
    - NVIDIA GPU: 使用 CUDA 优化
    - NPU: 使用 NPU 特化版本
    - AMD GPU: 使用 ROCm 优化
    
    客户使用方式完全一致！
    """
    from transformers.models.llama import modeling_llama
    
    # 检测设备（可选：打印日志）
    device_type = infer_device()
    logger.info(f"Detected device: {device_type}, applying optimized kernels")
    
    # 替换 RMSNorm - 自动使用设备感知版本
    if rms_norm:
        # LigerRMSNorm 内部会自动选择 NPU/CUDA 实现
        modeling_llama.LlamaRMSNorm = LigerRMSNorm
    
    # 其他算子的替换
    if rope:
        modeling_llama.apply_rotary_pos_emb = liger_rotary_pos_emb
    
    if swiglu:
        # SwiGLU 也可以做类似的设备感知实现
        modeling_llama.LlamaMLP = LigerSwiGLUMLP
    
    if cross_entropy:
        from transformers.loss.loss_utils import nn
        nn.functional.cross_entropy = liger_cross_entropy
    
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(llama_lce_forward, model)
        else:
            modeling_llama.LlamaForCausalLM.forward = llama_lce_forward
    
    # 实例级 patch（如果需要）
    if model is not None:
        base_model = getattr(model, model.base_model_prefix, model)
        
        if rms_norm:
            _patch_rms_norm_module(base_model.norm)
        
        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
```

---

## 完整使用示例

### 客户端代码（完全不变）

```python
# train.py - 客户代码无需任何修改

from liger_kernel.transformers import apply_liger_kernel_to_llama
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. 应用 Liger 优化（自动检测设备）
apply_liger_kernel_to_llama()

# 2. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",  # 自动分配到 NPU 或 GPU
)

# 3. 正常训练
trainer = Trainer(model=model, ...)
trainer.train()
```

**运行结果**：
- 🖥️ 在 NVIDIA GPU 上：自动使用 CUDA 优化内核
- 🔧 在 NPU 上：自动使用 NPU 特化内核
- 💻 客户代码完全相同，无需任何修改！

---

## 高级用法：部分算子使用 NPU 优化

### 场景：只优化部分算子

假设您只想为 NPU 优化 RMSNorm 和 CrossEntropy，其他算子使用通用实现：

```python
# src/liger_kernel/transformers/monkey_patch.py

def apply_liger_kernel_to_llama(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    force_device: str = None,  # 新增：强制指定设备类型
) -> None:
    """支持强制指定设备类型用于测试"""
    from transformers.models.llama import modeling_llama
    
    device_type = force_device or infer_device()
    
    # 根据设备和算子类型选择实现
    if rms_norm:
        if device_type == "npu":
            # NPU 使用特化版本
            modeling_llama.LlamaRMSNorm = LigerRMSNormNPU
        else:
            # 其他设备使用通用版本
            modeling_llama.LlamaRMSNorm = LigerRMSNorm
    
    if cross_entropy:
        if device_type == "npu":
            # NPU 使用特化的 CrossEntropy
            from liger_kernel.transformers.functional_npu import liger_cross_entropy_npu
            from transformers.loss.loss_utils import nn
            nn.functional.cross_entropy = liger_cross_entropy_npu
        else:
            # 通用版本
            from transformers.loss.loss_utils import nn
            nn.functional.cross_entropy = liger_cross_entropy
    
    # SwiGLU 可能在 NPU 上不需要特化，使用通用版本
    if swiglu:
        modeling_llama.LlamaMLP = LigerSwiGLUMLP  # 所有设备通用
    
    # ... 其他算子
```

---

## 调试和性能分析

### 1. 验证正确性

```python
# test/test_npu_correctness.py

import pytest
import torch
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.utils import is_npu_available


@pytest.mark.skipif(not is_npu_available(), reason="NPU not available")
def test_rms_norm_npu_correctness():
    """验证 NPU 实现的正确性"""
    hidden_size = 4096
    batch_size = 2
    seq_len = 128
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, hidden_size, device="npu", dtype=torch.bfloat16)
    x.requires_grad = True
    
    # 参考实现（CPU）
    x_cpu = x.detach().cpu().float()
    x_cpu.requires_grad = True
    ref_norm = torch.nn.RMSNorm(hidden_size).cpu()
    ref_output = ref_norm(x_cpu)
    ref_output.sum().backward()
    ref_grad = x_cpu.grad
    
    # NPU 实现
    x.grad = None
    npu_norm = LigerRMSNorm(hidden_size).to("npu")
    npu_norm.weight.data.copy_(ref_norm.weight.data)
    npu_output = npu_norm(x)
    npu_output.sum().backward()
    npu_grad = x.grad
    
    # 验证（允许一定误差）
    assert torch.allclose(
        npu_output.cpu().float(), 
        ref_output, 
        atol=1e-2, rtol=1e-2
    ), "NPU forward output mismatch"
    
    assert torch.allclose(
        npu_grad.cpu().float(),
        ref_grad,
        atol=1e-2, rtol=1e-2
    ), "NPU backward gradient mismatch"


def test_cross_device_compatibility():
    """验证在不同设备上的兼容性"""
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    
    # 应用 patch
    apply_liger_kernel_to_llama()
    
    # 加载模型到不同设备应该都能工作
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if is_npu_available():
        devices.append("npu")
    
    for device in devices:
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            device_map=device,
        )
        
        # 简单前向传播
        input_ids = torch.randint(0, 1000, (2, 10), device=device)
        output = model(input_ids)
        
        assert output.logits is not None
        print(f"✓ Model works on {device}")
```

### 2. 性能对比

```python
# benchmark/benchmark_npu_vs_cuda.py

import torch
import time
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.utils import is_npu_available


def benchmark_device(device, hidden_size=4096, seq_len=2048, num_runs=100):
    """在指定设备上进行性能测试"""
    x = torch.randn(8, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    norm = LigerRMSNorm(hidden_size).to(device)
    
    # 预热
    for _ in range(10):
        _ = norm(x)
    
    # 同步
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "npu":
        torch.npu.synchronize()  # 假设 NPU 有类似 API
    
    # 测量
    start = time.time()
    for _ in range(num_runs):
        _ = norm(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "npu":
        torch.npu.synchronize()
    
    elapsed = time.time() - start
    throughput = (num_runs * 8 * seq_len) / elapsed  # tokens/s
    
    return throughput


if __name__ == "__main__":
    results = {}
    
    if torch.cuda.is_available():
        results["CUDA"] = benchmark_device("cuda")
    
    if is_npu_available():
        results["NPU"] = benchmark_device("npu")
    
    # 打印结果
    print("\n=== Performance Comparison ===")
    for device, throughput in results.items():
        print(f"{device}: {throughput:.2f} tokens/s")
    
    if len(results) == 2:
        speedup = results["NPU"] / results["CUDA"]
        print(f"\nNPU vs CUDA speedup: {speedup:.2f}x")
```

---

## 配置和环境变量

### 支持环境变量控制

```python
# src/liger_kernel/utils.py

import os

def get_preferred_device():
    """
    支持通过环境变量强制指定设备
    
    用法：
        export LIGER_FORCE_DEVICE=npu
        export LIGER_FORCE_DEVICE=cuda
    """
    forced = os.environ.get("LIGER_FORCE_DEVICE", "").lower()
    if forced in ["npu", "cuda", "amd", "cpu"]:
        return forced
    return infer_device()


def is_npu_optimized_enabled():
    """
    支持禁用 NPU 优化（用于调试）
    
    用法：
        export LIGER_DISABLE_NPU_OPT=1
    """
    return os.environ.get("LIGER_DISABLE_NPU_OPT", "0") != "1"
```

### 客户使用

```bash
# 强制使用 NPU 优化
export LIGER_FORCE_DEVICE=npu
python train.py

# 禁用 NPU 优化（使用通用版本）
export LIGER_DISABLE_NPU_OPT=1
python train.py

# 调试模式：打印设备选择信息
export LIGER_DEBUG=1
python train.py
```

---

## 最佳实践总结

### ✅ 推荐做法

1. **设备检测自动化**
   - 在模块初始化时自动检测设备
   - 用户无需关心底层实现

2. **保持接口一致**
   - 所有设备使用相同的 API
   - 客户代码零修改

3. **性能优先级**
   ```python
   # 优先级：NPU 特化 > CUDA 优化 > 通用实现
   if is_npu_available():
       use_npu_kernel()
   elif is_cuda_available():
       use_cuda_kernel()
   else:
       use_generic_kernel()
   ```

4. **充分测试**
   - 正确性测试：确保数值一致
   - 性能测试：验证加速比
   - 兼容性测试：跨设备测试

5. **提供降级方案**
   - 如果 NPU 特化失败，回退到通用实现
   - 打印警告信息

### ❌ 避免的做法

1. **硬编码设备类型**
   ```python
   # ❌ 不好
   if device == "npu":
       ...
   
   # ✅ 好
   if is_npu_available():
       ...
   ```

2. **修改客户 API**
   ```python
   # ❌ 不好 - 要求客户修改代码
   apply_liger_kernel_to_llama_npu()
   
   # ✅ 好 - 自动检测
   apply_liger_kernel_to_llama()
   ```

3. **缺少回退机制**
   ```python
   # ❌ 不好 - NPU 失败就崩溃
   return npu_kernel(x)
   
   # ✅ 好 - 有回退
   try:
       return npu_kernel(x)
   except Exception as e:
       logger.warning(f"NPU kernel failed: {e}, using fallback")
       return generic_kernel(x)
   ```

---

## 目录结构建议

```
src/liger_kernel/
├── ops/
│   ├── rms_norm.py              # CUDA 实现（默认）
│   ├── rms_norm_npu.py          # NPU 特化实现
│   ├── cross_entropy.py         # CUDA 实现
│   ├── cross_entropy_npu.py     # NPU 特化实现
│   └── ...
├── transformers/
│   ├── rms_norm.py              # 设备感知的 Module
│   ├── cross_entropy.py         # 设备感知的 Module
│   └── monkey_patch.py          # 统一的 patch 接口
├── utils.py                     # 设备检测工具
└── config.py                    # 配置管理
```

---

## 常见问题

### Q: NPU 的 Triton 和 CUDA 的 Triton 有什么区别？

**A**: 主要区别可能在于：

1. **内存层次**：NPU 可能有不同的缓存结构
2. **块大小**：最优的 BLOCK_SIZE 可能不同
3. **指令集**：某些 Triton 操作在 NPU 上可能有特殊优化
4. **数据类型**：NPU 可能对某些精度有特殊支持

**解决方案**：通过 NPU 特化内核调整这些参数。

### Q: 如何确保 NPU 版本和 CUDA 版本数值一致？

**A**: 

1. **严格测试**：编写正确性测试对比输出
2. **允许误差**：考虑不同硬件的浮点误差
3. **使用相同算法**：确保数学逻辑一致

```python
# 正确性测试
def test_cross_device():
    cuda_output = model_cuda(input)
    npu_output = model_npu(input)
    
    # 允许硬件差异导致的小误差
    assert torch.allclose(cuda_output, npu_output, atol=1e-2, rtol=1e-2)
```

### Q: 性能不如预期怎么办？

**A**: 

1. **Profiling**：使用 NPU 的性能分析工具
2. **调整参数**：BLOCK_SIZE、num_warps 等
3. **内存优化**：调整内存访问模式
4. **对比基准**：与 NPU 原生实现对比

```python
# 性能分析
with torch.profiler.profile() as prof:
    output = model(input)

print(prof.key_averages().table())
```

---

## 完整示例：端到端流程

```python
# 1. 设备检测（自动）
from liger_kernel.utils import infer_device
device = infer_device()  # 返回 "npu" 或 "cuda"

# 2. 应用优化（客户代码不变）
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()  # 自动选择 NPU/CUDA 实现

# 3. 加载模型（客户代码不变）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
)

# 4. 训练（客户代码不变）
trainer = Trainer(model=model, ...)
trainer.train()

# 结果：
# - NPU 上自动使用 NPU 优化内核
# - CUDA 上自动使用 CUDA 优化内核
# - 客户代码完全一致！
```

---

## 总结

### 核心原则

1. **透明性**：设备选择对用户透明
2. **一致性**：API 接口保持完全一致
3. **性能**：自动选择最优实现
4. **兼容性**：支持多种设备
5. **可测试**：充分的测试保证正确性

### 关键优势

✅ **客户代码零修改**  
✅ **自动设备检测**  
✅ **性能自动优化**  
✅ **易于维护和扩展**  
✅ **完全向后兼容**

### 实现要点

1. 设备检测自动化（`infer_device()`）
2. 设备感知的 Module（内部选择实现）
3. 统一的 Monkey Patch 接口
4. 充分的测试和性能分析

---

**通过这种方式，您可以为 NPU 提供特化优化，同时保持客户使用方式完全不变！** 🎉

