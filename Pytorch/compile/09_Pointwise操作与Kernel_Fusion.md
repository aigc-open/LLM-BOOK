# 第九章：Pointwise 操作与 Kernel Fusion

## 📖 本章概要

本章深入讲解 TorchInductor 中最核心的优化技术之一：**Kernel Fusion（算子融合）**。通过理解 Pointwise 操作的本质，您将明白：
- 为什么 `torch.compile` 能带来显著的性能提升
- TorchInductor 如何决定生成 Triton Kernel 还是调用外部库
- 如何设计模型以最大化编译优化效果

## 目录

1. [Pointwise 操作详解](#1-pointwise-操作详解)
2. [TorchInductor 的 Kernel 策略](#2-torchinductor-的-kernel-策略)
3. [Kernel Fusion 原理](#3-kernel-fusion-原理)
4. [实战：观察 Kernel Fusion 效果](#4-实战观察-kernel-fusion-效果)
5. [优化建议](#5-优化建议)
6. [常见问题](#6-常见问题)

---

## 1. Pointwise 操作详解

### 1.1 什么是 Pointwise 操作

**Pointwise（逐点）操作** 是指对张量中的每个元素**独立**进行计算的操作。关键特征是：**每个输出元素只依赖对应位置的输入元素**。

```python
# Pointwise 操作示例
y[i] = f(x[i])           # 一元操作
z[i] = f(x[i], y[i])     # 二元操作

# 具体例子
y = torch.relu(x)        # y[i] = max(0, x[i])
y = x + 1                # y[i] = x[i] + 1
y = x * scale            # y[i] = x[i] * scale[i]
y = torch.sigmoid(x)     # y[i] = 1 / (1 + exp(-x[i]))
```

### 1.2 操作类型分类

| 操作类型 | 定义 | 示例 | 数据依赖 |
|---------|------|------|----------|
| **Pointwise** | 逐元素独立计算 | `relu`, `sigmoid`, `+`, `*` | 无 |
| **Reduction** | 多元素归约为一个 | `sum`, `mean`, `max` | 同维度所有元素 |
| **Matmul** | 矩阵乘法 | `@`, `mm`, `Linear` | 整行/整列 |
| **Scatter/Gather** | 索引操作 | `index_select`, `scatter_` | 索引位置 |

### 1.3 数据依赖图示

```
Pointwise 操作（无数据依赖）：
┌─────────────────────────────────────┐
│  x: [a, b, c, d, e]                 │
│      ↓  ↓  ↓  ↓  ↓   (独立计算)     │
│  y: [f(a), f(b), f(c), f(d), f(e)]  │
└─────────────────────────────────────┘

Reduction 操作（跨元素依赖）：
┌─────────────────────────────────────┐
│  x: [a, b, c, d, e]                 │
│      ↘  ↓  ↓  ↓  ↙   (全部参与)     │
│  y:      sum(x)                     │
└─────────────────────────────────────┘

Matmul 操作（行列依赖）：
┌─────────────────────────────────────┐
│  A[i,:] @ B[:,j] = C[i,j]           │
│  需要访问 A 的第 i 行和 B 的第 j 列   │
└─────────────────────────────────────┘
```

### 1.4 常见 Pointwise 操作列表

```python
# 一元 Pointwise 操作
torch.relu(x)
torch.sigmoid(x)
torch.tanh(x)
torch.gelu(x)
torch.exp(x)
torch.log(x)
torch.sqrt(x)
torch.abs(x)
torch.neg(x)
x.pow(2)
x.clamp(min=0, max=1)

# 二元 Pointwise 操作
x + y
x - y
x * y
x / y
torch.maximum(x, y)
torch.minimum(x, y)
torch.where(cond, x, y)

# 带广播的 Pointwise 操作
x + scalar
x * scale  # scale 可以是标量或可广播的张量
x + bias   # bias 可以是可广播的张量
```

---

## 2. TorchInductor 的 Kernel 策略

### 2.1 策略概览

TorchInductor 会根据操作类型选择不同的实现方式：

```
                     TorchInductor
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
      Pointwise      Reduction       Matmul
           │              │              │
           ▼              ▼              ▼
     Triton Kernel   Triton Kernel   extern_kernels
     (自动生成)      (自动生成)      (调用库函数)
```

### 2.2 为什么 Matmul 使用外部库？

**原因 1：cuBLAS/cuDNN 已极度优化**

```python
# cuBLAS GEMM 的优化历程
# - 1998: 第一版 BLAS
# - 2007: cuBLAS 发布
# - 至今: 20+ 年的持续优化
# - 包含: 手写汇编、硬件特定优化、多种 tiling 策略
```

**原因 2：Matmul 的复杂性**

```python
# Matmul 的高效实现需要考虑：
# 1. Tiling（分块）策略
# 2. 共享内存使用
# 3. 寄存器分配
# 4. 内存访问合并
# 5. 流水线隐藏延迟
# 6. Tensor Core 利用（如果支持）

# 一个简单的 Triton matmul 实现（性能通常不如 cuBLAS）
@triton.jit
def matmul_kernel(A, B, C, M, N, K, ...):
    # 需要复杂的 tiling 和内存管理
    # 即使精心优化，也很难超越 cuBLAS
    pass
```

**原因 3：成本效益**

| 方案 | 开发成本 | 性能 |
|------|---------|------|
| 调用 cuBLAS | 低（已有） | 最优 |
| Triton 生成 | 高（需优化） | 通常较差 |

### 2.3 为什么 Pointwise 使用 Triton？

**原因 1：没有现成的融合库**

```python
# 没有 cuBLAS 函数可以做这个：
y = relu(x * scale + bias)

# 只能拆成多个调用：
tmp1 = x * scale     # 一次 kernel
tmp2 = tmp1 + bias   # 又一次 kernel
y = relu(tmp2)       # 再一次 kernel
```

**原因 2：Pointwise 操作易于并行**

```python
# Pointwise 操作天然并行，每个线程独立处理
@triton.jit
def pointwise_kernel(x, scale, bias, out, N):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    
    val = tl.load(x + idx, mask=mask)
    s = tl.load(scale + idx, mask=mask)
    b = tl.load(bias + idx, mask=mask)
    
    # 简单直接，无需复杂的同步或通信
    result = tl.maximum(val * s + b, 0)  # fused relu(x*s+b)
    
    tl.store(out + idx, result, mask=mask)
```

**原因 3：Fusion 带来的收益巨大**

```python
# 未融合（5 次内存往返）
tmp1 = x * scale     # 读 x, scale -> 写 tmp1
tmp2 = tmp1 + bias   # 读 tmp1, bias -> 写 tmp2
tmp3 = relu(tmp2)    # 读 tmp2 -> 写 tmp3
tmp4 = tmp3 * 2.0    # 读 tmp3 -> 写 tmp4
y = sigmoid(tmp4)    # 读 tmp4 -> 写 y

# 融合后（1 次内存往返）
y = fused_kernel(x, scale, bias)  # 读 x, scale, bias -> 写 y
# 中间结果全部在寄存器中，不写回显存！
```

### 2.4 Kernel 类型决策树

```
                    操作分析
                       │
           ┌───────────┴───────────┐
           │                       │
      是否为 Matmul?          是否为 Pointwise?
           │                       │
      ┌────┴────┐            ┌────┴────┐
      Yes       No           Yes       No
      │         │            │         │
  extern     检查其他     生成 Triton  检查其他
  (cuBLAS)   操作类型     Kernel       操作类型
```

---

## 3. Kernel Fusion 原理

### 3.1 什么是 Kernel Fusion

**Kernel Fusion（算子融合）** 是将多个连续的操作合并成一个 GPU Kernel 执行的优化技术。

```python
# 概念图示
┌─────────────────────────────────────────────────────┐
│                    未融合                            │
│  Kernel1    Kernel2    Kernel3    Kernel4           │
│  x → tmp1 → tmp1 → tmp2 → tmp2 → tmp3 → tmp3 → y   │
│  [全局内存写入/读取]  [全局内存写入/读取]  ...        │
└─────────────────────────────────────────────────────┘

                         ↓ Fusion

┌─────────────────────────────────────────────────────┐
│                    融合后                            │
│              Fused Kernel                           │
│  x ────────────────────────────────────────────→ y  │
│        [中间结果保持在寄存器/共享内存中]              │
└─────────────────────────────────────────────────────┘
```

### 3.2 Fusion 的性能收益

#### 3.2.1 减少内存带宽消耗

```python
# 假设张量大小为 N，每个元素 4 字节

# 未融合：5 个操作
# 内存读写量 = 5 * 2 * N * 4 = 40N 字节（每个操作读+写）

# 融合后：1 个操作
# 内存读写量 = 2 * N * 4 = 8N 字节（只读一次，写一次）

# 带宽节省 = (40N - 8N) / 40N = 80%
```

#### 3.2.2 减少 Kernel Launch 开销

```python
# Kernel Launch 开销（典型值）
# - CPU 端调度：~5-10 μs
# - GPU 端启动：~2-5 μs
# - 总计：~7-15 μs / 次

# 5 个独立 Kernel：5 * 10 = 50 μs 开销
# 1 个融合 Kernel：1 * 10 = 10 μs 开销

# 对于小张量（执行时间 < 50μs），launch 开销可能占主导！
```

#### 3.2.3 提高缓存利用率

```python
# 融合后，数据保持在：
# 1. 寄存器（最快，~0 延迟）
# 2. L1 Cache（~20 cycles）
# 3. L2 Cache（~200 cycles）

# 而非：
# 全局显存（~400-800 cycles）
```

### 3.3 TorchInductor 的 Fusion 策略

#### 3.3.1 Pointwise Fusion

```python
# 连续的 Pointwise 操作会被自动融合
x = input
x = x * scale      # ┐
x = x + bias       # │ 这些会被融合成一个 Kernel
x = torch.relu(x)  # │
x = x * 2.0        # │
x = torch.tanh(x)  # ┘
```

#### 3.3.2 Reduction + Pointwise Fusion

```python
# LayerNorm 的实现会融合：
# 1. mean 计算（reduction）
# 2. variance 计算（reduction）
# 3. 归一化（pointwise）
# 4. scale + bias（pointwise）

x = F.layer_norm(x, normalized_shape)
# 内部生成 1-2 个融合的 Triton Kernel
```

#### 3.3.3 Fusion 边界

某些操作会阻止 fusion：

```python
# 不能融合的情况
x = pointwise_op1(x)
y = matmul(x, weight)    # ← 打断 fusion（调用 extern kernel）
y = pointwise_op2(y)

# 结果：
# Kernel 1: pointwise_op1
# Kernel 2: matmul (extern)
# Kernel 3: pointwise_op2
```

### 3.4 Fusion 可视化

```
原始计算图：
┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
│ * │ → │ + │ → │relu│ → │ * │ → │sig│
└───┘   └───┘   └───┘   └───┘   └───┘
  ↓       ↓       ↓       ↓       ↓
[mem]   [mem]   [mem]   [mem]   [mem]

融合后：
┌─────────────────────────────────────┐
│  triton_poi_fused_mul_add_relu_...  │
│  * → + → relu → * → sigmoid         │
│  (全部在寄存器中完成)                │
└─────────────────────────────────────┘
                 ↓
              [mem]
```

---

## 4. 实战：观察 Kernel Fusion 效果

### 4.1 创建测试模型

```python
import torch
import torch.nn as nn

class PointwiseModel(nn.Module):
    """纯 Pointwise 操作模型 - 展示 Kernel Fusion"""
    
    def __init__(self, dim=1024):
        super().__init__()
        self.scale1 = nn.Parameter(torch.randn(dim))
        self.bias1 = nn.Parameter(torch.randn(dim))
        self.scale2 = nn.Parameter(torch.randn(dim))
        self.bias2 = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        # 这些操作会被融合成 1 个 Triton Kernel
        x = x * self.scale1 + self.bias1
        x = torch.relu(x)
        x = x * self.scale2 + self.bias2
        x = torch.sigmoid(x)
        x = x * 2.0 + 1.0
        x = torch.tanh(x)
        return x
```

### 4.2 编译并观察

```python
# 设置环境变量以查看生成的代码
import os
os.environ['TORCH_LOGS'] = 'output_code'

model = PointwiseModel().cuda()
compiled_model = torch.compile(model, backend="inductor")

# 运行以触发编译
x = torch.randn(128, 1024, device='cuda')
y = compiled_model(x)
```

### 4.3 生成的 Triton Kernel

TorchInductor 会生成类似这样的融合 Kernel：

```python
@triton.jit
def triton_poi_fused_add_mul_relu_sigmoid_tanh_0(
    in_ptr0,   # x
    in_ptr1,   # scale1
    in_ptr2,   # bias1
    in_ptr3,   # scale2
    in_ptr4,   # bias2
    out_ptr0,  # output
    xnumel,
    XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    
    # 加载所有输入
    x = tl.load(in_ptr0 + xindex, xmask)
    scale1 = tl.load(in_ptr1 + (xindex % 1024), xmask)
    bias1 = tl.load(in_ptr2 + (xindex % 1024), xmask)
    scale2 = tl.load(in_ptr3 + (xindex % 1024), xmask)
    bias2 = tl.load(in_ptr4 + (xindex % 1024), xmask)
    
    # 所有计算在寄存器中完成
    tmp0 = x * scale1 + bias1
    tmp1 = tl.maximum(tmp0, 0)  # relu
    tmp2 = tmp1 * scale2 + bias2
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp3 * 2.0 + 1.0
    tmp5 = tl.libdevice.tanh(tmp4)
    
    # 只写一次
    tl.store(out_ptr0 + xindex, tmp5, xmask)
```

### 4.4 性能对比

```python
import time

def benchmark(fn, x, warmup=50, runs=100):
    # Warmup
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(runs):
        fn(x)
    torch.cuda.synchronize()
    
    return (time.time() - start) / runs * 1000  # ms

# 测试
x = torch.randn(1024, 1024, device='cuda')

eager_time = benchmark(model, x)
compiled_time = benchmark(compiled_model, x)

print(f"Eager:    {eager_time:.3f} ms")
print(f"Compiled: {compiled_time:.3f} ms")
print(f"Speedup:  {eager_time / compiled_time:.2f}x")
```

典型结果：

```
Eager:    0.245 ms
Compiled: 0.052 ms
Speedup:  4.71x
```

---

## 5. 优化建议

### 5.1 最大化 Fusion 效果

```python
# ✅ 好的模式：连续的 Pointwise 操作
def good_forward(x, scale, bias):
    x = x * scale + bias
    x = torch.relu(x)
    x = x * 0.5 + 0.5
    return torch.sigmoid(x)

# ❌ 避免打断 Fusion
def bad_forward(x, weight, scale, bias):
    x = x * scale + bias
    x = torch.relu(x)
    x = x @ weight        # ← Matmul 打断了 Fusion
    x = x * 0.5 + 0.5
    return torch.sigmoid(x)
```

### 5.2 选择合适的激活函数

所有激活函数都是 Pointwise，但复杂度不同：

```python
# 计算复杂度（相对）
torch.relu(x)      # 1x  - 最简单
torch.sigmoid(x)   # 3x  - exp 计算
torch.tanh(x)      # 3x  - exp 计算
torch.gelu(x)      # 5x  - 包含 erf
torch.silu(x)      # 4x  - x * sigmoid(x)
```

### 5.3 使用 torch.compile 的高级选项

```python
# 更激进的优化
compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="max-autotune",  # 尝试更多配置
    fullgraph=True,       # 强制整图编译
)

# 减少编译时间（牺牲部分性能）
compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="reduce-overhead",  # 减少 launch 开销
)
```

### 5.4 模型设计建议

```python
class OptimizedBlock(nn.Module):
    """针对 Fusion 优化的模块设计"""
    
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        # Pointwise 操作组 1（会被融合）
        residual = x * self.scale + self.bias
        
        # Matmul（单独的 kernel）
        x = self.linear(x)
        
        # Pointwise 操作组 2（会被融合）
        x = torch.gelu(x)
        x = x + residual  # 残差连接
        
        return x
```

---

## 6. 常见问题

### Q1: 为什么我的模型编译后反而变慢了？

**可能原因**：

1. **模型太小** - 编译开销 > 优化收益
2. **主要是 Matmul** - 没有 Pointwise 可以优化
3. **测量包含了编译时间** - 需要预热

```python
# 正确的测量方式
compiled_model = torch.compile(model)

# 1. 预热（触发编译）
for _ in range(10):
    _ = compiled_model(x)
torch.cuda.synchronize()

# 2. 然后再测量
start = time.time()
...
```

### Q2: 如何判断是否生成了 Triton Kernel？

```python
# 方法 1：设置环境变量
os.environ['TORCH_LOGS'] = 'output_code'

# 方法 2：使用 CUDA profiler
# nsys profile python script.py

# 方法 3：检查缓存目录
# ls /tmp/torchinductor_*/
```

### Q3: Linear 层能否用 Triton 实现？

可以，但**不推荐**：

```python
# 强制使用 Triton matmul（通常更慢）
import torch._inductor.config as config
config.triton.mm = "triton"  # 默认是 "aten"
```

### Q4: 哪些操作会打断 Fusion？

```python
# 会打断 Fusion 的操作：
torch.matmul(x, w)       # Matmul
F.conv2d(x, w)           # 卷积
x.view(-1)               # 某些 reshape
x[idx]                   # 复杂的索引
torch.sort(x)            # 排序
torch.unique(x)          # 去重
```

### Q5: 如何查看 Fusion 后的 Kernel 名称？

Kernel 名称包含了融合的操作：

```
triton_poi_fused_add_mul_relu_sigmoid_tanh_0
       │     │     │   │    │       │     │
       │     │     │   │    │       │     └─ 序号
       │     │     │   │    │       └─ tanh
       │     │     │   │    └─ sigmoid  
       │     │     │   └─ relu
       │     │     └─ mul (乘法)
       │     └─ add (加法)
       └─ poi = pointwise
```

---

## 7. 总结

### 核心要点

1. **Pointwise 操作** = 逐元素独立计算，无数据依赖
2. **TorchInductor** 对 Pointwise 生成 Triton Kernel，对 Matmul 调用 cuBLAS
3. **Kernel Fusion** 是 `torch.compile` 性能提升的关键
4. **Fusion 收益**：减少内存访问、减少 launch 开销、提高缓存利用

### 记忆口诀

```
Pointwise 用 Triton，Matmul 用 cuBLAS；
连续 Pointwise 自动融合，中间结果不写回；
想要加速看 Fusion，模型设计要配合。
```

### 推荐阅读

- [PyTorch 官方文档：torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/)
- [TorchInductor 设计文档](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)

---

## 附录 A：完整示例代码

```python
#!/usr/bin/env python3
"""
Pointwise 操作与 Kernel Fusion 示例
演示 TorchInductor 如何融合 Pointwise 操作
"""

import os
import time
import torch
import torch.nn as nn

# 设置环境变量以查看生成的代码（可选）
# os.environ['TORCH_LOGS'] = 'output_code'


class PointwiseModel(nn.Module):
    """纯 Pointwise 操作模型"""
    
    def __init__(self, dim=1024):
        super().__init__()
        self.scale1 = nn.Parameter(torch.randn(dim))
        self.bias1 = nn.Parameter(torch.randn(dim))
        self.scale2 = nn.Parameter(torch.randn(dim))
        self.bias2 = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        x = x * self.scale1 + self.bias1
        x = torch.relu(x)
        x = x * self.scale2 + self.bias2
        x = torch.sigmoid(x)
        x = x * 2.0 + 1.0
        x = torch.tanh(x)
        return x


class MixedModel(nn.Module):
    """混合模型：Linear + Pointwise"""
    
    def __init__(self, dim=512):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.randn(dim))
        self.bias = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        # Linear (extern kernel - cuBLAS)
        x = self.linear1(x)
        
        # Pointwise (Triton kernel - 会被融合)
        x = x * self.scale + self.bias
        x = torch.relu(x)
        x = x * 0.5 + 0.5
        x = torch.gelu(x)
        
        # Another Linear
        x = self.linear2(x)
        
        # More Pointwise
        x = torch.sigmoid(x)
        
        return x


def benchmark(fn, x, warmup=50, runs=100):
    """性能测试"""
    # Warmup
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(runs):
        fn(x)
    torch.cuda.synchronize()
    
    return (time.time() - start) / runs * 1000


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 测试 Pointwise 模型
    print("\n" + "="*60)
    print("Test 1: Pure Pointwise Model")
    print("="*60)
    
    model1 = PointwiseModel(1024).to(device)
    compiled1 = torch.compile(model1, backend="inductor")
    
    x1 = torch.randn(256, 1024, device=device)
    
    eager_time1 = benchmark(model1, x1)
    compiled_time1 = benchmark(compiled1, x1)
    
    print(f"Eager:    {eager_time1:.3f} ms")
    print(f"Compiled: {compiled_time1:.3f} ms")
    print(f"Speedup:  {eager_time1 / compiled_time1:.2f}x")
    
    # 测试混合模型
    print("\n" + "="*60)
    print("Test 2: Mixed Model (Linear + Pointwise)")
    print("="*60)
    
    model2 = MixedModel(512).to(device)
    compiled2 = torch.compile(model2, backend="inductor")
    
    x2 = torch.randn(256, 512, device=device)
    
    eager_time2 = benchmark(model2, x2)
    compiled_time2 = benchmark(compiled2, x2)
    
    print(f"Eager:    {eager_time2:.3f} ms")
    print(f"Compiled: {compiled_time2:.3f} ms")
    print(f"Speedup:  {eager_time2 / compiled_time2:.2f}x")
    
    # 验证正确性
    print("\n" + "="*60)
    print("Correctness Verification")
    print("="*60)
    
    with torch.no_grad():
        y_eager = model1(x1)
        y_compiled = compiled1(x1)
        max_diff = (y_eager - y_compiled).abs().max().item()
        print(f"Max difference: {max_diff:.2e}")
        print(f"✓ Results match" if max_diff < 1e-5 else "✗ Results differ")


if __name__ == "__main__":
    main()
```

---

## 附录 B：Kernel Fusion 决策流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    TorchInductor 编译流程                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FX Graph 分析                          │
│  识别操作类型：Pointwise / Reduction / Matmul / 其他        │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌───────────────────┐ ┌───────────────┐ ┌───────────────────┐
│    Pointwise      │ │   Reduction   │ │     Matmul        │
│  (element-wise)   │ │  (sum, mean)  │ │  (mm, linear)     │
└───────────────────┘ └───────────────┘ └───────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌───────────────────┐ ┌───────────────┐ ┌───────────────────┐
│  尝试与相邻的     │ │  检查是否可与 │ │  调用 extern      │
│  Pointwise 融合   │ │  Pointwise    │ │  kernel (cuBLAS)  │
│                   │ │  融合         │ │                   │
└───────────────────┘ └───────────────┘ └───────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌───────────────────┐ ┌───────────────┐ ┌───────────────────┐
│  生成融合的       │ │  生成融合的   │ │  生成 extern      │
│  Triton Kernel    │ │  Triton       │ │  调用代码         │
│  triton_poi_...   │ │  Kernel       │ │  aten.mm.default  │
└───────────────────┘ └───────────────┘ └───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    最终生成的代码                            │
│  = 若干 Triton Kernel + 若干 extern kernel 调用             │
└─────────────────────────────────────────────────────────────┘
```

