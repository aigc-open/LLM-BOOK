# 第七章：PyTorch Dispatcher 分发机制

## 本章目标

- 理解 PyTorch Dispatcher 的架构设计
- 掌握多后端支持的原理
- 学习 DispatchKey 机制
- 了解算子注册和自定义算子实现

## 1. 什么是 Dispatcher？

### 1.1 简单理解

**问题：同一个操作，不同的硬件**

```python
import torch

x = torch.randn(10)

# 同样是加法操作：
y1 = x + 1 # CPU 上执行
y2 = x.cuda() + 1 # GPU 上执行
y3 = x.to('mps') + 1 # Apple M1/M2 GPU 上执行

# 问题：PyTorch 如何知道调用哪个实现？
```

**Dispatcher = 中央调度系统**

```
用户调用: x + 1


Dispatcher (调度中心)


根据以下信息决定调用哪个实现：
- Tensor 在哪个设备？(CPU/CUDA/MPS)
- 是否需要梯度？(requires_grad)
- 是否在追踪？(torch.jit)
- 其他特殊模式？(vmap, grad)


调用对应的 kernel 实现
```

### 1.2 为什么需要 Dispatcher？

**挑战 1：多种设备支持**

```python
# PyTorch 支持的后端：
- CPU
- CUDA (NVIDIA GPU)
- ROCm (AMD GPU)
- MPS (Apple Silicon)
- XLA (TPU)
- Vulkan (移动端)
- ... 还在增加
```

**挑战 2：多种执行模式**

```python
# 同一个操作，不同模式下行为不同：

# 正常模式
y = x + 1 # 执行加法

# Autograd 模式
y = x + 1 # 执行加法 + 记录反向传播信息

# TorchScript 模式
y = x + 1 # 记录到计算图

# Vmap 模式
y = vmap(lambda x: x + 1)(x) # 批量映射

# FakeTensor 模式（用于元信息推导）
y = x + 1 # 不执行实际计算，只推导形状和类型
```

**Dispatcher 统一管理这一切！**

## 2. Dispatcher 的架构

### 2.1 整体架构图

```
+---------------------------------------------+
| Python API |
| torch.add(x, y) |
|---------------+-----------------------------+


+---------------------------------------------+
| ATen (PyTorch C++ API) |
| at::add(x, y) |
|---------------+-----------------------------+


+---------------------------------------------+
| Dispatcher (调度中心) | <- 本章重点
| - 分析 DispatchKey |
| - 选择正确的 kernel |
|---------------+-----------------------------+


+-------+-------+


+--------------+ +--------------+
| CPU Kernel | | CUDA Kernel | ...
| add_cpu() | | add_cuda() |
|--------------+ |--------------+
```

### 2.2 DispatchKey（调度键）

**DispatchKey = 决定调度的关键信息**

```cpp
// 简化的 DispatchKey 枚举
enum class DispatchKey {
CPU, // CPU 后端
CUDA, // CUDA 后端
MPS, // Apple Silicon

AutogradCPU, // CPU + Autograd
AutogradCUDA, // CUDA + Autograd

BackendSelect, // 选择后端
ADInplaceOrView, // Autograd 的 inplace 处理

Autocast, // 自动混合精度
Functionalize, // 函数化（移除 inplace）

Python, // Python 调度（__torch_dispatch__）

// ... 还有很多
};
```

### 2.3 DispatchKey 的优先级

Dispatcher 按照特定的优先级顺序检查 DispatchKey：

```
优先级从高到低：

1. Python (Python 自定义调度)
2. Functionalize (函数化转换)
3. ADInplaceOrView (Autograd inplace 处理)
4. AutogradCPU/CUDA (自动微分)
5. Autocast (自动混合精度)
6. BackendSelect (选择后端)
7. CPU/CUDA/... (实际计算 kernel)
```

**示例流程：**

```python
x = torch.randn(10, device='cuda', requires_grad=True)
y = x + 1

# Dispatcher 检查顺序：
# 1. Python dispatch? -> 否
# 2. Functionalize? -> 否
# 3. ADInplaceOrView? -> 否
# 4. AutogradCUDA? -> 是！执行 Autograd 逻辑
#

# 5. 记录反向传播信息
# 6. 继续调度到底层 kernel
#

# 7. BackendSelect? -> 选择 CUDA
# 8. CUDA kernel? -> 是！执行 add_cuda_kernel()
```

## 3. 算子注册

### 3.1 如何注册算子

**C++ 侧注册：**

```cpp
// 在 pytorch/aten/src/ATen/native/BinaryOps.cpp

// 1. 定义 CPU 实现
Tensor add_cpu(const Tensor& self, const Tensor& other, const Scalar& alpha) {
// CPU 实现代码
return result;
}

// 2. 定义 CUDA 实现
Tensor add_cuda(const Tensor& self, const Tensor& other, const Scalar& alpha) {
// CUDA 实现代码
return result;
}

// 3. 注册算子
TORCH_LIBRARY_IMPL(aten, CPU, m) {
m.impl("add.Tensor", &add_cpu);
}

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
m.impl("add.Tensor", &add_cuda);
}
```

**Python 侧查看：**

```python
import torch

# 查看算子的注册信息
print(torch._C._dispatch_dump("aten::add"))

# 输出类似：
# aten::add.Tensor
# CPU: registered
# CUDA: registered
# AutogradCPU: registered
# AutogradCUDA: registered
# ...
```

### 3.2 算子定义

**在 native_functions.yaml 中定义：**

```yaml
# pytorch/aten/src/ATen/native/native_functions.yaml

- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
structured_delegate: add.out
variants: function, method
dispatch:
CPU: add_cpu
CUDA: add_cuda
MPS: add_mps
tags: [core]
```

这个 YAML 文件是 PyTorch 算子的"字典"：
- 定义算子的签名
- 指定各个后端的实现
- 自动生成分发代码

## 4. 自定义算子和后端

### 4.1 Python 层自定义算子

```python
import torch
from torch import Tensor

# 使用 torch.library 注册自定义算子
torch.library.define(
"mylib::custom_add",
"(Tensor x, Tensor y) -> Tensor"
)

# CPU 实现
@torch.library.impl("mylib::custom_add", "CPU")
def custom_add_cpu(x: Tensor, y: Tensor) -> Tensor:
print("使用 CPU 实现")
return x + y

# CUDA 实现
@torch.library.impl("mylib::custom_add", "CUDA")
def custom_add_cuda(x: Tensor, y: Tensor) -> Tensor:
print("使用 CUDA 实现")
return x + y

# 使用自定义算子
x_cpu = torch.randn(10)
y_cpu = torch.randn(10)
result_cpu = torch.ops.mylib.custom_add(x_cpu, y_cpu) # "使用 CPU 实现"

x_cuda = torch.randn(10, device='cuda')
y_cuda = torch.randn(10, device='cuda')
result_cuda = torch.ops.mylib.custom_add(x_cuda, y_cuda) # "使用 CUDA 实现"
```

### 4.2 使用 __torch_dispatch__

**更高层的 Python 拦截：**

```python
import torch
from torch.utils._python_dispatch import TorchDispatchMode

class LoggingMode(TorchDispatchMode):
"""记录所有 PyTorch 操作"""

def __torch_dispatch__(self, func, types, args=(), kwargs=None):
print(f"调用: {func.__name__}")
print(f" 输入形状: {[a.shape if isinstance(a, torch.Tensor) else a for a in args]}")

# 调用原始实现
result = func(*args, **(kwargs or {}))

print(f" 输出形状: {result.shape if isinstance(result, torch.Tensor) else result}")
return result

# 使用
with LoggingMode():
x = torch.randn(3, 4)
y = x + 1
z = y * 2

# 输出：
# 调用: add
# 输入形状: [(3, 4), 1]
# 输出形状: (3, 4)
# 调用: mul
# 输入形状: [(3, 4), 2]
# 输出形状: (3, 4)
```

### 4.3 自定义 Tensor 子类

```python
import torch

class MyTensor(torch.Tensor):
"""自定义 Tensor 类"""

@staticmethod
def __new__(cls, data):
return torch.Tensor._make_subclass(cls, data)

def __repr__(self):
return f"MyTensor({super().__repr__()})"

@classmethod
def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
"""拦截所有操作"""

def unwrap(x):
return x.data if isinstance(x, MyTensor) else x

def wrap(x):
return MyTensor(x) if isinstance(x, torch.Tensor) else x

# 解包
args = tuple(unwrap(a) for a in args)
kwargs = {k: unwrap(v) for k, v in (kwargs or {}).items()}

# 执行原始操作
result = func(*args, **kwargs)

# 包装回 MyTensor
return wrap(result)

# 使用
x = MyTensor(torch.randn(3, 4))
y = x + 1 # 自动拦截并包装结果
print(type(y)) # <class '__main__.MyTensor'>
```

## 5. 重要的 DispatchKey

### 5.1 Autograd Keys

```python
# AutogradCPU, AutogradCUDA 等
# 作用：自动记录梯度信息

x = torch.randn(10, requires_grad=True)
y = x * 2 # 触发 AutogradCPU dispatch

# Dispatcher 流程：
# 1. 检测到 requires_grad=True
# 2. 使用 AutogradCPU key
# 3. 记录反向传播信息
# 4. 继续调度到 CPU kernel 执行实际计算
```

### 5.2 BackendSelect Key

```python
# 作用：根据输入 tensor 的设备选择后端

def example(x, y):
return x + y

# x 和 y 都在 CPU -> 选择 CPU backend
result1 = example(torch.randn(10), torch.randn(10))

# x 和 y 都在 CUDA -> 选择 CUDA backend
result2 = example(torch.randn(10, device='cuda'), torch.randn(10, device='cuda'))

# x 和 y 在不同设备 -> 错误！
# result3 = example(torch.randn(10), torch.randn(10, device='cuda')) # 报错
```

### 5.3 Python Key

```python
# 作用：允许 Python 代码拦截调度

class CustomTensor(torch.Tensor):
@classmethod
def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
print(f"Python dispatch: {func}")
return func(*args, **(kwargs or {}))

# 使用 Python Key 拦截
```

### 5.4 FakeTensor Key

```python
# 作用：不执行实际计算，只推导元信息（形状、dtype 等）

from torch._subclasses.fake_tensor import FakeTensorMode

with FakeTensorMode():
x = torch.randn(100, 100) # 不分配实际内存
y = x @ x # 不执行实际计算
print(y.shape) # 但知道输出形状：(100, 100)

# 用于编译时的形状推导，不需要实际数据
```

## 6. Dispatcher 的高级特性

### 6.1 算子组合（Decomposition）

```python
# 复杂算子可以分解为简单算子

# 例如：gelu 可以分解为
def gelu_decomposition(x):
return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# 在 Dispatcher 中注册分解规则
# 某些后端如果没有原生 gelu 实现，会自动使用分解版本
```

### 6.2 Functionalization

```python
# 作用：将 inplace 操作转换为 out-of-place

x = torch.randn(10)

# Inplace 操作
x.add_(1) # 修改 x

# Functionalization 后：
# x_new = x.add(1) # 创建新 tensor
# 然后更新引用

# 这对编译很有用（避免别名问题）
```

### 6.3 Structured Kernels

```python
# 结构化 kernel：统一处理 functional 和 out 变体

# 用户可能调用：
y = torch.add(x, 1) # functional 形式
torch.add(x, 1, out=y) # out 形式

# Dispatcher 内部统一调度到同一个"结构化 kernel"
# 减少代码重复
```

## 7. 源码分析

### 7.1 核心文件位置

```
pytorch/
|-- aten/src/ATen/
| |-- core/
| | |-- dispatch/
| | | |-- Dispatcher.h # Dispatcher 核心
| | | |-- DispatchKey.h # DispatchKey 定义
| | | |-- OperatorEntry.h # 算子注册表
| |-- native/
| | |-- native_functions.yaml # 算子定义
| | |-- BinaryOps.cpp # 二元操作实现
| | |-- ...
|-- torch/
| |-- _ops.py # Python 算子接口
| |-- library.py # Python 算子注册
|-- c10/core/DispatchKey.h # DispatchKey 底层定义
```

### 7.2 Dispatcher 的核心数据结构

```cpp
// aten/src/ATen/core/dispatch/Dispatcher.h (简化版)

class Dispatcher {
private:
// 算子注册表：算子名 -> OperatorEntry
std::unordered_map<OperatorName, OperatorEntry> operators_;

public:
// 调度函数
template<typename Return, typename... Args>
Return call(const OperatorHandle& op, Args... args) {
// 1. 计算 DispatchKey
DispatchKey key = computeDispatchKey(args...);

// 2. 查找对应的 kernel
KernelFunction* kernel = op.lookup(key);

// 3. 调用 kernel
return kernel->call(args...);
}
};
```

### 7.3 DispatchKey 计算

```cpp
// 简化的伪代码

DispatchKey computeDispatchKey(const Tensor& tensor) {
DispatchKeySet keys;

// 1. 添加后端 key
keys = keys | tensor.device().dispatchKey(); // CPU/CUDA/...

// 2. 添加 Autograd key（如果需要梯度）
if (tensor.requires_grad()) {
keys = keys | getAutogradKey(tensor.device());
}

// 3. 添加其他 key（functionalize, python 等）
keys = keys | getCurrentDispatchKeys();

// 4. 返回最高优先级的 key
return keys.highestPriorityKey();
}
```

## 8. 适配自定义硬件

### 8.1 添加新的 DispatchKey

```cpp
// 1. 在 c10/core/DispatchKey.h 中添加新的 key

enum class DispatchKey {
// ... 现有的 keys ...

MyCustomChip, // 自定义芯片 <- 新增

// ...
};
```

### 8.2 注册算子实现

```cpp
// 2. 为自定义芯片实现算子

// mylib/kernels/Add.cpp

Tensor add_custom_chip(const Tensor& self, const Tensor& other, const Scalar& alpha) {
// 调用自定义芯片的 API
MyChip::add(self.data(), other.data(), alpha.toDouble());
return result;
}

// 3. 注册
TORCH_LIBRARY_IMPL(aten, MyCustomChip, m) {
m.impl("add.Tensor", &add_custom_chip);
m.impl("mul.Tensor", &mul_custom_chip);
// ... 更多算子
}
```

### 8.3 设备管理

```python
# 4. Python 侧暴露设备

import torch

# 注册设备类型
torch.utils.cpp_extension.load(
name="my_custom_chip",
sources=["my_chip.cpp"],
)

# 使用
x = torch.randn(10, device='my_custom_chip')
y = x + 1 # 自动调度到 add_custom_chip()
```

## 9. 实际应用场景

### 9.1 性能分析工具

```python
class ProfilingMode(torch.utils._python_dispatch.TorchDispatchMode):
def __init__(self):
self.op_counts = {}
self.op_times = {}

def __torch_dispatch__(self, func, types, args=(), kwargs=None):
import time

# 记录调用次数
op_name = func.__name__
self.op_counts[op_name] = self.op_counts.get(op_name, 0) + 1

# 记录执行时间
start = time.time()
result = func(*args, **(kwargs or {}))
torch.cuda.synchronize() if torch.cuda.is_available() else None
elapsed = time.time() - start

self.op_times[op_name] = self.op_times.get(op_name, 0) + elapsed

return result

def report(self):
print("\n操作统计:")
for op, count in sorted(self.op_counts.items(), key=lambda x: -x[1]):
time_ms = self.op_times[op] * 1000
print(f" {op:30} 调用: {count:5} 次, 总时间: {time_ms:.2f}ms")

# 使用
profiler = ProfilingMode()
with profiler:
model(input_data)

profiler.report()
```

### 9.2 自动混合精度

```python
# PyTorch 的 autocast 就是通过 Dispatcher 实现的

with torch.cuda.amp.autocast():
# 某些操作自动转为 float16
output = model(input)

# 内部流程：
# 1. 进入 autocast 上下文
# 2. 添加 Autocast DispatchKey
# 3. Dispatcher 检测到 Autocast key
# 4. 根据规则自动转换精度
```

## 10. 小结

### 核心要点

1. **Dispatcher = PyTorch 的中央调度系统**
2. **DispatchKey**：决定调用哪个 kernel 的关键
3. **多后端支持**：统一的接口，不同的实现
4. **优先级机制**：按顺序检查多个 DispatchKey
5. **可扩展性**：易于添加新的后端和算子

### 关键概念

- **DispatchKey**：调度键（CPU/CUDA/Autograd/...）
- **Kernel**：算子的具体实现
- **算子注册**：将 kernel 与 DispatchKey 关联
- **__torch_dispatch__**：Python 层的拦截机制
- **BackendSelect**：自动选择后端

### Dispatcher 的作用

```
统一接口 + 多样化后端 + 扩展性


支持多种设备（CPU/GPU/TPU/...）
支持多种模式（Autograd/JIT/Vmap/...）
支持自定义扩展
```

### 在 torch.compile 中的角色

```
torch.compile 最终生成的代码也是通过 Dispatcher 调度：
Dynamo -> AOTAutograd -> Inductor 生成代码


调用 PyTorch 算子


Dispatcher


实际的 kernel 执行
```

### 下一步

现在你理解了 PyTorch 的底层分发机制，最后我们来看如何将这些知识应用到实际场景：

[第八章：国产芯片适配实战指南](./08_custom_chip_adaptation.md) - 如何为自定义硬件适配 torch.compile

---

## 练习题

1. 使用 `torch.library` 注册一个自定义算子
2. 编写一个 `TorchDispatchMode` 来记录所有算子调用
3. 查看一个算子（如 `add`）的所有注册信息

**继续学习** -> [国产芯片适配实战指南](./08_custom_chip_adaptation.md)

