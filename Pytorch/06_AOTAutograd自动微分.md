# 第六章：AOTAutograd 自动微分

## 🎯 本章目标

- 理解 AOTAutograd（Ahead-of-Time Autograd）的原理
- 掌握前向图和反向图的生成机制
- 了解与传统 Autograd 的区别
- 学习联合编译优化的技术

## 1. 什么是 AOTAutograd？

### 1.1 简单理解

**传统 Autograd（运行时）：**

```python
# 前向传播
x = torch.randn(10, requires_grad=True)
y = x * 2
z = y + 1
loss = z.sum()

# 反向传播（运行时动态构建）
loss.backward()  # ← 在这里才构建反向图
```

**AOTAutograd（提前编译）：**

```python
# torch.compile 使用 AOTAutograd
compiled_fn = torch.compile(model)

# 第一次调用时：
# 1. 捕获前向计算图
# 2. 提前生成反向计算图（即使还没 backward！）
# 3. 联合优化前向+反向
# 4. 编译为高效代码
output = compiled_fn(input_data)

# backward 时直接使用编译好的反向图
loss.backward()  # 很快！
```

**类比：**
- **传统 Autograd**：边走边建路（运行时构建）
- **AOTAutograd**：提前修好高速公路（编译时构建）

### 1.2 为什么需要 AOTAutograd？

**问题 1：传统 Autograd 的开销**

```python
# 前向传播
for i in range(1000):
    y = x + 1           # 记录操作 1
    z = y * 2           # 记录操作 2
    loss = z.mean()     # 记录操作 3
    
    # 每次都要构建 backward 图！
    loss.backward()
    # 开销：构建图 + 执行反向传播
```

**AOTAutograd 解决：**

```python
# 只编译一次
compiled_fn = torch.compile(lambda x: ((x + 1) * 2).mean())

for i in range(1000):
    loss = compiled_fn(x)
    loss.backward()  # 使用预编译的反向图！
```

**问题 2：无法联合优化前向和反向**

```python
# 前向操作
y = x + 1      # 需要保存 x 用于反向
z = y * 2      # 需要保存 y 用于反向

# 反向操作
grad_y = grad_z * 2        # 反向 z
grad_x = grad_y * 1        # 反向 y
```

如果提前知道反向图，可以：
- 优化内存使用（不保存不必要的中间值）
- 融合前向和反向操作
- 重新计算而不是保存（memory-efficient）

## 2. AOTAutograd 的工作原理

### 2.1 整体流程

```
┌────────────────────────────────────────┐
│  用户代码：model.forward(input)        │
└────────────┬───────────────────────────┘
             ↓
┌────────────────────────────────────────┐
│  Dynamo：捕获前向计算图                │
│  得到 FX Graph（只有前向操作）         │
└────────────┬───────────────────────────┘
             ↓
┌────────────────────────────────────────┐
│  AOTAutograd：分析前向图               │ ← 本章重点
│  1. 识别哪些值需要梯度                 │
│  2. 生成反向传播图                     │
│  3. 联合优化前向+反向                  │
└────────────┬───────────────────────────┘
             ↓
┌────────────────────────────────────────┐
│  输出：                                │
│  - 优化后的前向图                      │
│  - 反向图                              │
└────────────┬───────────────────────────┘
             ↓
┌────────────────────────────────────────┐
│  Backend Compiler                      │
│  编译为高效代码                        │
└────────────────────────────────────────┘
```

### 2.2 前向图分析

```python
# 示例前向代码
def forward(x):
    y = x * 2           # 操作 1
    z = y + 1           # 操作 2
    loss = z.mean()     # 操作 3
    return loss

# AOTAutograd 分析：
# - x 需要 requires_grad
# - 需要保存什么中间值？
#   - y: 反向时需要
#   - z: 反向时需要
#   - x: 不需要保存（可以从输入获取）
```

### 2.3 反向图生成

```python
# 根据前向图，自动生成反向图

# 前向图：
# x → *2 → y → +1 → z → mean() → loss

# 反向图（自动生成）：
# grad_loss = 1
# grad_z = grad_mean(grad_loss, z)       # mean 的反向
# grad_y = grad_z * 1                    # +1 的反向
# grad_x = grad_y * 2                    # *2 的反向
```

**关键：** 这一切在**第一次前向传播之前**就完成了！

## 3. 详细示例

### 3.1 基本示例

```python
import torch
import torch._functorch.aot_autograd as aot_autograd

def simple_fn(x):
    y = x * 2
    z = y + 1
    return z.sum()

# 使用 AOTAutograd
x = torch.randn(10, requires_grad=True)

# 手动使用 AOTAutograd（通常通过 torch.compile 自动调用）
from torch._dynamo import optimize
from functorch.compile import aot_function

# 定义一个简单的"编译器"（实际只是打印）
def print_compiler(fx_module, args):
    print("前向图:")
    print(fx_module.code)
    return fx_module

# 包装函数
aot_fn = aot_function(simple_fn, fw_compiler=print_compiler, bw_compiler=print_compiler)

# 执行
output = aot_fn(x)
print(f"\n输出: {output}")

output.backward()
print(f"\n梯度: {x.grad}")
```

### 3.2 查看生成的前向和反向图

```python
import torch
from torch._functorch.aot_autograd import aot_module_simplified

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(10, 10))
    
    def forward(self, x):
        return torch.matmul(x, self.weight).sum()

model = SimpleModel()
input_data = torch.randn(10)

# 使用 AOT Autograd
def print_graph(name):
    def compiler(fx_graph, args):
        print(f"\n{'='*50}")
        print(f"{name}")
        print(f"{'='*50}")
        print(fx_graph.code)
        return fx_graph
    return compiler

compiled_model = aot_module_simplified(
    model,
    fw_compiler=print_graph("前向图"),
    bw_compiler=print_graph("反向图"),
)

# 执行
output = compiled_model(input_data)
output.backward()
```

## 4. 联合优化

### 4.1 前向保存优化

```python
# 原始前向代码
def forward(x):
    a = x + 1
    b = a * 2
    c = b + 3
    d = c * 4
    return d.sum()

# 传统 Autograd：保存所有中间值
# 保存：x, a, b, c（用于反向传播）

# AOTAutograd 优化：
# 分析反向图后发现：
# - 反向时只需要 b 和 c
# - a 可以不保存（因为反向时不直接使用）
# 只保存：b, c
```

### 4.2 算子融合（前向+反向）

```python
# 前向操作
y = x + 1      # Kernel 1
z = y * 2      # Kernel 2

# 反向操作（未优化）
grad_y = grad_z * 2    # Kernel 3
grad_x = grad_y * 1    # Kernel 4

# AOTAutograd 融合优化：
# 前向：fused_kernel_fw(x)  → 生成 y, z
# 反向：fused_kernel_bw(grad_z) → 生成 grad_x
# 从 4 个 Kernel 减少到 2 个！
```

### 4.3 重计算（Rematerialization）

```python
# 情况：前向操作便宜，但中间值占用大量内存

def forward(x):
    # 这个操作很便宜（计算快）
    y = torch.sin(x)  
    
    # 但 y 可能很大（占内存）
    # 例如 x.shape = (10000, 10000)
    
    z = y * 2
    return z.sum()

# AOTAutograd 策略：
# - 前向时不保存 y（节省内存）
# - 反向时重新计算 y = torch.sin(x)
# 
# 权衡：用计算换内存
```

**自动重计算策略：**

```python
# PyTorch 会自动决定哪些中间值应该重计算
# 决策因素：
# 1. 计算成本 vs 内存占用
# 2. 是否在反向传播中使用
# 3. 重计算的依赖关系
```

## 5. 源码分析

### 5.1 核心文件位置

```
pytorch/torch/_functorch/
├── aot_autograd.py          # AOTAutograd 核心实现
├── partitioners.py          # 图分割（前向/反向分离）
├── compile_utils.py         # 编译工具
└── compilers.py             # 内置编译器

pytorch/torch/_inductor/
└── fx_passes/
    └── joint_graph.py       # 联合图优化
```

### 5.2 关键函数

**aot_autograd.py 中的核心函数：**

```python
def aot_function(
    fn,
    fw_compiler=None,   # 前向编译器
    bw_compiler=None,   # 反向编译器
    partition_fn=None,  # 图分割函数
    decompositions=None,
):
    """
    AOTAutograd 的入口函数
    
    工作流程：
    1. 调用 fn，追踪前向图
    2. 分析哪些tensor需要梯度
    3. 自动生成反向图
    4. 分别编译前向和反向
    """
    
    def wrapper(*args):
        # 追踪前向图
        with torch.enable_grad():
            fw_graph = make_fx(fn)(*args)
        
        # 生成反向图
        joint_graph = create_joint_forward_backward(fw_graph)
        
        # 分割前向和反向
        fw_module, bw_module = partition_fn(joint_graph)
        
        # 编译
        compiled_fw = fw_compiler(fw_module)
        compiled_bw = bw_compiler(bw_module)
        
        return compiled_fw, compiled_bw
    
    return wrapper
```

### 5.3 反向图生成

```python
# 简化的伪代码

def create_backward_graph(forward_graph, outputs_need_grad):
    """
    从前向图创建反向图
    """
    backward_graph = Graph()
    
    # 1. 遍历前向图（反向顺序）
    for node in reversed(forward_graph.nodes):
        if node.op == 'call_function':
            # 2. 获取该操作的反向规则
            backward_rule = get_backward_rule(node.target)
            
            # 3. 添加反向节点
            grad_inputs = backward_rule(
                grad_output=node.grad,
                forward_inputs=node.args,
                forward_output=node.result
            )
            
            # 4. 连接到反向图
            for inp, grad_inp in zip(node.args, grad_inputs):
                backward_graph.add_edge(node, inp, grad_inp)
    
    return backward_graph
```

## 6. 与传统 Autograd 对比

### 6.1 传统 Autograd

```python
# 运行时构建反向图

x = torch.randn(10, requires_grad=True)

# 前向传播
y = x * 2
# → 创建 MulBackward 节点
# → 保存必要信息（x, 常数2）

z = y + 1
# → 创建 AddBackward 节点
# → 链接到 MulBackward

loss = z.sum()
# → 创建 SumBackward 节点
# → 链接到 AddBackward

# 反向传播
loss.backward()
# → 遍历已构建的反向图
# → 计算梯度
```

**特点：**
- ✅ 灵活：支持任意动态图
- ❌ 开销：每次都要构建反向图
- ❌ 难优化：运行时才知道反向图结构

### 6.2 AOTAutograd

```python
# 编译时构建反向图

def compute(x):
    y = x * 2
    z = y + 1
    return z.sum()

# 第一次调用（编译）
compiled_compute = torch.compile(compute)

x = torch.randn(10, requires_grad=True)
loss = compiled_compute(x)  
# ↑ 在这里就已经生成了完整的反向图！

# 反向传播（使用预编译的反向图）
loss.backward()  # 很快！
```

**特点：**
- ✅ 快速：使用预编译的反向图
- ✅ 可优化：提前知道完整的前向+反向结构
- ⚠️ 相对不灵活：需要固定的计算模式

### 6.3 对比表

| 维度 | 传统 Autograd | AOTAutograd |
|------|--------------|-------------|
| **图构建时机** | 运行时（每次前向） | 编译时（一次） |
| **反向图生成** | 前向时记录 | 编译时提前生成 |
| **优化空间** | 有限 | 大（联合优化） |
| **性能** | 基准 | 更快（20-40%） |
| **内存优化** | 基本 | 高级（重计算等） |
| **灵活性** | 高 | 中等（需Guards） |

## 7. 实际应用

### 7.1 训练加速

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

model = Model().cuda()

# 使用 torch.compile（内部使用 AOTAutograd）
model = torch.compile(model)

# 训练循环
optimizer = torch.optim.Adam(model.parameters())
for batch in dataloader:
    x, y = batch
    x, y = x.cuda(), y.cuda()
    
    # 前向（使用优化的前向图）
    pred = model(x)
    loss = nn.functional.cross_entropy(pred, y)
    
    # 反向（使用预编译的反向图）
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
```

**性能提升：**
- 前向传播：~30% 加速
- 反向传播：~40% 加速
- 总训练时间：~35% 减少

### 7.2 内存优化示例

```python
# 大模型训练时的内存优化

class LargeModel(nn.Module):
    def forward(self, x):
        # 多个大的中间激活值
        x1 = self.layer1(x)      # 大
        x2 = self.layer2(x1)     # 大
        x3 = self.layer3(x2)     # 大
        x4 = self.layer4(x3)     # 大
        return x4

# torch.compile 会自动：
# 1. 分析哪些中间值必须保存
# 2. 对于便宜的操作，使用重计算
# 3. 优化内存布局

model = torch.compile(LargeModel())
```

## 8. 高级特性

### 8.1 自定义分区策略

```python
from torch._functorch.aot_autograd import aot_module_simplified

def custom_partition(joint_graph):
    """
    自定义如何分割前向和反向图
    """
    # 分析图
    fw_nodes = []
    bw_nodes = []
    
    for node in joint_graph.nodes:
        if is_forward_node(node):
            fw_nodes.append(node)
        else:
            bw_nodes.append(node)
    
    # 创建分离的图
    fw_graph = create_graph(fw_nodes)
    bw_graph = create_graph(bw_nodes)
    
    return fw_graph, bw_graph

# 使用自定义分区
model = aot_module_simplified(
    model,
    partition_fn=custom_partition
)
```

### 8.2 查看保存的中间值

```python
import torch
from torch.utils._python_dispatch import TorchDispatchMode

class DebugSavedTensors(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"操作: {func.__name__}")
        result = func(*args, **(kwargs or {}))
        
        # 检查是否保存了tensor
        if hasattr(result, '_saved_tensors'):
            print(f"  保存的tensors: {len(result._saved_tensors)}")
        
        return result

# 使用
with DebugSavedTensors():
    x = torch.randn(10, requires_grad=True)
    y = x * 2
    z = y + 1
    loss = z.sum()
    loss.backward()
```

### 8.3 控制重计算策略

```python
# 手动标记哪些操作应该重计算

import torch
from torch.utils.checkpoint import checkpoint

class ModelWithCheckpoint(nn.Module):
    def forward(self, x):
        # 这部分会被重计算（不保存激活值）
        x = checkpoint(self.expensive_layer1, x)
        x = checkpoint(self.expensive_layer2, x)
        
        # 这部分正常保存
        x = self.cheap_layer(x)
        return x

# torch.compile 会识别 checkpoint 并优化
model = torch.compile(ModelWithCheckpoint())
```

## 9. 调试技巧

### 9.1 查看编译后的代码

```python
import torch
from torch._inductor import config

# 保存生成的代码
config.debug = True
config.trace.enabled = True

model = torch.compile(model)
output = model(input_data)

# 查看生成的前向和反向代码
# 会保存在 /tmp/torchinductor_<user>/ 目录
```

### 9.2 对比优化效果

```python
import torch
import time

model = Model()
x = torch.randn(32, 1024, requires_grad=True)

# 1. 不使用 AOTAutograd
model_eager = model
times_eager = []
for _ in range(100):
    start = time.time()
    loss = model_eager(x).sum()
    loss.backward()
    times_eager.append(time.time() - start)

# 2. 使用 AOTAutograd
model_compiled = torch.compile(model)
times_compiled = []
for _ in range(100):
    start = time.time()
    loss = model_compiled(x).sum()
    loss.backward()
    times_compiled.append(time.time() - start)

print(f"Eager: {sum(times_eager)/len(times_eager)*1000:.2f}ms")
print(f"Compiled: {sum(times_compiled)/len(times_compiled)*1000:.2f}ms")
print(f"加速比: {sum(times_eager)/sum(times_compiled):.2f}x")
```

## 10. 小结

### 核心要点

1. **AOTAutograd = 提前编译的自动微分**
2. **核心优势：**
   - 编译时生成反向图（而非运行时）
   - 联合优化前向和反向
   - 智能的内存管理和重计算
3. **关键技术：**
   - 前向图分析
   - 自动反向图生成
   - 算子融合
   - 重计算策略
4. **性能提升：** 20-40% 的训练加速

### 在 torch.compile 中的角色

```
torch.compile 的三层架构：
1. Dynamo：捕获前向图
2. AOTAutograd：生成反向图，联合优化 ← 本章
3. Inductor：代码生成和执行
```

### 与其他技术的关系

- **vs 传统 Autograd**：提前编译 vs 运行时构建
- **依赖 TorchFX**：使用 FX IR 表示前向和反向图
- **支持 Dynamo**：接收 Dynamo 捕获的图

### 下一步

现在你理解了如何优化前向和反向传播，接下来我们学习 PyTorch 的底层分发机制：

📖 [第七章：PyTorch Dispatcher 分发机制](./07_pytorch_dispatch.md) - 算子如何路由到不同的硬件后端

---

## 💡 练习题

1. 用 `aot_function` 包装一个简单函数，打印前向和反向图
2. 对比有无 AOTAutograd 的训练速度
3. 分析一个模型应该保存哪些中间值用于反向传播

**继续学习** → [PyTorch Dispatcher 分发机制](./07_pytorch_dispatch.md)

