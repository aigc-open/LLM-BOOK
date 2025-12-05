# 第五章：Torch Dynamo 动态图编译

## 本章目标

- 深入理解 Dynamo 的核心作用
- 掌握字节码捕获技术
- 理解 Guards 机制
- 学习 Dynamo 如何实现动态图到静态图的转换

## 1. Torch Dynamo 是什么？

### 1.1 简单理解

**问题：如何捕获 Python 动态图？**

```python
# Python 代码很灵活
def model_forward(x, use_dropout=True):
x = self.linear1(x)
if use_dropout: # 动态控制流
x = self.dropout(x)
x = torch.relu(x)
return x

# 如何把这个转成静态图优化？
# TorchScript：要求用户改代码
# Dynamo：自动捕获，无需改代码
```

**Dynamo 的magic：**
```python
# 用户只需一行
compiled_fn = torch.compile(model_forward)

# Dynamo 自动：
# 1. 拦截 Python 字节码
# 2. 识别 PyTorch 操作
# 3. 构建计算图
# 4. 交给后端优化
```

### 1.2 Dynamo 在 torch.compile 中的位置

```
torch.compile(model)


+-------------------------------------+
| Torch Dynamo (本章重点) |
| - 拦截 Python 字节码 |
| - 提取 PyTorch 操作 |
| - 构建 FX 图 |
| - 处理控制流和动态性 |
|--------------+----------------------+


[FX Graph]


+-------------------------------------+
| AOTAutograd (下一章) |
| - 生成前向/反向图 |
|--------------+----------------------+


+-------------------------------------+
| Backend Compiler |
| - TorchInductor |
| - 生成优化代码 |
|-------------------------------------+
```

## 2. 字节码捕获：Dynamo 的核心技术

### 2.1 Python 字节码基础

Python 代码会被编译成字节码：

```python
import dis

def simple_add(a, b):
return a + b

# 查看字节码
dis.dis(simple_add)
```

**输出：**
```
2 0 LOAD_FAST 0 (a)
2 LOAD_FAST 1 (b)
4 BINARY_ADD
6 RETURN_VALUE
```

### 2.2 Dynamo 如何拦截字节码

**正常执行流程：**
```
Python 源码 -> 字节码 -> Python 解释器执行
```

**Dynamo 拦截：**
```
Python 源码 -> 字节码 -> Dynamo Frame Evaluation Hook ->


分析 PyTorch 操作 -> 构建 FX 图 -> 编译优化


执行优化后的代码
```

### 2.3 实际示例

```python
import torch

def example_function(x, y):
a = x + y # PyTorch 操作
b = a * 2 # PyTorch 操作
c = torch.sin(b) # PyTorch 操作
return c

# 编译
compiled_fn = torch.compile(example_function)

# 第一次执行时，Dynamo 会：
x = torch.randn(10)
y = torch.randn(10)
result = compiled_fn(x, y) # 触发捕获和编译

# 查看 Dynamo 捕获的图
import torch._dynamo as dynamo
explanation = dynamo.explain(example_function)(x, y)
print(explanation)
```

## 3. Guards 机制

### 3.1 什么是 Guards？

**问题：动态 Python 代码的挑战**

```python
def dynamic_fn(x, threshold):
if x.sum() > threshold: # 数据依赖的分支
return x * 2
else:
return x + 1

# 第一次调用：x.sum() = 10, threshold = 5 -> 走第一个分支
# 第二次调用：x.sum() = 2, threshold = 5 -> 走第二个分支
# 如何处理？
```

**Guards = 假设的条件检查**

Dynamo 会生成类似这样的代码：

```python
# Guard: 检查是否走第一个分支
if x.sum() > threshold: # Guard 通过
return optimized_branch1(x)
else: # Guard 失败，重新编译
return optimized_branch2(x)
```

### 3.2 Guards 的类型

**1. Shape Guards（形状检查）**

```python
def shape_dependent(x):
if x.shape[0] > 100:
return x.sum()
return x.mean()

# Dynamo 生成 Guard:
# if x.shape[0] == 128: # 假设第一次调用时 batch_size=128
# return compiled_version_for_128(x)
# else:
# recompile()
```

**2. Type Guards（类型检查）**

```python
# Guard: 检查 x 是否是 Tensor
assert isinstance(x, torch.Tensor)

# Guard: 检查 dtype
assert x.dtype == torch.float32
```

**3. Value Guards（值检查）**

```python
def value_dependent(x, mode):
if mode == "train":
return dropout(x)
else:
return x

# Dynamo Guard:
# if mode == "train": # 记住第一次的值
# return compiled_train_version(x)
```

### 3.3 Guards 失败和重新编译

```python
import torch

def example(x):
return x + 1

compiled_example = torch.compile(example)

# 第一次：shape = (10,)
x1 = torch.randn(10)
compiled_example(x1) # 编译，生成 Guard: shape == (10,)

# 第二次：shape = (10,)
x2 = torch.randn(10)
compiled_example(x2) # Guard 通过，使用缓存的编译结果

# 第三次：shape = (20,) ️
x3 = torch.randn(20)
compiled_example(x3) # Guard 失败，重新编译！
```

**查看 Guards：**

```python
import torch._dynamo as dynamo

# 启用详细日志
import logging
torch._dynamo.config.log_level = logging.INFO
torch._dynamo.config.verbose = True

compiled_fn = torch.compile(example)
compiled_fn(torch.randn(10))

# 可以看到生成的 Guards
```

## 4. Graph Breaks（图中断）

### 4.1 什么是 Graph Break？

有些 Python 代码无法被捕获到图中：

```python
def example_with_break(x):
a = x + 1 # 可以捕获
b = a * 2 # 可以捕获

print(b.shape) # Graph Break！print 无法捕获

c = b - 1 # 可以捕获（新的图）
return c
```

**Dynamo 会分成多个图：**

```
图 1: x -> +1 -> *2 -> b

(Graph Break)
Python 执行: print(b.shape)


图 2: b -> -1 -> c
```

### 4.2 常见的 Graph Breaks

```python
# 1. print 语句
print(x) #

# 2. 访问全局变量（某些情况）
global_var = []
global_var.append(x) #

# 3. 不支持的库
import numpy as np
np_array = x.numpy() #

# 4. 动态属性访问
attr_name = "weight"
value = getattr(model, attr_name) # (某些情况)

# 5. 复杂的控制流
for i in range(x.item()): # x.item() 需要实际执行
...
```

### 4.3 查看 Graph Breaks

```python
import torch._dynamo as dynamo

def function_with_breaks(x):
a = x + 1
print("Debug:", a.shape) # Graph break
b = a * 2
return b

# 查看 graph breaks
explanation = dynamo.explain(function_with_breaks)(torch.randn(10))
print(explanation)
```

**输出示例：**
```
Dynamo produced 2 graphs:
Graph 1: 1 node
LOAD_FAST, LOAD_CONST, BINARY_ADD
[Graph Break: print]
Graph 2: 1 node
LOAD_FAST, LOAD_CONST, BINARY_MULTIPLY
```

### 4.4 减少 Graph Breaks

```python
# 不好：频繁 Graph Break
def bad_example(x):
a = x + 1
print(a.shape) # Break 1
b = a * 2
print(b.shape) # Break 2
c = b - 1
return c

# 好：避免中间的 print
def good_example(x):
a = x + 1
b = a * 2
c = b - 1
return c
# 最后再 print 调试信息
```

## 5. 控制流处理

### 5.1 简单的条件语句

```python
def conditional_example(x, flag: bool):
if flag:
return x * 2
else:
return x + 1

# Dynamo 处理方式：
# 1. 第一次调用 flag=True，编译第一个分支
# 2. 添加 Guard: if flag == True
# 3. 如果后续 flag=False，Guard 失败，重新编译
```

### 5.2 数据依赖的控制流

```python
def data_dependent(x):
if x.sum() > 0: # 依赖数据的值
return x * 2
else:
return x - 1

# Dynamo 的挑战：
# - x.sum() 需要实际计算
# - 可能导致 Graph Break 或者重新编译
```

**Dynamo 的策略：**

1. **Specialize（特化）：** 为每个分支编译不同的版本
2. **Guard：** 记住条件，当条件变化时重新编译

### 5.3 循环处理

```python
# 固定次数的循环
def fixed_loop(x):
for i in range(10): # 可以展开
x = x + 1
return x

# ️ 动态次数的循环
def dynamic_loop(x, n):
for i in range(n): # 难以处理
x = x + 1
return x

# 数据依赖的循环
def data_loop(x):
while x.sum() > 0: # 极难处理
x = x - 0.1
return x
```

## 6. Dynamo 的配置和调试

### 6.1 配置选项

```python
import torch._dynamo as dynamo

# 查看配置
print(dynamo.config)

# 常用配置
dynamo.config.verbose = True # 详细日志
dynamo.config.suppress_errors = False # 不抑制错误
dynamo.config.log_level = logging.DEBUG

# 缓存大小
dynamo.config.cache_size_limit = 64 # 最多缓存 64 个编译版本
```

### 6.2 调试工具

```python
import torch._dynamo as dynamo

# 1. explain: 查看编译信息
def my_function(x):
return x + 1

explanation = dynamo.explain(my_function)(torch.randn(10))
print(explanation)

# 2. 禁用 Dynamo（对比测试）
@dynamo.disable
def no_compile_function(x):
return x + 1

# 3. 重置 Dynamo
dynamo.reset() # 清空所有缓存

# 4. 查看生成的代码
compiled_fn = torch.compile(my_function)
print(compiled_fn.__code__)
```

### 6.3 常见问题排查

```python
# 问题 1：编译错误
try:
compiled_fn = torch.compile(problematic_function)
compiled_fn(input_data)
except Exception as e:
print(f"编译错误: {e}")
# 尝试禁用编译运行
result = problematic_function(input_data)

# 问题 2：性能反而变慢
# 可能原因：
# - 太多 Graph Breaks
# - 频繁重新编译（Guards 失败）
# - 首次编译开销

# 解决方案：
# - 减少 Graph Breaks
# - 固定输入形状
# - 预热（warm-up）
```

## 7. 源码分析

### 7.1 核心文件位置

```
pytorch/torch/_dynamo/
|-- __init__.py # 公共 API
|-- eval_frame.py # 字节码拦截核心
|-- symbolic_convert.py # 字节码转 FX 图
|-- guards.py # Guards 生成和检查
|-- bytecode_transformation.py # 字节码转换
|-- output_graph.py # 输出图管理
|-- variables/ # 变量追踪
|-- tensor.py # Tensor 变量
|-- functions.py # 函数变量
|-- ...
```

### 7.2 关键代码片段

**字节码拦截（eval_frame.py）：**

```python
# 简化版本
def _custom_eval_frame(frame, cache_size):
"""Dynamo 的核心：拦截 Python frame 执行"""

# 检查是否应该编译这个 frame
if should_compile(frame):
# 转换字节码为 FX 图
graph = convert_frame_to_graph(frame)

# 编译图
compiled_fn = compile_graph(graph)

# 执行编译后的代码
return compiled_fn(*frame.f_locals.values())
else:
# 正常执行
return DEFAULT_EVAL_FRAME(frame)
```

**Guards 生成（guards.py）：**

```python
# 简化版本
class GuardBuilder:
def __init__(self):
self.guards = []

def add_tensor_match_guard(self, tensor, name):
"""添加 Tensor 匹配的 Guard"""
# 检查形状
self.guards.append(
f"{name}.shape == {tensor.shape}"
)
# 检查 dtype
self.guards.append(
f"{name}.dtype == {tensor.dtype}"
)
# 检查设备
self.guards.append(
f"{name}.device == {tensor.device}"
)
```

### 7.3 执行流程

```python
# 执行流程（伪代码）

def torch_compile(fn):
def wrapper(*args, **kwargs):
# 1. 检查缓存
cache_key = compute_cache_key(args, kwargs)
if cache_key in CACHE:
compiled_fn = CACHE[cache_key]

# 2. 检查 Guards
if check_guards(compiled_fn.guards, args, kwargs):
# Guards 通过，使用缓存
return compiled_fn(*args, **kwargs)

# 3. Guards 失败或无缓存，重新编译
graph = capture_graph(fn, args, kwargs) # Dynamo 捕获
guards = extract_guards(graph)

optimized_fn = backend_compile(graph) # 后端编译
optimized_fn.guards = guards

# 4. 缓存
CACHE[cache_key] = optimized_fn

# 5. 执行
return optimized_fn(*args, **kwargs)

return wrapper
```

## 8. Dynamo 的优势与限制

### 8.1 优势

**无侵入性**
```python
# 只需一行
model = torch.compile(model)
# 无需修改模型代码！
```

**支持动态 Python**
```python
# 支持控制流
if condition:
x = branch1(x)
else:
x = branch2(x)
```

**灵活的后端**
```python
# 可以选择不同的编译后端
model = torch.compile(model, backend="inductor")
model = torch.compile(model, backend="aot_eager")
```

### 8.2 限制

**首次编译开销**
```python
# 第一次调用很慢（编译）
output = compiled_model(input) # 可能需要几秒

# 后续调用快
output = compiled_model(input) # 毫秒级
```

**Graph Breaks 影响性能**
```python
# 太多 Graph Breaks 会降低性能
def many_breaks(x):
x = x + 1
print(x) # Break
x = x * 2
print(x) # Break
x = x - 1
return x
```

**动态形状支持有限**
```python
# 形状变化导致重新编译
model(torch.randn(32, 128)) # 编译版本 1
model(torch.randn(64, 128)) # 重新编译！
```

## 9. 实战技巧

### 9.1 预热策略

```python
model = torch.compile(model)

# 预热：触发编译
dummy_input = torch.randn(batch_size, input_dim, device=device)
with torch.no_grad():
_ = model(dummy_input)

# 之后正常使用
for batch in dataloader:
output = model(batch) # 已经编译好，很快
```

### 9.2 固定输入形状

```python
# 不好：动态 batch size
for batch in dataloader: # batch size 可能变化
output = model(batch)

# 好：固定 batch size
from torch.utils.data import DataLoader

dataloader = DataLoader(
dataset,
batch_size=32, # 固定
drop_last=True # 丢弃最后不完整的 batch
)
```

### 9.3 逐步编译

```python
# 不要一次性编译整个复杂系统
# 先编译性能瓶颈部分

class ComplexModel(nn.Module):
def __init__(self):
super().__init__()
self.encoder = Encoder()
self.decoder = Decoder()

def forward(self, x):
# 只编译 encoder（假设它是瓶颈）
encoded = torch.compile(self.encoder)(x)
decoded = self.decoder(encoded)
return decoded
```

## 10. 小结

### 核心要点

1. **Dynamo = 字节码拦截 + 图捕获引擎**
2. **核心技术：**
- 拦截 Python frame execution
- 分析字节码，提取 PyTorch 操作
- 构建 FX 图
- 生成 Guards
3. **Guards**：确保编译版本的正确性
4. **Graph Breaks**：无法捕获的操作会中断图
5. **动态性支持**：通过 Guards 和重新编译

### 关键概念

- **Bytecode Capture**：字节码捕获
- **Guards**：条件检查
- **Graph Break**：图中断
- **Specialization**：特化编译
- **Recompilation**：重新编译

### 使用建议

1. 减少 Graph Breaks
2. 固定输入形状
3. 预热模型
4. 从简单场景开始
5. ️ 注意首次编译开销

### 下一步

Dynamo 捕获了计算图后，接下来需要生成高效的前向和反向传播：

[第六章：AOTAutograd 自动微分](./06_aotautograd.md) - 提前编译自动微分

---

## 练习题

1. 使用 `dynamo.explain()` 分析一个函数的编译过程
2. 编写一个有 Graph Break 的函数，观察生成的多个图
3. 测试不同输入形状下的重新编译行为

**继续学习** -> [AOTAutograd 自动微分](./06_aotautograd.md)

