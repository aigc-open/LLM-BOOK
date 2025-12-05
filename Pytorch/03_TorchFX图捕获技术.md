# 第三章：TorchFX 图捕获技术

## 🎯 本章目标

- 理解 TorchFX 的设计理念和优势
- 掌握符号追踪（Symbolic Tracing）技术
- 学习 TorchFX IR 的结构和操作
- 能够编写图变换和优化

## 1. 什么是 TorchFX？

### 1.1 简单理解

**TorchScript 的问题：**
```python
# TorchScript 把代码变成"黑盒"
scripted_model = torch.jit.script(model)
# 难以修改、难以分析内部结构
```

**TorchFX 的解决方案：**
```python
# TorchFX 提供"透明的"、可编辑的图表示
traced_graph = torch.fx.symbolic_trace(model)
# 可以查看、修改每个操作！
```

**类比：**
- TorchScript = 编译后的二进制文件（难以修改）
- TorchFX = 源代码（可以自由编辑）

### 1.2 核心特点

1. **Python-First**：完全是 Python 对象，易于操作
2. **可编程**：可以轻松修改计算图
3. **保留动态性**：与 Python 生态兼容
4. **轻量级**：无需重量级编译器

## 2. 符号追踪（Symbolic Tracing）

### 2.1 基本用法

```python
import torch
import torch.nn as nn
import torch.fx as fx

# 定义一个简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

# 符号追踪
model = SimpleModel()
traced_graph = fx.symbolic_trace(model)

# 查看生成的代码
print(traced_graph.code)
```

**输出：**
```python
def forward(self, x):
    linear1 = self.linear1(x)
    relu = torch.relu(linear1)
    linear2 = self.linear2(relu)
    return linear2
```

### 2.2 与 torch.jit.trace 的区别

| 特性 | torch.jit.trace | torch.fx.symbolic_trace |
|------|----------------|------------------------|
| **执行方式** | 运行实际计算 | 符号执行（不计算） |
| **返回类型** | ScriptModule | GraphModule |
| **可修改性** | ❌ 困难 | ✅ 容易 |
| **性能开销** | 需要实际计算 | 几乎无开销 |

```python
# jit.trace 需要实际数据
import torch.jit as jit
traced_jit = jit.trace(model, torch.randn(1, 10))  # 需要示例输入

# fx.symbolic_trace 不需要
traced_fx = fx.symbolic_trace(model)  # 无需示例输入！
```

## 3. TorchFX IR 详解

### 3.1 IR 的组成

TorchFX 的 IR 由以下几部分组成：

```python
traced = fx.symbolic_trace(model)

# 1. Graph：计算图
print(traced.graph)

# 2. Nodes：图中的节点
for node in traced.graph.nodes:
    print(node)

# 3. Code：生成的 Python 代码
print(traced.code)
```

### 3.2 Node 类型

TorchFX 中有 6 种节点类型：

```python
import torch.fx as fx

traced = fx.symbolic_trace(SimpleModel())

for node in traced.graph.nodes:
    print(f"{node.name}: {node.op}")
```

**节点类型详解：**

| 类型 | 说明 | 示例 |
|------|------|------|
| **placeholder** | 输入参数 | `x` (函数参数) |
| **get_attr** | 获取模块属性 | `self.linear1` |
| **call_function** | 调用函数 | `torch.relu(x)` |
| **call_method** | 调用方法 | `x.view(...)` |
| **call_module** | 调用子模块 | `self.linear1(x)` |
| **output** | 输出 | `return x` |

### 3.3 详细示例

```python
class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(5, 5))
        self.linear = nn.Linear(5, 3)
    
    def forward(self, x):
        # 各种不同类型的操作
        w = self.weight              # get_attr
        x = torch.matmul(x, w)       # call_function
        x = x.relu()                 # call_method
        x = self.linear(x)           # call_module
        return x

traced = fx.symbolic_trace(ExampleModel())

print("=" * 50)
print("图结构：")
print(traced.graph)

print("\n" + "=" * 50)
print("详细节点信息：")
for node in traced.graph.nodes:
    print(f"名称: {node.name:15} | 类型: {node.op:15} | 目标: {node.target}")
```

**输出：**
```
==================================================
图结构：
graph():
    %x : [#users=1] = placeholder[target=x]
    %weight : [#users=1] = get_attr[target=weight]
    %matmul : [#users=1] = call_function[target=torch.matmul](args = (%x, %weight), kwargs = {})
    %relu : [#users=1] = call_method[target=relu](args = (%matmul,), kwargs = {})
    %linear : [#users=1] = call_module[target=linear](args = (%relu,), kwargs = {})
    return linear

==================================================
详细节点信息：
名称: x               | 类型: placeholder     | 目标: x
名称: weight          | 类型: get_attr        | 目标: weight
名称: matmul          | 类型: call_function   | 目标: <built-in function matmul>
名称: relu            | 类型: call_method     | 目标: relu
名称: linear          | 类型: call_module     | 目标: linear
名称: output          | 类型: output          | 目标: output
```

### 3.4 Node 的属性

每个 Node 都有丰富的属性：

```python
traced = fx.symbolic_trace(model)

for node in traced.graph.nodes:
    if node.op == 'call_function':
        print(f"节点名: {node.name}")
        print(f"  操作类型: {node.op}")
        print(f"  目标函数: {node.target}")
        print(f"  参数: {node.args}")
        print(f"  关键字参数: {node.kwargs}")
        print(f"  使用者: {[u.name for u in node.users]}")
        print(f"  使用次数: {len(node.users)}")
        print()
```

## 4. 图变换（Graph Transformation）

### 4.1 基本操作：遍历和查看

```python
def analyze_graph(traced_model):
    """分析图的基本信息"""
    print("=" * 50)
    print("图分析报告")
    print("=" * 50)
    
    # 统计节点类型
    op_counts = {}
    for node in traced_model.graph.nodes:
        op_counts[node.op] = op_counts.get(node.op, 0) + 1
    
    print("\n节点类型统计:")
    for op, count in op_counts.items():
        print(f"  {op}: {count}")
    
    # 找到所有函数调用
    print("\n函数调用:")
    for node in traced_model.graph.nodes:
        if node.op == 'call_function':
            print(f"  {node.name}: {node.target.__name__}")
    
    # 找到所有模块调用
    print("\n模块调用:")
    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
            print(f"  {node.name}: {type(traced_model.get_submodule(node.target)).__name__}")

# 使用
traced = fx.symbolic_trace(SimpleModel())
analyze_graph(traced)
```

### 4.2 修改图：替换操作

**示例 1：将 ReLU 替换为 GELU**

```python
import torch.nn.functional as F

def replace_relu_with_gelu(traced_model):
    """将所有 ReLU 替换为 GELU"""
    graph = traced_model.graph
    
    # 遍历所有节点
    for node in graph.nodes:
        # 找到 relu 调用
        if node.op == 'call_function' and node.target == torch.relu:
            # 替换目标函数
            node.target = F.gelu
            print(f"替换节点 {node.name}: relu → gelu")
    
    # 重新编译
    traced_model.recompile()
    return traced_model

# 使用
traced = fx.symbolic_trace(SimpleModel())
print("原始代码:")
print(traced.code)

modified = replace_relu_with_gelu(traced)
print("\n修改后的代码:")
print(modified.code)
```

**输出：**
```python
# 原始代码:
def forward(self, x):
    linear1 = self.linear1(x)
    relu = torch.relu(linear1)
    linear2 = self.linear2(relu)
    return linear2

# 修改后的代码:
def forward(self, x):
    linear1 = self.linear1(x)
    relu = torch.nn.functional.gelu(linear1)  # 已替换！
    linear2 = self.linear2(relu)
    return linear2
```

### 4.3 插入新节点

**示例 2：在每个操作后插入打印**

```python
def insert_print_statements(traced_model):
    """在每个操作后插入打印语句（用于调试）"""
    graph = traced_model.graph
    
    # 需要在迭代时修改，所以先收集节点
    nodes_to_instrument = []
    for node in graph.nodes:
        if node.op in ['call_function', 'call_module', 'call_method']:
            nodes_to_instrument.append(node)
    
    # 插入打印节点
    for node in nodes_to_instrument:
        with graph.inserting_after(node):
            # 插入 print 调用
            print_node = graph.call_function(
                print,
                args=(f"[DEBUG] {node.name} shape:", node,)
            )
    
    graph.lint()  # 验证图的正确性
    traced_model.recompile()
    return traced_model

# 使用
traced = fx.symbolic_trace(SimpleModel())
instrumented = insert_print_statements(traced)
print(instrumented.code)
```

### 4.4 删除节点

**示例 3：移除所有 Dropout**

```python
def remove_dropout(traced_model):
    """移除所有 Dropout 层"""
    graph = traced_model.graph
    
    # 找到所有 Dropout 节点
    nodes_to_remove = []
    for node in graph.nodes:
        if node.op == 'call_module':
            module = traced_model.get_submodule(node.target)
            if isinstance(module, nn.Dropout):
                nodes_to_remove.append(node)
    
    # 删除节点
    for node in nodes_to_remove:
        # Dropout 的输出就是输入，直接替换
        node.replace_all_uses_with(node.args[0])
        graph.erase_node(node)
        print(f"移除节点: {node.name}")
    
    graph.lint()
    traced_model.recompile()
    return traced_model
```

## 5. 实际应用：算子融合

### 5.1 Conv-BN 融合

卷积和 BatchNorm 可以融合为一个操作，提升性能：

```python
def fuse_conv_bn(traced_model):
    """
    融合 Conv2d + BatchNorm2d
    原理：y = BN(Conv(x)) 可以合并为一个 Conv
    """
    graph = traced_model.graph
    modules = dict(traced_model.named_modules())
    
    for node in graph.nodes:
        # 查找 BatchNorm 节点
        if node.op == 'call_module':
            bn = modules.get(node.target)
            if not isinstance(bn, nn.BatchNorm2d):
                continue
            
            # 检查前一个节点是否是 Conv2d
            if len(node.args) > 0:
                conv_node = node.args[0]
                if conv_node.op == 'call_module':
                    conv = modules.get(conv_node.target)
                    if isinstance(conv, nn.Conv2d):
                        # 执行融合
                        fused_conv = fuse_conv_bn_eval(conv, bn)
                        
                        # 替换 conv
                        setattr(traced_model, conv_node.target, fused_conv)
                        
                        # 移除 BN
                        node.replace_all_uses_with(conv_node)
                        graph.erase_node(node)
                        
                        print(f"融合: {conv_node.target} + {node.target}")
    
    graph.lint()
    traced_model.recompile()
    return traced_model

def fuse_conv_bn_eval(conv, bn):
    """
    数学上融合 Conv 和 BN
    """
    # 获取参数
    conv_weight = conv.weight
    conv_bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
    
    bn_weight = bn.weight
    bn_bias = bn.bias
    bn_mean = bn.running_mean
    bn_var = bn.running_var
    bn_eps = bn.eps
    
    # 计算融合后的权重和偏置
    bn_std = torch.sqrt(bn_var + bn_eps)
    fused_weight = conv_weight * (bn_weight / bn_std).reshape(-1, 1, 1, 1)
    fused_bias = (conv_bias - bn_mean) * bn_weight / bn_std + bn_bias
    
    # 创建新的 Conv
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=True
    )
    fused_conv.weight.data = fused_weight
    fused_conv.bias.data = fused_bias
    
    return fused_conv
```

### 5.2 使用融合

```python
class ConvBNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 原始模型
model = ConvBNModel()
model.eval()  # 必须是 eval 模式

# Trace
traced = fx.symbolic_trace(model)
print("原始代码:")
print(traced.code)

# 融合
fused = fuse_conv_bn(traced)
print("\n融合后代码:")
print(fused.code)

# 验证正确性
input_data = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output_original = model(input_data)
    output_fused = fused(input_data)
    print(f"\n输出差异: {(output_original - output_fused).abs().max().item()}")
```

## 6. 高级特性

### 6.1 Proxy 对象

在 symbolic_trace 过程中，实际操作的是 Proxy 对象：

```python
import torch.fx as fx

class MyTracer(fx.Tracer):
    def trace(self, root, concrete_args=None):
        print("开始追踪...")
        graph = super().trace(root, concrete_args)
        print("追踪完成！")
        return graph

tracer = MyTracer()
traced = tracer.trace(SimpleModel())
```

### 6.2 自定义追踪规则

```python
class CustomTracer(fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        """
        决定哪些模块应该作为"叶子"（不进一步追踪内部）
        """
        # 默认 nn.Linear 会被追踪，这里我们让它成为叶子
        if isinstance(m, nn.Linear):
            return True
        return super().is_leaf_module(m, module_qualified_name)

tracer = CustomTracer()
traced = tracer.trace(SimpleModel())
```

### 6.3 Interpreter 模式

手动执行图：

```python
class CustomInterpreter(fx.Interpreter):
    def call_function(self, target, args, kwargs):
        """拦截所有函数调用"""
        print(f"调用函数: {target.__name__}")
        result = super().call_function(target, args, kwargs)
        print(f"  结果形状: {result.shape if isinstance(result, torch.Tensor) else type(result)}")
        return result

traced = fx.symbolic_trace(SimpleModel())
interpreter = CustomInterpreter(traced)

input_data = torch.randn(2, 10)
output = interpreter.run(input_data)
```

## 7. TorchFX 在 PyTorch 2.0 中的角色

### 7.1 作为 IR 层

```
torch.compile 流程:
    ↓
Dynamo (捕获字节码)
    ↓
TorchFX Graph (IR 表示) ← 我们在这里！
    ↓
AOTAutograd (生成前向/反向图)
    ↓
Backend Compiler (TorchInductor/...)
```

### 7.2 查看 torch.compile 生成的图

```python
import torch
import torch._dynamo as dynamo

model = SimpleModel()
input_data = torch.randn(1, 10)

# 导出 torch.compile 捕获的图
exported, _ = dynamo.export(model, input_data)

# 这就是一个 TorchFX GraphModule！
print(type(exported))  # <class 'torch.fx.graph_module.GraphModule'>
print(exported.code)
```

## 8. 源码位置

在 `pytorch/torch/fx/` 目录下：

```
torch/fx/
├── __init__.py           # 主要 API
├── graph.py              # Graph 和 Node 定义
├── graph_module.py       # GraphModule 实现
├── interpreter.py        # Interpreter 模式
├── proxy.py              # Proxy 对象
├── symbolic_trace.py     # 符号追踪核心
├── node.py               # Node 详细实现
└── passes/               # 各种图优化 pass
    ├── shape_prop.py     # 形状推导
    ├── split_module.py   # 图分割
    └── ...
```

**关键文件：**
- `torch/fx/symbolic_trace.py`：符号追踪的主要逻辑
- `torch/fx/graph.py`：Graph 的数据结构
- `torch/fx/graph_module.py`：GraphModule 实现

## 9. 小结

### 核心要点

1. **TorchFX = 可编程的计算图表示**
2. **符号追踪**：不执行实际计算，只记录操作
3. **6 种节点类型**：placeholder, get_attr, call_function, call_method, call_module, output
4. **图变换**：可以轻松修改、优化计算图
5. **在 PyTorch 2.0 中作为 IR 层**

### 关键 API

```python
# 1. 追踪
traced = torch.fx.symbolic_trace(model)

# 2. 查看
print(traced.code)
print(traced.graph)

# 3. 遍历
for node in traced.graph.nodes:
    print(node)

# 4. 修改
node.target = new_function
graph.erase_node(node)

# 5. 重新编译
traced.recompile()
```

### 典型应用

- ✅ 算子融合
- ✅ 量化
- ✅ 自定义优化
- ✅ 图分割（模型并行）
- ✅ 调试和可视化

### 下一步

现在你掌握了 TorchFX 这个强大的图操作工具，接下来我们学习：

📖 [第四章：Lazy Tensor 延迟执行机制](./04_lazy_tensor.md) - 理解延迟执行的原理

---

## 💡 练习题

1. 用 TorchFX 追踪一个自定义模型，打印所有节点信息
2. 编写一个图变换：将所有 `sigmoid` 替换为 `tanh`
3. 实现一个 Interpreter，统计每个操作的执行时间

**继续学习** → [Lazy Tensor 延迟执行机制](./04_lazy_tensor.md)

