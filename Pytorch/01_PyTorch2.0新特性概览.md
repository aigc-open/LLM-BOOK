# 第一章：PyTorch 2.0 新特性概览

## 🎯 本章目标

- 理解 PyTorch 2.0 带来的重大变革
- 掌握 `torch.compile` 的使用方法
- 了解性能提升的原理
- 认识 PyTorch 2.0 的完整技术栈

## 1. PyTorch 2.0 的革命性变化

### 1.1 为什么需要 PyTorch 2.0？

在 PyTorch 2.0 之前，虽然 PyTorch 以**易用性**和**灵活性**著称（动态图的优势），但在**性能**方面一直落后于静态图框架（如 TensorFlow）。

**痛点：**
- 🐌 Python 解释器开销大
- 🔄 频繁的 CPU-GPU 数据传输
- 💾 内存使用不够优化
- ⚡ 算子融合机会少

**PyTorch 2.0 的解决方案：**
在保持动态图灵活性的同时，引入编译技术获得静态图的性能！

### 1.2 核心特性：torch.compile

`torch.compile` 是 PyTorch 2.0 的核心特性，**一行代码即可加速模型**：

```python
import torch

# 定义一个简单的模型
model = MyModel()

# 神奇的一行代码！
compiled_model = torch.compile(model)

# 之后正常使用即可
output = compiled_model(input_data)
```

**真的就这么简单！** 🎉

## 2. torch.compile 的工作原理

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────┐
│              用户 Python 代码                        │
│         model = torch.compile(model)                │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  第一层：Torch Dynamo (图捕获)                       │
│  - 捕获 Python 字节码                                │
│  - 识别 PyTorch 操作                                 │
│  - 构建计算图                                        │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  第二层：AOTAutograd (自动微分)                      │
│  - 生成前向图                                        │
│  - 生成反向图                                        │
│  - 联合优化                                          │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  第三层：后端编译器 (代码生成)                       │
│  - TorchInductor (默认)                             │
│  - 生成优化的 C++/Triton 代码                       │
│  - 算子融合、内存优化                                │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              优化后的可执行代码                      │
│         在 GPU/CPU 上高效运行                        │
└─────────────────────────────────────────────────────┘
```

### 2.2 关键技术栈

| 组件 | 作用 | 详细章节 |
|------|------|----------|
| **Torch Dynamo** | 动态捕获 Python 字节码，构建图 | 第 5 章 |
| **AOTAutograd** | 提前编译自动微分 | 第 6 章 |
| **TorchInductor** | 代码生成后端（Triton/C++） | - |
| **TorchFX** | 图表示和变换 | 第 3 章 |
| **PrimTorch** | 原语算子集合 | - |

## 3. 性能提升示例

### 3.1 简单示例

```python
import torch
import torch.nn as nn
import time

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x

# 准备数据
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleModel().to(device)
input_data = torch.randn(128, 1024, device=device)

# 1. 原始 PyTorch（eager mode）
model_eager = model
start = time.time()
for _ in range(100):
    output = model_eager(input_data)
torch.cuda.synchronize()
eager_time = time.time() - start

# 2. 使用 torch.compile
model_compiled = torch.compile(model)
start = time.time()
for _ in range(100):
    output = model_compiled(input_data)
torch.cuda.synchronize()
compiled_time = time.time() - start

print(f"Eager mode: {eager_time:.4f}s")
print(f"Compiled mode: {compiled_time:.4f}s")
print(f"加速比: {eager_time/compiled_time:.2f}x")
```

**典型输出：**
```
Eager mode: 0.3250s
Compiled mode: 0.1820s
加速比: 1.79x
```

### 3.2 真实模型性能提升

根据 PyTorch 官方测试（在 A100 GPU 上）：

| 模型 | 训练加速 | 推理加速 |
|------|---------|---------|
| ResNet-50 | 1.43x | 1.71x |
| BERT-base | 1.51x | 1.87x |
| GPT-2 | 1.36x | 1.54x |
| Vision Transformer | 1.44x | 1.66x |

## 4. torch.compile 的使用模式

### 4.1 基本用法

```python
# 方式 1：编译整个模型
model = torch.compile(model)

# 方式 2：编译单个函数
@torch.compile
def my_function(x, y):
    return torch.matmul(x, y) + torch.sin(y)
```

### 4.2 指定编译模式

```python
# 默认模式（推荐）
model = torch.compile(model)

# reduce-overhead 模式：减少 Python 开销
model = torch.compile(model, mode="reduce-overhead")

# max-autotune 模式：最大性能优化（编译时间更长）
model = torch.compile(model, mode="max-autotune")
```

**模式对比：**

| 模式 | 编译时间 | 运行性能 | 适用场景 |
|------|---------|---------|---------|
| **default** | 适中 | 良好 | 日常开发、实验 |
| **reduce-overhead** | 较长 | 更好 | 生产环境、小批次 |
| **max-autotune** | 很长 | 最好 | 部署、大规模训练 |

### 4.3 指定后端

```python
# 使用默认后端 (TorchInductor)
model = torch.compile(model)

# 使用其他后端
model = torch.compile(model, backend="aot_eager")  # 仅 AOTAutograd
model = torch.compile(model, backend="onnxrt")     # ONNX Runtime
model = torch.compile(model, backend="tensorrt")   # TensorRT

# 自定义后端（第 8 章详解）
model = torch.compile(model, backend=my_custom_backend)
```

## 5. 性能优化的秘密

### 5.1 算子融合（Operator Fusion）

**未优化的代码：**
```python
def forward(x):
    x = x + 1      # Kernel 1
    x = x * 2      # Kernel 2
    x = x.relu()   # Kernel 3
    return x
```

每个操作都需要：
1. 从 GPU 读取数据
2. 计算
3. 写回 GPU

**优化后（融合）：**
```python
# 编译器生成融合的 kernel
def fused_forward(x):
    # 一次性完成：x = relu((x + 1) * 2)
    # 只需一次读取，一次写入
    return x
```

**性能提升：** 减少内存访问，提高计算密度

### 5.2 图优化

```python
# 原始代码
y = x + 0      # 无用操作
z = y * 1      # 无用操作
output = z

# 优化后
output = x     # 直接使用 x
```

**常见优化：**
- 常量折叠（Constant Folding）
- 死代码消除（Dead Code Elimination）
- 公共子表达式消除（CSE）
- 内存规划优化

### 5.3 自动调优

对于矩阵乘法等操作，编译器会：
1. 生成多个实现版本
2. 在实际硬件上测试
3. 选择最快的版本

## 6. 兼容性与限制

### 6.1 支持的特性 ✅

- 标准的 PyTorch 操作
- 自定义 nn.Module
- 控制流（if/for/while）
- 动态形状（有限支持）

### 6.2 已知限制 ⚠️

1. **首次运行慢**：第一次调用时需要编译
   ```python
   # 首次调用：慢（编译）
   output = model_compiled(input1)  # 可能需要几秒
   
   # 后续调用：快
   output = model_compiled(input2)  # 毫秒级
   ```

2. **动态形状**：输入形状变化会触发重新编译
   ```python
   model_compiled(torch.randn(32, 128))   # 编译 shape=(32, 128)
   model_compiled(torch.randn(64, 128))   # 重新编译 shape=(64, 128)
   ```

3. **不支持的 Python 特性**：
   - 某些动态 Python 特性（如 eval()）
   - 部分第三方库

### 6.3 调试技巧

```python
# 查看编译后的图
import torch._dynamo as dynamo

# 导出捕获的图
explanation = dynamo.explain(model)(input_data)
print(explanation)

# 调试模式：查看编译过程
import torch._dynamo
torch._dynamo.config.verbose = True

model = torch.compile(model)
```

## 7. 与其他技术的对比

### 7.1 vs TorchScript

| 特性 | torch.compile | TorchScript |
|------|--------------|-------------|
| **易用性** | 🌟🌟🌟🌟🌟 一行代码 | 🌟🌟 需要标注/trace |
| **Python 支持** | 🌟🌟🌟🌟 广泛支持 | 🌟🌟 有限支持 |
| **性能** | 🌟🌟🌟🌟🌟 更优 | 🌟🌟🌟 良好 |
| **调试** | 🌟🌟🌟🌟 较容易 | 🌟🌟 较困难 |

### 7.2 vs JAX

PyTorch 2.0 借鉴了 JAX 的很多思想，但更注重易用性和生态兼容性。

## 8. 实践建议

### 8.1 何时使用 torch.compile？

✅ **推荐场景：**
- 生产环境部署
- 训练大模型
- 推理服务
- 需要挤压最后一点性能时

❌ **不推荐场景：**
- 快速原型开发（首次编译开销）
- 调试代码
- 输入形状频繁变化

### 8.2 最佳实践

```python
# 1. 在训练开始前预热
model = torch.compile(model)
dummy_input = torch.randn(batch_size, ..., device=device)
_ = model(dummy_input)  # 触发编译

# 2. 固定输入形状
# 使用固定的 batch size，避免重新编译

# 3. 逐步应用
# 先编译性能瓶颈部分，而不是整个程序
```

## 9. 源码位置参考

在 `pytorch/torch/` 目录下：

```
torch/
├── _dynamo/              # Dynamo 核心实现
├── _inductor/            # Inductor 后端
├── fx/                   # TorchFX 框架
├── _functorch/           # AOTAutograd 实现
└── compile.py            # torch.compile 入口
```

## 10. 小结

### 核心要点

1. **PyTorch 2.0 = 动态图的灵活性 + 静态图的性能**
2. **torch.compile 是核心入口**，一行代码即可使用
3. **性能提升来自**：算子融合、图优化、自动调优
4. **技术栈**：Dynamo → AOTAutograd → 后端编译器
5. **注意首次编译开销**和动态形状问题

### 下一步

现在你已经了解了 PyTorch 2.0 的整体架构，接下来我们将深入每个组件：

- 📖 [第二章：TorchScript 详解](./02_torchscript.md) - 理解早期的静态图方案
- 📖 [第三章：TorchFX 图捕获技术](./03_torchfx.md) - 学习图表示和变换

---

## 💡 练习题

1. 用 `torch.compile` 编译你自己的一个模型，测试性能提升
2. 尝试不同的编译模式（default/reduce-overhead/max-autotune）
3. 观察不同输入形状下的编译行为

**准备好了吗？** 继续下一章 → [TorchScript 详解](./02_torchscript.md)

