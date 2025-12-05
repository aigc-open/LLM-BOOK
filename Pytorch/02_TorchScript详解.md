# 第二章：TorchScript 详解

## 🎯 本章目标

- 理解 TorchScript 的设计初衷
- 掌握 trace 和 script 两种模式
- 了解 TorchScript 的应用场景和局限性
- 认识 TorchScript 与 torch.compile 的关系

## 1. 什么是 TorchScript？

### 1.1 简单理解

想象你写了一个 Python 函数：

```python
def add(a, b):
    return a + b
```

这段代码：
- ✅ 在 Python 中运行
- ❌ 不能在 C++ 中直接运行
- ❌ 不能在移动设备上运行
- ❌ 无法进行深度优化

**TorchScript 的作用：** 把 Python 代码转换成一个**中间表示**（IR），这个 IR 可以：
- 脱离 Python 运行
- 在 C++ 中加载和执行
- 部署到移动端和嵌入式设备
- 进行各种优化

### 1.2 为什么需要 TorchScript？

**问题场景：**

```python
# 训练模型（Python）
model = MyModel()
model.train()
# ... 训练代码 ...

# 部署到生产环境？
# ❌ 不能要求生产环境安装 Python
# ❌ Python 性能不够
# ❌ Python GIL 限制并发
```

**TorchScript 解决方案：**

```python
# 1. Python 中转换
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")

# 2. C++ 中加载
// torch::jit::load("model.pt");
// 无需 Python！
```

## 2. TorchScript 的两种模式

### 2.1 模式一：Tracing（追踪模式）

**原理：** 运行一次代码，记录所有操作

```python
import torch

# 定义模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

# 创建模型和示例输入
model = SimpleModel()
example_input = torch.randn(1, 10)

# Trace 模式
traced_model = torch.jit.trace(model, example_input)

# 保存
traced_model.save("traced_model.pt")
```

**工作流程：**

```
输入数据 → 模型forward → 记录每一步操作 → 生成静态图
```

**示例：**

```python
def my_function(x, y):
    if x.sum() > 0:
        return x + y
    else:
        return x - y

# Trace
traced = torch.jit.trace(my_function, (torch.randn(3, 4), torch.randn(3, 4)))
```

**问题：** trace 只会记录**实际执行的路径**！

```python
# 假设 trace 时 x.sum() > 0 为 True
# 那么 traced 模型永远执行 x + y
# else 分支永远不会被记录！

print(traced.code)
# 输出类似：
# def forward(self, x, y):
#     return torch.add(x, y)  # 只有这一个分支！
```

### 2.2 模式二：Scripting（脚本模式）

**原理：** 分析 Python 源代码，转换为 TorchScript

```python
# Script 模式
@torch.jit.script
def my_function(x, y):
    if x.sum() > 0:
        return x + y
    else:
        return x - y

print(my_function.code)
# 输出：
# def forward(self, x, y):
#     if torch.gt(torch.sum(x), 0):
#         _0 = torch.add(x, y)
#     else:
#         _0 = torch.sub(x, y)
#     return _0
```

**两个分支都保留了！**

### 2.3 Trace vs Script 对比

| 特性 | Trace | Script |
|------|-------|--------|
| **使用方式** | `torch.jit.trace(model, example)` | `@torch.jit.script` 装饰器 |
| **原理** | 记录实际执行路径 | 分析 Python 代码 |
| **控制流** | ❌ 不支持（会固化） | ✅ 完全支持 |
| **易用性** | 🌟🌟🌟🌟🌟 简单 | 🌟🌟🌟 需要代码兼容 |
| **适用场景** | 简单模型、固定流程 | 复杂逻辑、动态控制流 |

## 3. 详细示例

### 3.1 Trace 示例：图像分类模型

```python
import torch
import torchvision.models as models

# 加载预训练模型
model = models.resnet18(pretrained=True)
model.eval()

# 准备示例输入
example_input = torch.randn(1, 3, 224, 224)

# Trace
traced_model = torch.jit.trace(model, example_input)

# 测试
with torch.no_grad():
    output1 = model(example_input)
    output2 = traced_model(example_input)
    print("输出差异:", (output1 - output2).abs().max().item())
    # 应该接近 0

# 保存
traced_model.save("resnet18_traced.pt")
```

### 3.2 Script 示例：带控制流的模型

```python
import torch
import torch.nn as nn

class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 5)
    
    def forward(self, x, use_second_layer: bool):
        x = self.linear1(x)
        
        # 动态控制流
        if use_second_layer:
            x = self.linear2(x)
        
        return x

# Script 模式（支持控制流）
model = DynamicModel()
scripted_model = torch.jit.script(model)

# 测试两个分支
input_data = torch.randn(2, 10)
output1 = scripted_model(input_data, True)   # 使用第二层
output2 = scripted_model(input_data, False)  # 不使用第二层

print("Output 1 shape:", output1.shape)  # torch.Size([2, 5])
print("Output 2 shape:", output2.shape)  # torch.Size([2, 10])
```

### 3.3 混合模式

```python
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        
        # 复杂的子模块用 script
        self.decision = torch.jit.script(DecisionModule())
    
    def forward(self, x):
        x = self.conv(x)
        # 调用 scripted 子模块
        x = self.decision(x)
        return x

# 整体用 trace
model = HybridModel()
traced_model = torch.jit.trace(model, torch.randn(1, 3, 32, 32))
```

## 4. TorchScript IR（中间表示）

### 4.1 查看 IR

```python
import torch

@torch.jit.script
def example(x, y):
    z = x + y
    return z * 2

# 查看 Python 风格的代码
print(example.code)

# 查看底层 IR
print(example.graph)
```

**输出示例：**

```
graph(%x : Tensor,
      %y : Tensor):
  %z : Tensor = aten::add(%x, %y, %1)
  %output : Tensor = aten::mul(%z, %2)
  return (%output)
```

### 4.2 IR 的结构

```python
# TorchScript IR 是一个图结构
# 节点（Node）：操作（如 add、mul）
# 边（Edge）：数据流

# 示例
def forward(self, x):
    a = x + 1      # Node 1: aten::add
    b = a * 2      # Node 2: aten::mul
    return b       # Node 3: return
```

**图表示：**

```
     x (输入)
      ↓
   aten::add (+ 1)
      ↓
     a
      ↓
   aten::mul (* 2)
      ↓
     b
      ↓
   return (输出)
```

## 5. 优化与性能

### 5.1 自动优化

TorchScript 会自动进行多种优化：

```python
# 原始代码
def unoptimized(x):
    a = x + 0      # 无用操作
    b = a * 1      # 无用操作
    c = b + b      # 可以优化为 b * 2
    return c

scripted = torch.jit.script(unoptimized)
print(scripted.graph)  # 查看优化后的图
```

**常见优化：**

1. **常量折叠**
   ```python
   # 优化前
   y = x * 2 * 3
   
   # 优化后
   y = x * 6  # 2 * 3 在编译时计算
   ```

2. **死代码消除**
   ```python
   # 优化前
   unused = x + 1
   result = x * 2
   return result
   
   # 优化后
   result = x * 2
   return result  # unused 被删除
   ```

3. **算子融合**
   ```python
   # 优化前
   y = x.relu()
   z = y * 2
   
   # 优化后
   z = fused_relu_mul(x, 2)  # 融合为一个 kernel
   ```

### 5.2 性能对比

```python
import time
import torch

# 原始模型
model = torch.nn.Sequential(
    torch.nn.Linear(1000, 1000),
    torch.nn.ReLU(),
    torch.nn.Linear(1000, 100)
)

# Scripted 模型
scripted_model = torch.jit.script(model)

# 基准测试
input_data = torch.randn(128, 1000)

# 预热
for _ in range(10):
    _ = model(input_data)
    _ = scripted_model(input_data)

# 测试
times_eager = []
times_script = []

for _ in range(100):
    start = time.time()
    _ = model(input_data)
    times_eager.append(time.time() - start)
    
    start = time.time()
    _ = scripted_model(input_data)
    times_script.append(time.time() - start)

print(f"Eager: {sum(times_eager)/len(times_eager)*1000:.2f}ms")
print(f"Script: {sum(times_script)/len(times_script)*1000:.2f}ms")
```

## 6. 部署到 C++

### 6.1 Python 端：保存模型

```python
import torch

model = MyModel()
model.eval()

# 方式 1: Script
scripted = torch.jit.script(model)
scripted.save("model_scripted.pt")

# 方式 2: Trace
traced = torch.jit.trace(model, example_input)
traced.save("model_traced.pt")
```

### 6.2 C++ 端：加载模型

```cpp
#include <torch/script.h>
#include <iostream>

int main() {
    // 加载模型
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("model_scripted.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading model\n";
        return -1;
    }
    
    // 准备输入
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({1, 3, 224, 224}));
    
    // 执行推理
    at::Tensor output = model.forward(inputs).toTensor();
    
    std::cout << "Output shape: " << output.sizes() << std::endl;
    
    return 0;
}
```

### 6.3 编译 C++ 程序

```bash
# CMakeLists.txt
cmake_minimum_required(VERSION 3.0)
project(inference)

find_package(Torch REQUIRED)

add_executable(inference main.cpp)
target_link_libraries(inference "${TORCH_LIBRARIES}")

# 编译
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make
```

## 7. 局限性与问题

### 7.1 不支持的 Python 特性

```python
# ❌ 不支持：动态类型
@torch.jit.script
def bad_example(x):
    if random.random() > 0.5:
        return x.sum()  # 返回标量
    else:
        return x        # 返回 tensor
    # 类型不一致！

# ✅ 正确：固定类型
@torch.jit.script
def good_example(x):
    if x.sum() > 0:
        return x + 1    # 都返回 tensor
    else:
        return x - 1
```

### 7.2 不支持的库

```python
# ❌ 不支持：numpy
@torch.jit.script
def use_numpy(x):
    import numpy as np  # 错误！
    return torch.from_numpy(np.array([1, 2, 3]))

# ✅ 使用纯 PyTorch
@torch.jit.script
def use_torch(x):
    return torch.tensor([1, 2, 3])
```

### 7.3 调试困难

```python
# 错误信息可能不清晰
@torch.jit.script
def buggy(x):
    return x.unknown_method()  # 运行时错误

# 建议：先在 eager mode 测试
```

## 8. TorchScript vs PyTorch 2.0

### 8.1 对比

| 维度 | TorchScript | torch.compile (PyTorch 2.0) |
|------|------------|------------------------------|
| **发布时间** | 2018 | 2023 |
| **易用性** | 中等 | 极简 |
| **Python 支持** | 有限 | 广泛 |
| **性能** | 良好 | 更优 |
| **部署** | 优秀（C++） | 有限（主要 Python） |
| **开发体验** | 需要适配代码 | 几乎无侵入 |

### 8.2 何时使用 TorchScript？

✅ **适合场景：**
- 需要 C++ 部署
- 移动端/嵌入式部署
- 已有 TorchScript 代码（兼容性）

❌ **不推荐场景：**
- 纯 Python 训练（用 torch.compile）
- 复杂动态逻辑（用 PyTorch 2.0）

### 8.3 torch.compile 可以使用 TorchScript 后端

```python
# PyTorch 2.0 中，TorchScript 成为一个可选后端
model = torch.compile(model, backend="torchscript")
```

## 9. 实际应用案例

### 9.1 移动端部署

```python
# 1. 训练模型
model = MobileNetV2()
# ... 训练 ...

# 2. 转换为 TorchScript
model.eval()
scripted = torch.jit.script(model)

# 3. 优化（移动端）
from torch.utils.mobile_optimizer import optimize_for_mobile
optimized_model = optimize_for_mobile(scripted)

# 4. 保存
optimized_model._save_for_lite_interpreter("model_mobile.ptl")
```

### 9.2 服务器端推理

```python
# server.py
import torch
from flask import Flask, request

app = Flask(__name__)

# 加载 TorchScript 模型
model = torch.jit.load("model.pt")
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_tensor = torch.tensor(data['input'])
    
    with torch.no_grad():
        output = model(input_tensor)
    
    return {'output': output.tolist()}

if __name__ == '__main__':
    app.run()
```

## 10. 源码位置

在 `pytorch/` 目录下：

```
torch/
├── jit/
│   ├── __init__.py          # torch.jit 入口
│   ├── _trace.py            # Trace 实现
│   ├── _script.py           # Script 实现
│   ├── frontend/            # Python → IR 转换
│   └── mobile/              # 移动端优化
├── csrc/jit/                # C++ 实现
│   ├── api/                 # C++ API
│   ├── passes/              # IR 优化 passes
│   ├── runtime/             # 运行时
│   └── serialization/       # 序列化
```

**关键文件：**
- `torch/jit/_trace.py`：Trace 逻辑
- `torch/jit/_script.py`：Script 逻辑
- `torch/csrc/jit/frontend/tracer.cpp`：Trace 的 C++ 实现
- `torch/csrc/jit/passes/`：各种优化 pass

## 11. 小结

### 核心要点

1. **TorchScript = Python 模型转为可部署的中间表示**
2. **两种模式：**
   - Trace：简单但不支持控制流
   - Script：完整但对代码有要求
3. **主要用途：C++ 部署、移动端部署**
4. **PyTorch 2.0 时代：** torch.compile 更优，TorchScript 作为部署选项

### 关键技巧

```python
# 1. 选择合适的模式
# 简单模型 → trace
# 复杂逻辑 → script

# 2. 验证转换结果
torch.testing.assert_close(
    model(input),
    scripted_model(input)
)

# 3. 查看优化效果
print(scripted_model.graph)
```

### 下一步

现在你理解了 TorchScript 这个"老前辈"，接下来我们学习 PyTorch 2.0 的核心技术：

📖 [第三章：TorchFX 图捕获技术](./03_torchfx.md) - 更灵活的图表示和变换工具

---

## 💡 练习题

1. 用 trace 和 script 两种方式转换一个简单模型，对比结果
2. 编写一个包含 if-else 的模型，观察 trace 的问题
3. 尝试在 C++ 中加载一个 TorchScript 模型

**继续学习** → [TorchFX 图捕获技术](./03_torchfx.md)

