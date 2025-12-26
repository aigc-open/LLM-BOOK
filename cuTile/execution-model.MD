# cuTile Python 执行模型详解

## 目录
1. [概述](#概述)
2. [抽象机器模型](#抽象机器模型)
3. [执行空间](#执行空间)
4. [Tile 函数](#tile-函数)
5. [Tile 内核](#tile-内核)
6. [Python 子集](#python-子集)
7. [Tile 并行性](#tile-并行性)
8. [常量性](#常量性)
9. [实战示例](#实战示例)

---

## 概述

cuTile Python 的执行模型定义了代码如何在 GPU 上执行。理解执行模型对于编写高效的 GPU 程序至关重要。

**核心概念：**
- **Block（块）**: 执行单元，多个线程的集合
- **Tile（数据块）**: 数据单元，内存中的数据片段
- **Grid（网格）**: Block 的组织方式（1D/2D/3D）
- **Thread（线程）**: 最小执行单位（在 Tile 编程中是隐式的）

**重要区分：**
> **Block ≠ Tile**
> - **Block** 是**执行单元**，负责运行代码
> - **Tile** 是**数据单元**，是要处理的数据
> - 一个 Block 可以处理多个不同形状的 Tile

---

## 抽象机器模型

### Block 与 Grid

cuTile 内核由一个网格（Grid）中的多个逻辑线程块（Block）执行。

```
Grid 组织方式：

1D Grid:     [Block₀][Block₁][Block₂]...[Blockₙ]

2D Grid:     Block₀₀  Block₀₁  Block₀₂
             Block₁₀  Block₁₁  Block₁₂
             Block₂₀  Block₂₁  Block₂₂

3D Grid:     多层 2D 网格堆叠
```

### Block 的执行机制

**1. 并行执行**
```python
# 每个 Block 独立执行内核代码
grid = (8, 1, 1)  # 8 个 Block 并行执行
ct.launch(stream, grid, kernel, args)
```

**2. 标量操作 vs 数组操作**

```python
@ct.kernel
def example_kernel(a, b, c):
    # 标量操作：由 Block 中的单个线程串行执行
    pid = ct.bid(0)  # 标量：Block ID
    offset = pid * 16  # 标量：计算偏移量
    
    # 数组操作：由 Block 中的所有线程并行执行
    a_tile = ct.load(a, index=(pid,), shape=(16,))  # Tile：16 个元素
    b_tile = ct.load(b, index=(pid,), shape=(16,))  # Tile：16 个元素
    result = a_tile + b_tile  # Tile 操作：所有元素并行相加
```

**操作类型对比：**

| 操作类型 | 执行方式 | 示例 |
|---------|---------|------|
| **标量操作** | 单线程串行 | `pid = ct.bid(0)`, `x = 5 + 3` |
| **数组操作** | 多线程并行 | `result = a_tile + b_tile` |

### 显式并行 vs 隐式并行

**cuTile 的特点：**
- **显式描述 Block 级并行**：程序员指定有多少个 Block
- **隐式处理 Thread 级并行**：线程由系统自动管理
- **不能直接操作线程**：无法识别或控制单个线程

```python
# 显式指定 Block 数量
grid = (256, 1, 1)  # 256 个 Block

# 无法指定线程数量（由系统决定）
# 无 threadIdx, blockDim 等概念
```

### Block 间同步与通信

**规则：**
- **Block 内不允许显式同步/通信**（由系统隐式处理）
- **Block 间允许同步/通信**（通过全局内存）

```python
# Block 内无需显式同步（自动处理）
# 不需要 __syncthreads() 这样的操作

# Block 间可以通过全局内存通信
@ct.kernel
def cross_block_comm(data, flags):
    pid = ct.bid(0)
    # Block 间通过全局内存交换数据
    # 但需要使用原子操作或 barrier 确保一致性
```

---

## 执行空间

执行空间（Execution Space）定义了代码可以在哪些环境中运行。

### 三种执行空间

```
┌─────────────────────────────────────────┐
│          Host Code (CPU)                │  ← Python 标准代码
│  - 可以调用 @ct.kernel                  │
│  - 可以调用 host=True 的函数            │
└─────────────────────────────────────────┘
                   │
                   │ ct.launch()
                   ↓
┌─────────────────────────────────────────┐
│         Tile Code (GPU)                 │  ← cuTile 内核代码
│  - 使用 Tile 抽象                       │
│  - 不能直接访问线程                     │
│  - 可以调用 @ct.function                │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│        SIMT Code (GPU)                  │  ← 传统 CUDA 代码
│  - 直接操作线程（threadIdx 等）         │
│  - 与 Tile Code 是不同的执行空间         │
└─────────────────────────────────────────┘
```

### 可用性（Usability）

**定义：**
- **函数可用**：可以被调用
- **类型/对象可用**：属性可以访问，方法可以调用

**执行空间限制：**
某些函数、类型和对象只能在特定的执行空间中使用。

### Host Code（主机代码）

```python
import cupy as cp
import cuda.tile as ct

# Host Code：在 CPU 上执行
def main():
    a = cp.random.rand(1024)  # CuPy 在 Host 可用
    b = cp.zeros(1024)
    
    grid = (64, 1, 1)
    ct.launch(stream, grid, my_kernel, (a, b))  # 启动内核
```

### Tile Code（Tile 代码）

```python
@ct.kernel
def my_kernel(a, b):
    # Tile Code：在 GPU 上执行
    pid = ct.bid(0)  # ct.bid() 在 Tile Code 可用
    
    tile_a = ct.load(a, index=(pid,), shape=(16,))  # ct.load() 可用
    result = tile_a * 2.0  # Tile 操作可用
    
    ct.store(b, index=(pid,), tile=result)  # ct.store() 可用
    
    # 注意：print() 不在 Tile Code 可用（无 Python 运行时）
    # 注意：cp.random.rand() 不在 Tile Code 可用
```

### SIMT Code（SIMT 代码）

传统 CUDA 编程模型，与 Tile Code 是不同的执行空间：

```cuda
// SIMT Code：传统 CUDA 内核
__global__ void simt_kernel(float* a, float* b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // 直接访问线程索引
    if (idx < n) {
        b[idx] = a[idx] * 2.0f;
    }
}
```

---

## Tile 函数

### 基本概念

Tile 函数是可以在 Tile Code 中调用的函数。

### 装饰器：`@ct.function`

```python
class cuda.tile.function(func=None, /, *, host=False, tile=True)
```

**参数：**
- `host` (bool): 是否可以从 Host Code 调用，默认 `False`
- `tile` (bool): 是否可以从 Tile Code 调用，默认 `True`

### 使用示例

#### 示例 1：仅在 Tile Code 可用

```python
import cuda.tile as ct

@ct.function  # 等价于 @ct.function(host=False, tile=True)
def compute_square(x):
    """仅能在 Tile Code 中调用"""
    return x * x

@ct.kernel
def kernel_example(a, b):
    pid = ct.bid(0)
    tile_a = ct.load(a, index=(pid,), shape=(16,))
    
    # 在 Tile Code 中调用
    result = compute_square(tile_a)
    
    ct.store(b, index=(pid,), tile=result)

# 注意：不能在 Host Code 中调用
# x = compute_square(5)  # 错误！
```

#### 示例 2：Host 和 Tile 都可用

```python
@ct.function(host=True, tile=True)
def add_constant(x, c):
    """在 Host 和 Tile Code 都可以调用"""
    return x + c

# Host Code 中调用
result = add_constant(10, 5)  # 返回 15

@ct.kernel
def kernel_with_shared_func(a, b):
    pid = ct.bid(0)
    tile_a = ct.load(a, index=(pid,), shape=(16,))
    
    # Tile Code 中调用
    result = add_constant(tile_a, 10.0)
    
    ct.store(b, index=(pid,), tile=result)
```

### 自动推断执行空间

**重要特性：**
> 当一个**未标注的函数**被 Tile 函数调用时，它会自动被添加到 Tile 执行空间。这个过程是递归的，不需要显式标注。

#### 示例 1：基础自动推断

```python
# 未使用 @ct.function 装饰器
def helper_add(a, b):
    return a + b

def helper_multiply(a, b):
    # 调用另一个未标注的函数
    return helper_add(a, b) * 2

@ct.function
def compute(x):
    # 调用未标注的函数会自动将其加入 Tile 执行空间
    temp = helper_multiply(x, 5)  # helper_multiply 自动可用
    return temp * temp  # helper_add 也自动可用（递归）

@ct.kernel
def my_kernel(a, b):
    pid = ct.bid(0)
    tile_a = ct.load(a, index=(pid,), shape=(16,))
    result = compute(tile_a)  # 所有函数都可用
    ct.store(b, index=(pid,), tile=result)
```

#### 示例 2：神经网络激活函数 - ReLU

```python
import cuda.tile as ct

# 方式 A：自动推断（推荐用于简单工具函数）
def relu(x):
    """ReLU 激活函数：max(0, x) - 未使用装饰器"""
    return ct.maximum(x, 0.0)

def relu_derivative(x):
    """ReLU 导数：x > 0 ? 1 : 0 - 也会被自动推断"""
    return ct.where(x > 0, 1.0, 0.0)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU - 调用其他未标注函数"""
    positive = relu(x)  # relu 被递归推断
    negative = x * alpha
    return ct.where(x > 0, positive, negative)

@ct.function  # 只需标注入口函数
def activation_pipeline(x):
    # 所有被调用的函数都会被递归自动推断
    # relu → relu_derivative → leaky_relu 全部自动可用
    return leaky_relu(x)

@ct.kernel
def apply_activation_kernel(input_arr, output_arr):
    pid = ct.bid(0)
    
    # 加载数据
    tile_in = ct.load(input_arr, index=(pid,), shape=(16,))
    
    # 激活函数自动可用，无需显式标注
    tile_out = activation_pipeline(tile_in)
    
    # 存储结果
    ct.store(output_arr, index=(pid,), tile=tile_out)

# 方式 B：显式标注（推荐用于核心逻辑函数）
@ct.function
def relu_explicit(x):
    """显式标注的 ReLU - 提高代码可读性"""
    return ct.maximum(x, 0.0)

@ct.function
def gelu(x):
    """GELU 激活函数 - 更复杂的逻辑建议显式标注"""
    return 0.5 * x * (1.0 + ct.tanh(0.797885 * (x + 0.044715 * x * x * x)))
```

#### 对比：三种写法

```python
# 写法 1：完全自动推断（简洁，适合简单函数）
def clip_min(x, min_val):
    return ct.maximum(x, min_val)

def relu_v1(x):
    return clip_min(x, 0.0)  # clip_min 自动推断

# 写法 2：显式标注（清晰，适合核心函数）
@ct.function
def clip_min_v2(x, min_val):
    return ct.maximum(x, min_val)

@ct.function
def relu_v2(x):
    return clip_min_v2(x, 0.0)

# 写法 3：混合模式（灵活，推荐）
def clip_min_v3(x, min_val):
    """简单工具函数，不标注"""
    return ct.maximum(x, min_val)

@ct.function
def relu_v3(x):
    """核心逻辑函数，显式标注"""
    return clip_min_v3(x, 0.0)  # clip_min_v3 自动推断
```

**最佳实践：**
- 对于明确只在 Tile Code 使用的**核心逻辑函数**，显式使用 `@ct.function` 标注
- 对于**简单数学工具函数**（如 `relu`、`clip` 等），可以不标注，让系统自动推断
- 对于需要在 Host 和 Tile 共享的函数，使用 `@ct.function(host=True, tile=True)`
- 递归调用链中只需标注入口函数，其他函数会被自动推断

### 参数类型

Tile 函数的参数类型在[数据模型](data-model.MD)中详细描述。

**支持的参数类型：**
- 标量类型：`int`, `float`, `bool`
- Tile 类型：从 `ct.load()` 返回的数组
- 常量类型：`ct.Constant[T]`

---

## Tile 内核

### 基本概念

Tile 内核是由网格中每个 Block 执行的函数，是 Tile Code 的入口点。

### 装饰器：`@ct.kernel`

```python
class cuda.tile.kernel(function=None, /, **kwargs)
```

**参数：**
- `num_ctas`: CGA 中的 CTA 数量（必须是 1-16 之间的 2 的幂），默认：`None`（自动）
- `occupancy`: 每个 SM 的活跃 CTA 预期数量 [1, 32]，默认：`None`（自动）
- `opt_level`: 优化级别 [0, 3]，默认：`3`

**特性：**
- 内核是 Tile Code 的入口点
- 内核**不能**从 Host Code 直接调用
- 必须使用 `ct.launch()` 启动内核

### 基本示例

```python
import cuda.tile as ct
import cupy as cp

@ct.kernel
def simple_kernel(a, b, c):
    """
    简单的向量加法内核
    
    每个 Block 处理一个 Tile
    """
    # 获取 Block ID
    pid = ct.bid(0)
    
    # 加载数据
    tile_a = ct.load(a, index=(pid,), shape=(16,))
    tile_b = ct.load(b, index=(pid,), shape=(16,))
    
    # 计算
    tile_c = tile_a + tile_b
    
    # 存储结果
    ct.store(c, index=(pid,), tile=tile_c)

# Host Code：启动内核
def main():
    n = 1024
    a = cp.random.rand(n).astype(cp.float32)
    b = cp.random.rand(n).astype(cp.float32)
    c = cp.zeros(n, dtype=cp.float32)
    
    tile_size = 16
    num_blocks = n // tile_size  # 64 个 Block
    grid = (num_blocks, 1, 1)
    
    stream = cp.cuda.get_current_stream()
    
    # 使用 ct.launch() 启动内核
    ct.launch(stream, grid, simple_kernel, (a, b, c))
    
    # 注意：不能直接调用
    # simple_kernel(a, b, c)  # 错误！
```

### 多维 Grid 示例

```python
@ct.kernel
def matrix_kernel(matrix, output):
    """
    2D Grid 处理矩阵
    """
    # 获取 2D Block 索引
    row = ct.bid(0)  # Y 维度
    col = ct.bid(1)  # X 维度
    
    # 加载 2D Tile
    tile = ct.load(matrix, index=(row, col), shape=(16, 16))
    
    # 处理
    result = tile * 2.0
    
    # 存储
    ct.store(output, index=(row, col), tile=result)

# 启动 2D Grid
def main():
    matrix = cp.random.rand(256, 256).astype(cp.float32)
    output = cp.zeros_like(matrix)
    
    tile_h, tile_w = 16, 16
    grid_y = 256 // tile_h  # 16
    grid_x = 256 // tile_w  # 16
    grid = (grid_y, grid_x, 1)  # 2D Grid: 16×16
    
    stream = cp.cuda.get_current_stream()
    ct.launch(stream, grid, matrix_kernel, (matrix, output))
```

### 高级参数配置

```python
from cuda.tile import ByTarget

@ct.kernel(
    num_ctas=4,      # 每个 CGA 4 个 CTA
    occupancy=16,    # 每个 SM 16 个活跃 CTA
    opt_level=3      # 最高优化级别
)
def optimized_kernel(a, b):
    pid = ct.bid(0)
    tile_a = ct.load(a, index=(pid,), shape=(32,))
    result = tile_a * 2.0
    ct.store(b, index=(pid,), tile=result)

# 针对不同目标的配置
@ct.kernel(
    num_ctas=ByTarget(sm_90=8, sm_80=4),  # 不同架构不同配置
    occupancy=ByTarget(sm_90=32, sm_80=16)
)
def target_specific_kernel(a, b):
    # 内核代码
    pass
```

### 启动内核：`ct.launch()`

```python
cuda.tile.launch(stream, grid, kernel, kernel_args, /)
```

**参数：**
- `stream`: CUDA 流对象
- `grid`: 最多 3 个维度的元组 `(dim_x, dim_y, dim_z)`
- `kernel`: 要执行的内核函数
- `kernel_args`: 传递给内核的位置参数元组

**示例：**

```python
# 1D Grid
grid = (256, 1, 1)
ct.launch(stream, grid, kernel_1d, (a, b, c))

# 2D Grid
grid = (16, 16, 1)
ct.launch(stream, grid, kernel_2d, (matrix, output))

# 3D Grid
grid = (8, 8, 8)
ct.launch(stream, grid, kernel_3d, (volume, result))
```

---

## Python 子集

### 核心限制

**重要：** Tile Code 中没有 Python 运行时！

> **只有本文档中明确列举的 Python 特性才被支持。**

**不支持的特性（部分）：**
- Lambda 表达式
- 异常处理（try/except）
- 协程（async/await）
- 动态类型特性
- 反射/元编程
- 文件 I/O
- 大部分标准库

### 对象模型与生命周期

#### 不可变性规则

**核心规则：**
> 在 Tile Code 中创建的所有对象都是**不可变的**。

```python
@ct.kernel
def immutable_demo(a, b):
    pid = ct.bid(0)
    
    # 创建新对象
    x = 10
    
    # 注意：不能修改 x（概念上创建新对象）
    x = x + 5  # 这会创建新的对象，而不是修改原对象
    
    # Tile 操作创建新 Tile
    tile_a = ct.load(a, index=(pid,), shape=(16,))
    tile_b = tile_a * 2  # 创建新 Tile，不修改 tile_a
    
    # 注意：不能动态添加属性
    # tile_a.new_attr = 5  # 错误！
```

#### 可变对象：数组

**唯一例外：** 作为内核参数传入的**数组**是可变的。

```python
@ct.kernel
def array_mutation(a, b):
    pid = ct.bid(0)
    
    # 可以修改数组内容（通过 ct.store()）
    tile = ct.load(a, index=(pid,), shape=(16,))
    result = tile * 2
    ct.store(a, index=(pid,), tile=result)  # 修改数组 a
    ct.store(b, index=(pid,), tile=result)  # 修改数组 b
```

**调用者责任：**

调用者必须确保：
1. **传递给内核的数组不能互相别名（alias）**
2. **所有数组在内核执行完成前保持有效**

```python
# 错误示例：数组别名
a = cp.arange(100)
ct.launch(stream, grid, kernel, (a, a))  # 危险！a 传递两次

# 正确示例：不同数组
a = cp.arange(100)
b = cp.zeros(100)
ct.launch(stream, grid, kernel, (a, b))

# 正确示例：确保生命周期
def safe_launch():
    a = cp.arange(100)
    b = cp.zeros(100)
    stream = cp.cuda.get_current_stream()
    ct.launch(stream, grid, kernel, (a, b))
    stream.synchronize()  # 等待内核完成
    # a, b 在内核完成后才会被释放
```

### 控制流

#### 支持的语句

cuTile 支持 Python 的控制流语句：

**1. if 语句**

```python
@ct.kernel
def conditional_kernel(a, b, threshold: ct.Constant[float]):
    pid = ct.bid(0)
    tile_a = ct.load(a, index=(pid,), shape=(16,))
    
    # if-else 语句
    if pid < 10:
        result = tile_a * 2.0
    else:
        result = tile_a * 0.5
    
    ct.store(b, index=(pid,), tile=result)
```

**2. for 循环**

```python
@ct.kernel
def loop_kernel(a, b, iterations: ct.Constant[int]):
    pid = ct.bid(0)
    tile_a = ct.load(a, index=(pid,), shape=(16,))
    
    result = tile_a
    # for range 循环
    for i in range(iterations):
        result = result * 1.1
    
    ct.store(b, index=(pid,), tile=result)
```

**3. while 循环**

```python
@ct.kernel
def while_kernel(a, b, max_iter: ct.Constant[int]):
    pid = ct.bid(0)
    tile_a = ct.load(a, index=(pid,), shape=(16,))
    
    result = tile_a
    count = 0
    
    # while 循环
    while count < max_iter:
        result = result * 0.99
        count = count + 1
    
    ct.store(b, index=(pid,), tile=result)
```

**4. 嵌套控制流**

```python
@ct.kernel
def nested_control(a, b, c):
    pid = ct.bid(0)
    tile_a = ct.load(a, index=(pid,), shape=(16,))
    
    # 任意嵌套
    if pid % 2 == 0:
        for i in range(5):
            if i > 2:
                tile_a = tile_a * 1.5
            else:
                tile_a = tile_a * 0.5
    
    ct.store(b, index=(pid,), tile=tile_a)
```

#### 当前限制

**注意：range() 的 step 必须严格为正**

```python
# 正向循环（step > 0）
for i in range(10):  # step 默认为 1
    pass

for i in range(0, 20, 2):  # step = 2
    pass

# 不支持负步长
for i in range(10, 0, -1):  # 错误！不支持
    pass

# 注意：通过变量传递负步长会导致未定义行为
step = -1
for i in range(10, 0, step):  # 未定义行为！
    pass
```

**解决方案：** 使用正向循环并调整索引

```python
# 需要反向遍历时，手动计算索引
n = 10
for i in range(n):
    actual_index = n - 1 - i  # 从 9 到 0
    # 使用 actual_index
```

---

## Tile 并行性

### 并行执行模型

当 Block 执行一个接受 Tile 作为参数的函数时，它可以跨 Block 的执行资源并行化函数求值。

```python
@ct.function
def process_tile(tile):
    """
    这个函数接受 Tile 作为参数
    Block 会自动并行化 Tile 上的操作
    """
    return tile * 2.0 + 1.0

@ct.kernel
def parallel_kernel(a, b):
    pid = ct.bid(0)
    tile_a = ct.load(a, index=(pid,), shape=(128,))
    
    # Tile 操作会被自动并行化
    # 128 个元素的计算会分配给多个线程并行执行
    result = process_tile(tile_a)
    
    ct.store(b, index=(pid,), tile=result)
```

**并行执行示意：**

```
Tile: [e₀, e₁, e₂, e₃, e₄, e₅, e₆, e₇, ...]
      │   │   │   │   │   │   │   │
      ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
    Thread₀ Thread₁ Thread₂ Thread₃ ...
      │   │   │   │   │   │   │   │
      ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
    [r₀, r₁, r₂, r₃, r₄, r₅, r₆, r₇, ...]

所有线程并行执行相同的操作，但处理不同的元素
```

### 同步保证

**重要保证：**
> 除非另有说明，函数执行将在函数返回前完成。

```python
@ct.kernel
def sync_guaranteed(a, b, c):
    pid = ct.bid(0)
    
    # 加载完成后才继续
    tile_a = ct.load(a, index=(pid,), shape=(16,))
    
    # 计算完成后才继续
    result = tile_a * 2.0
    
    # 存储完成后才继续
    ct.store(b, index=(pid,), tile=result)
    
    # 所有上述操作都保证完成后，才会到达这里
    # 不需要显式的 __syncthreads()
```

**对比传统 CUDA：**

```cuda
// 传统 CUDA：需要显式同步
__global__ void cuda_kernel(float* a, float* b) {
    __shared__ float temp[16];
    int tid = threadIdx.x;
    
    temp[tid] = a[tid];
    __syncthreads();  // ❗ 显式同步
    
    b[tid] = temp[tid] * 2.0f;
}
```

```python
# cuTile：自动同步
@ct.kernel
def cutile_kernel(a, b):
    pid = ct.bid(0)
    tile = ct.load(a, index=(pid,), shape=(16,))
    # 自动同步，不需要显式调用
    result = tile * 2.0
    ct.store(b, index=(pid,), tile=result)
```

---

## 常量性

### 常量表达式与对象

某些功能需要在**编译时静态已知**的参数，这些参数必须是**常量对象**。

### 什么是常量表达式？

**常量表达式包括：**

1. **字面量对象**
   ```python
   x = 42        # 整数字面量
   y = 3.14      # 浮点字面量
   z = True      # 布尔字面量
   ```

2. **整数算术表达式（操作数都是字面量）**
   ```python
   a = 10 + 20           # 常量表达式 → 30
   b = 5 * 6             # 常量表达式 → 30
   c = (100 - 10) // 3   # 常量表达式 → 30
   ```

3. **从字面量或常量表达式赋值的局部对象/参数**
   ```python
   tile_size = 16           # 常量
   double_size = tile_size * 2  # 也是常量 → 32
   ```

4. **编译或启动时定义的全局对象**
   ```python
   GLOBAL_SIZE = 128  # 全局常量
   
   @ct.kernel
   def kernel(a):
       # GLOBAL_SIZE 是常量
       tile = ct.load(a, index=(0,), shape=(GLOBAL_SIZE,))
   ```

### 松散类型 vs 严格类型常量

#### 松散类型常量（默认）

```python
# 默认情况下，常量是松散类型的
x = 42        # 无限精度整数
y = 3.14      # IEEE 754 双精度

# 松散类型常量直到需要时才确定具体类型
result = x + 100  # 仍然是松散类型
```

#### 严格类型常量

```python
import cuda.tile as ct

# 创建严格类型常量
x = ct.int16(5)      # 严格的 int16 类型
y = ct.float32(3.14) # 严格的 float32 类型

# 严格类型 + 松散类型 → 严格类型
z = ct.int16(5) + 2  # 结果：int16(7)

# 严格类型 + 严格类型 → 类型提升
a = ct.int16(5) + ct.int32(7)  # 结果：int32(12)
```

**类型提升规则：**

```
int8 → int16 → int32 → int64
float16 → float32 → float64
```

### 常量嵌入（Constant Embedding）

#### 什么是常量嵌入？

如果内核参数被标记为**常量嵌入**，则：

1. 参数的所有使用都像被字面值替换一样
2. 每个不同的参数值都会生成独立的内核机器码
3. 参数的机器表示为 0 字节（不占用寄存器/内存）

**优点：**
- 更激进的编译器优化
- 更少的寄存器使用
- 可能展开循环和内联操作

**缺点：**
- 每个不同值都会编译一次（增加编译时间）
- 多个值会增加代码缓存大小

### 常量类型提示

#### 使用 `ct.Constant`

```python
import cuda.tile as ct
from typing import Annotated

# 方式 1：不指定类型
@ct.kernel
def kernel1(a, b, tile_size: ct.Constant):
    """tile_size 必须是常量，任意类型"""
    tile_a = ct.load(a, index=(0,), shape=(tile_size,))
    ct.store(b, index=(0,), tile=tile_a)

# 方式 2：指定类型
@ct.kernel
def kernel2(a, b, tile_size: ct.Constant[int]):
    """tile_size 必须是常量 int 类型"""
    tile_a = ct.load(a, index=(0,), shape=(tile_size,))
    ct.store(b, index=(0,), tile=tile_a)
```

#### 完整示例

```python
import cupy as cp
import cuda.tile as ct

@ct.kernel
def constant_embedded_kernel(
    a, b, 
    tile_size: ct.Constant[int],      # 常量嵌入
    scale_factor: ct.Constant[float]  # 常量嵌入
):
    """
    tile_size 和 scale_factor 会被常量嵌入
    """
    pid = ct.bid(0)
    
    # tile_size 在编译时已知，可以进行更多优化
    tile_a = ct.load(a, index=(pid,), shape=(tile_size,))
    
    # scale_factor 也在编译时已知
    result = tile_a * scale_factor
    
    ct.store(b, index=(pid,), tile=result)

# 使用示例
def main():
    n = 1024
    a = cp.random.rand(n).astype(cp.float32)
    b = cp.zeros(n, dtype=cp.float32)
    
    tile_size = 16       # 常量值
    scale_factor = 2.5   # 常量值
    
    grid = (n // tile_size, 1, 1)
    stream = cp.cuda.get_current_stream()
    
    # 第一次调用：tile_size=16, scale_factor=2.5
    ct.launch(stream, grid, constant_embedded_kernel, 
              (a, b, tile_size, scale_factor))
    
    # 第二次调用：tile_size=32（不同的值）
    # 会触发重新编译！
    tile_size2 = 32
    grid2 = (n // tile_size2, 1, 1)
    ct.launch(stream, grid2, constant_embedded_kernel,
              (a, b, tile_size2, scale_factor))
```

### `ct.ConstantAnnotation` 类

```python
from typing import Annotated
import cuda.tile as ct

# 显式使用 ConstantAnnotation
def custom_function(x: Annotated[int, ct.ConstantAnnotation()]):
    """x 必须是常量嵌入的 int"""
    return x * 2

# 等价于
def custom_function2(x: ct.Constant[int]):
    """使用便捷的 ct.Constant 别名"""
    return x * 2
```

### 常量性最佳实践

**推荐使用常量嵌入的场景：**

1. **Tile 形状参数**
   ```python
   @ct.kernel
   def kernel(a, b, tile_size: ct.Constant[int]):
       tile = ct.load(a, index=(0,), shape=(tile_size,))  # 形状需要是常量
   ```

2. **循环边界（需要展开时）**
   ```python
   @ct.kernel
   def unrolled_kernel(a, b, iterations: ct.Constant[int]):
       result = a
       for i in range(iterations):  # 编译器可能展开循环
           result = result * 1.1
       ct.store(b, index=(0,), tile=result)
   ```

3. **配置参数**
   ```python
   @ct.kernel
   def configured_kernel(a, b, 
                         use_fast_math: ct.Constant[bool],
                         precision: ct.Constant[int]):
       tile = ct.load(a, index=(0,), shape=(16,))
       if use_fast_math:
           result = fast_compute(tile)
       else:
           result = accurate_compute(tile)
   ```

**不推荐使用常量嵌入的场景：**

1. **频繁变化的参数**（会导致大量重新编译）
2. **大范围的参数值**（会生成大量不同的内核版本）
3. **运行时才知道的数据**

---

## 实战示例

### 示例 1：完整的矩阵转置

```python
import cupy as cp
import cuda.tile as ct
import numpy as np

@ct.function
def transpose_tile(tile):
    """
    转置一个 2D Tile
    这个函数会被自动加入 Tile 执行空间
    """
    # 假设 ct 提供了 transpose 操作
    return ct.transpose(tile)

@ct.kernel
def matrix_transpose(input_matrix, output_matrix, 
                     tile_h: ct.Constant[int], 
                     tile_w: ct.Constant[int]):
    """
    矩阵转置内核
    
    输入矩阵形状: (H, W)
    输出矩阵形状: (W, H)
    Tile 形状: (tile_h, tile_w)
    """
    # 获取 2D Block 索引
    row = ct.bid(0)
    col = ct.bid(1)
    
    # 加载输入 Tile: (tile_h, tile_w)
    input_tile = ct.load(input_matrix, 
                         index=(row, col), 
                         shape=(tile_h, tile_w))
    
    # 转置 Tile: (tile_h, tile_w) → (tile_w, tile_h)
    transposed = transpose_tile(input_tile)
    
    # 存储到转置位置
    ct.store(output_matrix, 
             index=(col, row),  # 注意：行列互换
             tile=transposed)

def test_matrix_transpose():
    """测试矩阵转置"""
    # 创建测试数据
    H, W = 256, 128
    tile_h, tile_w = 16, 16
    
    input_matrix = cp.random.rand(H, W).astype(cp.float32)
    output_matrix = cp.zeros((W, H), dtype=cp.float32)
    
    # 计算网格大小
    grid_rows = H // tile_h  # 16
    grid_cols = W // tile_w  # 8
    grid = (grid_rows, grid_cols, 1)
    
    # 启动内核
    stream = cp.cuda.get_current_stream()
    ct.launch(stream, grid, matrix_transpose,
              (input_matrix, output_matrix, tile_h, tile_w))
    
    # 验证结果
    expected = input_matrix.T
    np.testing.assert_array_almost_equal(
        cp.asnumpy(output_matrix), 
        cp.asnumpy(expected)
    )
    
    print("Matrix transpose test passed!")

if __name__ == "__main__":
    test_matrix_transpose()
```

### 示例 2：多 Tile 处理 - 滑动窗口

```python
@ct.kernel
def sliding_window_avg(input_arr, output_arr, 
                       tile_size: ct.Constant[int],
                       window_size: ct.Constant[int]):
    """
    计算滑动窗口平均值
    
    演示一个 Block 处理多个 Tile
    """
    pid = ct.bid(0)
    
    # 每个 Block 处理 3 个相邻的 Tile
    # 当前 Tile
    current_tile = ct.load(input_arr, 
                          index=(pid,), 
                          shape=(tile_size,))
    
    # 前一个 Tile（如果存在）
    if pid > 0:
        prev_tile = ct.load(input_arr, 
                           index=(pid - 1,), 
                           shape=(tile_size,))
    else:
        prev_tile = ct.zeros(shape=(tile_size,))
    
    # 后一个 Tile（如果存在）
    if pid < ct.grid_dim(0) - 1:
        next_tile = ct.load(input_arr, 
                           index=(pid + 1,), 
                           shape=(tile_size,))
    else:
        next_tile = ct.zeros(shape=(tile_size,))
    
    # 计算平均值（简化版）
    avg = (prev_tile + current_tile + next_tile) / 3.0
    
    # 存储结果
    ct.store(output_arr, index=(pid,), tile=avg)

def test_sliding_window():
    n = 1024
    tile_size = 32
    
    input_arr = cp.arange(n, dtype=cp.float32)
    output_arr = cp.zeros(n, dtype=cp.float32)
    
    grid = (n // tile_size, 1, 1)
    stream = cp.cuda.get_current_stream()
    
    ct.launch(stream, grid, sliding_window_avg,
              (input_arr, output_arr, tile_size, 3))
    
    print("Sliding window test passed!")
```

### 示例 3：条件执行与循环

```python
@ct.kernel
def iterative_refinement(data, 
                        iterations: ct.Constant[int],
                        threshold: ct.Constant[float],
                        tile_size: ct.Constant[int]):
    """
    迭代精炼算法
    
    演示控制流和常量参数的使用
    """
    pid = ct.bid(0)
    
    # 加载数据
    tile = ct.load(data, index=(pid,), shape=(tile_size,))
    
    # 迭代处理
    for iter in range(iterations):
        # 应用平滑操作
        tile = tile * 0.9 + 0.1
        
        # 条件处理
        if iter > iterations // 2:
            # 后半部分迭代使用不同的策略
            tile = tile * threshold
        else:
            # 前半部分迭代
            tile = tile + (1.0 - threshold)
    
    # 最终归一化
    if pid % 2 == 0:
        tile = tile * 2.0
    else:
        tile = tile * 0.5
    
    # 存储结果
    ct.store(data, index=(pid,), tile=tile)
```

---

## 总结

### 关键要点

1. **Block vs Tile**
   - Block = 执行单元
   - Tile = 数据单元
   - 一个 Block 可以处理多个 Tile

2. **执行空间**
   - Host Code（CPU）
   - Tile Code（GPU - Tile 抽象）
   - SIMT Code（GPU - 传统 CUDA）

3. **函数类型**
   - `@ct.kernel`: 内核，Grid 入口
   - `@ct.function`: Tile 函数，可复用逻辑
   - 自动推断：未标注函数自动加入执行空间

4. **Python 子集**
   - 无 Python 运行时
   - 对象不可变（除了数组参数）
   - 支持基本控制流

5. **并行性**
   - Block 级：显式指定
   - Thread 级：自动处理
   - Tile 操作：自动并行化

6. **常量性**
   - 常量嵌入：编译时优化
   - 类型提示：`ct.Constant[T]`
   - 不同值触发重新编译

### 最佳实践

**推荐做法：**
- 使用常量嵌入标注 Tile 形状参数
- 合理组织 Grid 大小以匹配问题规模
- 利用 `@ct.function` 复用代码
- 让编译器自动推断执行空间（简单函数）

**避免的做法：**
- 不要混淆 Block 和 Tile 的概念
- 不要在 Tile Code 中使用未支持的 Python 特性
- 不要对频繁变化的参数使用常量嵌入
- 不要传递别名数组给内核

---

**版权声明**: Copyright © 2025, NVIDIA Corporation.  
**许可证**: Apache-2.0
**官方文档**: https://docs.nvidia.com/cuda/cutile-python/quickstart.html

