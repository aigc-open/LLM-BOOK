# cuTile Python 数据模型详解

## 目录
1. [概述](#概述)
2. [全局数组 (Global Arrays)](#全局数组-global-arrays)
3. [Tile 数组 (Tiles)](#tile-数组-tiles)
4. [元素空间与 Tile 空间](#元素空间与-tile-空间)
5. [形状广播 (Shape Broadcasting)](#形状广播-shape-broadcasting)
6. [数据类型 (Data Types)](#数据类型-data-types)
7. [数值与算术类型](#数值与算术类型)
8. [算术类型提升](#算术类型提升)
9. [标量 (Scalars)](#标量-scalars)
10. [元组 (Tuples)](#元组-tuples)
11. [舍入模式 (Rounding Modes)](#舍入模式-rounding-modes)
12. [填充模式 (Padding Modes)](#填充模式-padding-modes)
13. [实战示例](#实战示例)

---

## 概述

cuTile 是一个**基于数组的编程模型**。核心数据结构是具有单一同类型元素的多维数组。

### 为什么选择数组模型？

cuTile Python 不暴露指针，只使用数组。这样设计的原因：

1. **安全性**：数组知道自己的边界，可以检查访问以确保安全和正确性
2. **性能**：基于数组的加载/存储操作可以高效地映射到硬件机制
3. **熟悉性**：Python 程序员习惯于使用 NumPy 等基于数组的框架
4. **自然性**：指针不是 Python 的自然选择

**核心原则：**
> 在 Tile Code 中，只支持本节描述的类型。

---

## 全局数组 (Global Arrays)

### 基本概念

```python
class cuda.tile.Array
```

**全局数组（Global Array 或简称 Array）** 是存储在逻辑多维空间中的对象容器。

### 关键特性

**存储位置：**
- 全局数组总是存储在内存中
- 复制数组不会复制底层数据（浅拷贝）

**可用性：**
- 可以在 Host Code 和 Tile Code 中使用
- 可以作为内核参数

**兼容性：**
- 任何实现 **DLPack 格式** 或 **CUDA Array Interface** 的对象都可以用作全局数组
- 示例：CuPy 数组、PyTorch 张量

### 使用示例

#### 示例 1：使用 CuPy 数组作为全局数组

```python
import cupy as cp
import cuda.tile as ct

# 创建 CuPy 数组（实现了 CUDA Array Interface）
array_a = cp.arange(1024, dtype=cp.float32)
array_b = cp.zeros(1024, dtype=cp.float32)

# 可以直接作为内核参数
@ct.kernel
def process_array(a, b):
    """
    a, b 都是全局数组（Global Arrays）
    """
    pid = ct.bid(0)
    
    # 从全局数组加载到 Tile
    tile_a = ct.load(a, index=(pid,), shape=(16,))
    
    # 处理
    result = tile_a * 2.0
    
    # 存储回全局数组
    ct.store(b, index=(pid,), tile=result)

# 启动内核
grid = (1024 // 16, 1, 1)
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, process_array, (array_a, array_b))
```

#### 示例 2：使用 PyTorch 张量作为全局数组

```python
import torch
import cuda.tile as ct

# 创建 PyTorch CUDA 张量（实现了 DLPack）
tensor_a = torch.arange(256, dtype=torch.float32, device='cuda')
tensor_b = torch.zeros(256, dtype=torch.float32, device='cuda')

@ct.kernel
def torch_kernel(a, b):
    pid = ct.bid(0)
    tile = ct.load(a, index=(pid,), shape=(16,))
    result = tile + 1.0
    ct.store(b, index=(pid,), tile=result)

# PyTorch 张量可以直接使用
stream = torch.cuda.current_stream().cuda_stream
grid = (256 // 16, 1, 1)
ct.launch(stream, grid, torch_kernel, (tensor_a, tensor_b))
```

#### 示例 3：数组的浅拷贝特性

```python
import cupy as cp

# 创建原始数组
original = cp.arange(100)

# 复制数组（浅拷贝）
copy = original

# 修改复制的数组会影响原始数组
copy[0] = 999
print(original[0])  # 输出：999

# 如果需要深拷贝
deep_copy = original.copy()
deep_copy[1] = 888
print(original[1])  # 输出：1（未被修改）
```

### 全局数组属性

```python
import cuda.tile as ct
import cupy as cp

@ct.kernel
def array_info_kernel(arr):
    """展示全局数组的属性"""
    # 注意：arr.ndim 和 arr.shape 可以在内核中访问
    # 但某些属性可能需要是常量
    
    pid = ct.bid(0)
    
    # 数组维度（编译时常量）
    # ndim = arr.ndim
    
    # 数组形状（运行时值，默认 int32）
    # shape = arr.shape
    
    # 加载数据
    tile = ct.load(arr, index=(pid,), shape=(16,))
    ct.store(arr, index=(pid,), tile=tile * 2)
```

---

## Tile 数组 (Tiles)

### 基本概念

```python
class cuda.tile.Tile
```

**Tile 数组（Tile Array 或简称 Tile）** 是块（Block）局部的不可变多维值集合。

### 关键特性

**存储特性：**
- Tile 的内容**不一定**在内存中有表示
- Tile 可以由全局数组加载创建，或通过工厂函数创建
- Tile 可以存储到全局数组

**可用性：**
- **不能**在 Host Code 中使用
- **只能**在 Tile Code 中使用
- **不能**作为内核参数

**形状限制：**
- Tile 的每个维度必须是 **2 的幂次方**

### Tile vs 全局数组对比

| 特性 | 全局数组 (Array) | Tile 数组 (Tile) |
|------|-----------------|------------------|
| **存储位置** | 总是在内存中 | 可能不在内存中 |
| **可变性** | 可变（可修改） | 不可变 |
| **作用域** | 全局 | Block 局部 |
| **Host 可用** | 是 | 否 |
| **内核参数** | 是 | 否 |
| **形状限制** | 无限制 | 必须是 2 的幂 |

### 使用示例

#### 示例 1：创建和使用 Tile

```python
import cuda.tile as ct
import cupy as cp

@ct.kernel
def tile_demo(input_arr, output_arr):
    pid = ct.bid(0)
    
    # 方式 1：从全局数组加载创建 Tile
    # shape 必须是 2 的幂：(2, 4, 8, 16, 32, 64, 128, ...)
    tile_a = ct.load(input_arr, index=(pid,), shape=(16,))  # 1D Tile
    
    # 方式 2：从多维数组加载
    tile_b = ct.load(input_arr, index=(pid, 0), shape=(16, 16))  # 2D Tile
    
    # Tile 操作（不可变，返回新 Tile）
    tile_c = tile_a * 2.0  # tile_a 保持不变，tile_c 是新 Tile
    tile_d = tile_c + 1.0  # tile_c 保持不变，tile_d 是新 Tile
    
    # 存储回全局数组
    ct.store(output_arr, index=(pid,), tile=tile_d)

# 使用示例
n = 1024
input_data = cp.random.rand(n).astype(cp.float32)
output_data = cp.zeros(n, dtype=cp.float32)

grid = (n // 16, 1, 1)
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, tile_demo, (input_data, output_data))
```

#### 示例 2：Tile 的不可变性

```python
@ct.kernel
def immutable_tile_demo(arr):
    pid = ct.bid(0)
    
    # 创建 Tile
    tile_original = ct.load(arr, index=(pid,), shape=(16,))
    
    # 操作 1：返回新 Tile
    tile_modified = tile_original * 2.0
    
    # tile_original 保持不变
    # tile_modified 是新的 Tile
    
    # 操作 2：链式操作
    tile_result = tile_original + 1.0 - 0.5 * 2.0
    
    # 每个操作都创建新 Tile，原 Tile 不变
    ct.store(arr, index=(pid,), tile=tile_result)
```

#### 示例 3：多维 Tile

```python
@ct.kernel
def matrix_tile_demo(matrix, output):
    """处理 2D Tile"""
    row_id = ct.bid(0)
    col_id = ct.bid(1)
    
    # 加载 2D Tile（16x16）
    # 两个维度都必须是 2 的幂
    tile_2d = ct.load(matrix, index=(row_id, col_id), shape=(16, 16))
    
    # 2D Tile 操作
    result = tile_2d * 2.0 + 1.0
    
    # 存储 2D Tile
    ct.store(output, index=(row_id, col_id), tile=result)

# 使用示例
matrix = cp.random.rand(256, 256).astype(cp.float32)
output = cp.zeros_like(matrix)

# 2D Grid
grid = (256 // 16, 256 // 16, 1)
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, matrix_tile_demo, (matrix, output))
```

#### 示例 4：Tile 形状限制

```python
import cuda.tile as ct

@ct.kernel
def tile_shape_examples(arr):
    pid = ct.bid(0)
    
    # 合法的 Tile 形状（2 的幂）
    tile_2 = ct.load(arr, index=(pid,), shape=(2,))      # 2^1
    tile_4 = ct.load(arr, index=(pid,), shape=(4,))      # 2^2
    tile_8 = ct.load(arr, index=(pid,), shape=(8,))      # 2^3
    tile_16 = ct.load(arr, index=(pid,), shape=(16,))    # 2^4
    tile_32 = ct.load(arr, index=(pid,), shape=(32,))    # 2^5
    
    # 2D Tile（每个维度都是 2 的幂）
    tile_16x32 = ct.load(arr, index=(pid, 0), shape=(16, 32))  # 合法
    tile_8x8 = ct.load(arr, index=(pid, 0), shape=(8, 8))      # 合法
    
    # 非法的形状（不是 2 的幂）- 会导致编译错误
    # tile_3 = ct.load(arr, index=(pid,), shape=(3,))      # 错误！
    # tile_10 = ct.load(arr, index=(pid,), shape=(10,))    # 错误！
    # tile_15x20 = ct.load(arr, index=(pid, 0), shape=(15, 20))  # 错误！
```

### Tile 工厂函数

```python
import cuda.tile as ct

@ct.kernel
def tile_factory_demo(output):
    """使用工厂函数创建 Tile"""
    pid = ct.bid(0)
    
    # 创建全零 Tile
    tile_zeros = ct.zeros(shape=(16,), dtype=ct.float32)
    
    # 创建全一 Tile
    tile_ones = ct.ones(shape=(16,), dtype=ct.float32)
    
    # 创建常量 Tile
    tile_constant = ct.full(shape=(16,), fill_value=3.14, dtype=ct.float32)
    
    # 使用这些 Tile
    result = tile_zeros + tile_ones * tile_constant
    
    ct.store(output, index=(pid,), tile=result)
```

---

## 元素空间与 Tile 空间

### 概念说明

**元素空间（Element Space）**
- 数组中包含的元素的多维空间
- 元素按照特定布局（行主序、列主序等）存储在内存中

**Tile 空间（Tile Space）**
- 特定 Tile 形状的数组的多维 Tile 空间
- Tile 索引 `(i, j, ...)` 配合形状 `S` 指向属于第 `(i+1)`, `(j+1)`, ... 个 Tile 的数组元素

### 可视化理解

```
原始数组形状：12 x 16 (元素空间)

┌─────────────────────────────────────────┐
│ 0  1  2  3  4  5  6  7  8  9 10 11 12...│ ← 12 行
│                                         │
│  ... 共 16 列 ...                       │
└─────────────────────────────────────────┘

示例 1：Tile 形状 = 2 x 4 (Tile 空间 = 6 x 4)

┌─────┬─────┬─────┬─────┐
│(0,0)│(0,1)│(0,2)│(0,3)│  ← Tile 索引
├─────┼─────┼─────┼─────┤
│(1,0)│(1,1)│(1,2)│(1,3)│
├─────┼─────┼─────┼─────┤
│(2,0)│(2,1)│(2,2)│(2,3)│
├─────┼─────┼─────┼─────┤
│(3,0)│(3,1)│(3,2)│(3,3)│
├─────┼─────┼─────┼─────┤
│(4,0)│(4,1)│(4,2)│(4,3)│
├─────┼─────┼─────┼─────┤
│(5,0)│(5,1)│(5,2)│(5,3)│
└─────┴─────┴─────┴─────┘

每个 Tile 包含 2x4=8 个元素
总共 6x4=24 个 Tile，覆盖 12x16=192 个元素


示例 2：Tile 形状 = 4 x 2 (Tile 空间 = 3 x 8)

┌──┬──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │  │  │  │  │  ← 3 行 Tile
├──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │
└──┴──┴──┴──┴──┴──┴──┴──┘
   ↑ 8 列 Tile

每个 Tile 包含 4x2=8 个元素
总共 3x8=24 个 Tile，覆盖 12x16=192 个元素
```

### 代码示例

#### 示例 1：理解 Tile 索引

```python
import cupy as cp
import cuda.tile as ct

@ct.kernel
def tile_index_demo(array, output):
    """
    演示 Tile 索引的含义
    """
    # 获取 2D Tile 索引
    tile_row = ct.bid(0)  # Tile 行索引
    tile_col = ct.bid(1)  # Tile 列索引
    
    # 加载对应的 Tile（每个 Tile 是 4x4 元素）
    # Tile (i, j) 包含元素 [i*4:(i+1)*4, j*4:(j+1)*4]
    tile = ct.load(array, index=(tile_row, tile_col), shape=(4, 4))
    
    # 处理这个 Tile
    result = tile * 2.0
    
    # 存储回对应位置
    ct.store(output, index=(tile_row, tile_col), tile=result)

# 创建测试数据
# 数组形状：16x16 元素
# Tile 形状：4x4 元素
# Tile 空间：4x4 Tile
H, W = 16, 16
array = cp.arange(H * W, dtype=cp.float32).reshape(H, W)
output = cp.zeros_like(array)

# Grid 大小 = Tile 空间大小
tile_h, tile_w = 4, 4
grid_h = H // tile_h  # 4 个 Tile 行
grid_w = W // tile_w  # 4 个 Tile 列
grid = (grid_h, grid_w, 1)

stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, tile_index_demo, (array, output))

print("原始数组第一个 Tile (0,0) 的元素:")
print(array[0:4, 0:4])
print("\n处理后的对应元素:")
print(output[0:4, 0:4])
```

#### 示例 2：内存布局与 order 参数

```python
import cuda.tile as ct
import cupy as cp

@ct.kernel
def layout_demo(array_row_major, array_col_major, output):
    """
    演示不同内存布局的访问
    """
    pid = ct.bid(0)
    
    # 默认按照数组的内存布局加载
    tile_row = ct.load(array_row_major, index=(pid, 0), shape=(4, 4))
    
    # 使用 order 参数指定不同的布局
    # order='C' 表示行主序（C-style）
    # order='F' 表示列主序（Fortran-style）
    tile_col = ct.load(array_col_major, index=(pid, 0), shape=(4, 4), order='F')
    
    # 处理
    result = tile_row + tile_col
    
    # 存储（也可以指定 order）
    ct.store(output, index=(pid, 0), tile=result, order='C')

# 行主序数组（默认）
array_c = cp.arange(64, dtype=cp.float32).reshape(16, 4, order='C')

# 列主序数组
array_f = cp.arange(64, dtype=cp.float32).reshape(16, 4, order='F')

output = cp.zeros((16, 4), dtype=cp.float32, order='C')

grid = (16 // 4, 1, 1)
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, layout_demo, (array_c, array_f, output))
```

#### 示例 3：1D 数组的 Tile 视图

```python
@ct.kernel
def linear_tile_view(array, output):
    """
    1D 数组的 Tile 划分
    
    数组：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    Tile 大小：4
    Tile 空间：4 个 Tile
    
    Tile 0: [0, 1, 2, 3]
    Tile 1: [4, 5, 6, 7]
    Tile 2: [8, 9, 10, 11]
    Tile 3: [12, 13, 14, 15]
    """
    tile_index = ct.bid(0)
    
    # 加载 Tile
    tile = ct.load(array, index=(tile_index,), shape=(4,))
    
    # Tile 索引 i 对应元素 [i*4, i*4+1, i*4+2, i*4+3]
    result = tile + 100.0
    
    ct.store(output, index=(tile_index,), tile=result)

# 测试
array = cp.arange(16, dtype=cp.float32)
output = cp.zeros(16, dtype=cp.float32)

grid = (4, 1, 1)  # 4 个 Tile
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, linear_tile_view, (array, output))

print("原始数组:", array)
print("处理后:", output)
# 输出：
# 原始数组: [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
# 处理后: [100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115]
```

---

## 形状广播 (Shape Broadcasting)

### 基本概念

**形状广播** 允许不同形状的 Tile 在算术操作中组合。较小的 Tile 会自动扩展以匹配较大 Tile 的形状。

### 广播规则

cuTile 遵循与 **NumPy 相同的广播语义**：

1. **尾部对齐**：Tile 按其尾部维度对齐
2. **兼容性**：对应维度的大小相同，或其中一个为 1
3. **维度填充**：维度较少的 Tile 在左侧用 1 填充

### 广播示例

```
示例 1：标量广播
Tile A: shape (16,)        [a0, a1, a2, ..., a15]
Scalar: shape ()           5
结果:   shape (16,)        [a0+5, a1+5, a2+5, ..., a15+5]

示例 2：1D 广播
Tile A: shape (16,)        [a0, a1, ..., a15]
Tile B: shape (1,)         [b0]
结果:   shape (16,)        [a0+b0, a1+b0, ..., a15+b0]

示例 3：2D 广播（行向量）
Tile A: shape (4, 8)       4x8 矩阵
Tile B: shape (1, 8)       1x8 行向量
结果:   shape (4, 8)       B 的行被广播到 A 的每一行

示例 4：2D 广播（列向量）
Tile A: shape (4, 8)       4x8 矩阵
Tile B: shape (4, 1)       4x1 列向量
结果:   shape (4, 8)       B 的列被广播到 A 的每一列

示例 5：维度填充
Tile A: shape (4, 8, 16)   3D Tile
Tile B: shape (8, 16)      2D Tile（被视为 (1, 8, 16)）
结果:   shape (4, 8, 16)   B 在第一维被广播
```

### 代码示例

#### 示例 1：标量与 Tile 广播

```python
import cuda.tile as ct
import cupy as cp

@ct.kernel
def scalar_broadcast_demo(array, output, scale):
    """
    标量自动广播到 Tile 的每个元素
    """
    pid = ct.bid(0)
    
    # 加载 Tile
    tile = ct.load(array, index=(pid,), shape=(16,))
    
    # 标量广播
    # scale 是标量，自动广播到 tile 的每个元素
    result = tile * scale + 10.0  # 两个标量都被广播
    
    ct.store(output, index=(pid,), tile=result)

# 测试
array = cp.arange(64, dtype=cp.float32)
output = cp.zeros(64, dtype=cp.float32)

grid = (64 // 16, 1, 1)
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, scalar_broadcast_demo, (array, output, 2.0))
```

#### 示例 2：1D Tile 广播

```python
@ct.kernel
def vector_broadcast_demo(matrix, row_bias, output):
    """
    1D Tile 广播到 2D Tile
    """
    row = ct.bid(0)
    col = ct.bid(1)
    
    # 加载 2D Tile (4x8)
    tile_2d = ct.load(matrix, index=(row, col), shape=(4, 8))
    
    # 加载 1D Tile (8,) - 将作为行向量广播
    tile_1d = ct.load(row_bias, index=(col,), shape=(8,))
    
    # 广播相加
    # tile_1d 的形状 (8,) 被视为 (1, 8)
    # 然后广播到 (4, 8) 以匹配 tile_2d
    result = tile_2d + tile_1d
    
    ct.store(output, index=(row, col), tile=result)

# 测试
matrix = cp.random.rand(16, 32).astype(cp.float32)
row_bias = cp.arange(32, dtype=cp.float32)  # 行偏置
output = cp.zeros_like(matrix)

grid = (16 // 4, 32 // 8, 1)
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, vector_broadcast_demo, (matrix, row_bias, output))
```

#### 示例 3：完整的广播示例

```python
@ct.kernel
def full_broadcast_demo(matrix, row_vec, col_vec, scalar, output):
    """
    演示多种广播场景
    """
    row = ct.bid(0)
    col = ct.bid(1)
    
    # 加载 2D Tile (8x8)
    tile_matrix = ct.load(matrix, index=(row, col), shape=(8, 8))
    
    # 加载行向量 Tile (8,) - 将广播到每一行
    tile_row = ct.load(row_vec, index=(col,), shape=(8,))
    
    # 加载列向量 Tile (8,) - 需要 reshape 或使用特殊加载
    # 这里简化处理，假设 col_vec 已经是适当形状
    
    # 场景 1：矩阵 + 标量
    result1 = tile_matrix + scalar
    
    # 场景 2：矩阵 + 行向量（广播到每一行）
    result2 = tile_matrix + tile_row
    
    # 场景 3：组合操作
    result = (tile_matrix + tile_row) * scalar
    
    ct.store(output, index=(row, col), tile=result)
```

#### 示例 4：实际应用 - 批归一化

```python
@ct.kernel
def batch_norm_demo(data, mean, std, gamma, beta, output):
    """
    使用广播实现批归一化
    
    data:  (N, C) - N 个样本，C 个通道
    mean:  (C,)   - 每个通道的均值
    std:   (C,)   - 每个通道的标准差
    gamma: (C,)   - 缩放参数
    beta:  (C,)   - 偏移参数
    """
    batch_id = ct.bid(0)
    channel_id = ct.bid(1)
    
    # 加载数据 Tile (16, 16)
    tile_data = ct.load(data, index=(batch_id, channel_id), shape=(16, 16))
    
    # 加载统计量 (16,)
    tile_mean = ct.load(mean, index=(channel_id,), shape=(16,))
    tile_std = ct.load(std, index=(channel_id,), shape=(16,))
    tile_gamma = ct.load(gamma, index=(channel_id,), shape=(16,))
    tile_beta = ct.load(beta, index=(channel_id,), shape=(16,))
    
    # 归一化（广播操作）
    # (16, 16) - (16,) 自动广播
    normalized = (tile_data - tile_mean) / (tile_std + 1e-5)
    
    # 缩放和偏移（广播操作）
    result = normalized * tile_gamma + tile_beta
    
    ct.store(output, index=(batch_id, channel_id), tile=result)
```

### 广播兼容性检查

```python
# 兼容的形状对
(16,)      + (16,)      → (16,)        # 相同形状
(16, 8)    + (16, 8)    → (16, 8)      # 相同形状
(16, 8)    + (1, 8)     → (16, 8)      # 维度 0 广播
(16, 8)    + (16, 1)    → (16, 8)      # 维度 1 广播
(16, 8)    + (8,)       → (16, 8)      # 尾部对齐
(16, 8, 4) + (8, 4)     → (16, 8, 4)   # 维度填充
(16,)      + ()         → (16,)        # 标量广播

# 不兼容的形状对（会导致错误）
(16, 8) + (16, 4)  # 维度 1 不兼容：8 ≠ 4 且都不为 1
(16, 8) + (8, 16)  # 形状不匹配
```

---

## 数据类型 (Data Types)

### 基本概念

```python
class cuda.tile.DType
```

**数据类型（DType）** 描述数组、Tile 或操作中对象的类型。

### DType 的特性

- **用途**：决定值如何在内存中存储以及如何执行操作
- **不可变性**：DType 是不可变的
- **可用性**：可以在 Host Code 和 Tile Code 中使用，可以作为内核参数

### DType 属性

```python
import cuda.tile as ct

# DType 的属性
dtype = ct.float32

print(dtype.name)      # 'float32'
print(dtype.bitwidth)  # 32
```

### 完整的数据类型列表

#### 1. 布尔类型

```python
ct.bool_
```
- 8位算术类型（True 或 False）

#### 2. 无符号整数类型

| 类型 | 位宽 | 范围 |
|------|------|------|
| `ct.uint8` | 8位 | [0, 256) |
| `ct.uint16` | 16位 | [0, 65,536) |
| `ct.uint32` | 32位 | [0, 4,294,967,296) |
| `ct.uint64` | 64位 | [0, 18,446,744,073,709,551,616) |

#### 3. 有符号整数类型

| 类型 | 位宽 | 范围 |
|------|------|------|
| `ct.int8` | 8位 | [-128, 128) |
| `ct.int16` | 16位 | [-32,768, 32,768) |
| `ct.int32` | 32位 | [-2,147,483,648, 2,147,483,648) |
| `ct.int64` | 64位 | [-9,223,372,036,854,775,808, 9,223,372,036,854,775,808) |

#### 4. 浮点类型

| 类型 | 位宽 | 说明 |
|------|------|------|
| `ct.float16` | 16位 | IEEE 754 半精度浮点 |
| `ct.float32` | 32位 | IEEE 754 单精度浮点 |
| `ct.float64` | 64位 | IEEE 754 双精度浮点 |
| `ct.bfloat16` | 16位 | 1符号位 + 8指数位 + 7尾数位 |
| `ct.tfloat32` | 32位 | 1符号位 + 8指数位 + 10尾数位（19位表示存储在32位容器中）|
| `ct.float8_e4m3fn` | 8位 | 1符号位 + 4指数位 + 3尾数位 |
| `ct.float8_e5m2` | 8位 | 1符号位 + 5指数位 + 2尾数位 |

### 代码示例

#### 示例 1：使用不同数据类型

```python
import cuda.tile as ct
import cupy as cp

@ct.kernel
def dtype_demo(input_i32, input_f32, output_i32, output_f32):
    """
    演示不同数据类型的使用
    """
    pid = ct.bid(0)
    
    # 加载不同类型的 Tile
    tile_int = ct.load(input_i32, index=(pid,), shape=(16,))
    tile_float = ct.load(input_f32, index=(pid,), shape=(16,))
    
    # 类型特定的操作
    result_int = tile_int + 10
    result_float = tile_float * 2.5
    
    # 存储
    ct.store(output_i32, index=(pid,), tile=result_int)
    ct.store(output_f32, index=(pid,), tile=result_float)

# 创建不同类型的数组
input_i32 = cp.arange(64, dtype=cp.int32)
input_f32 = cp.random.rand(64).astype(cp.float32)
output_i32 = cp.zeros(64, dtype=cp.int32)
output_f32 = cp.zeros(64, dtype=cp.float32)

grid = (64 // 16, 1, 1)
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, dtype_demo, 
          (input_i32, input_f32, output_i32, output_f32))
```

#### 示例 2：低精度类型的使用

```python
@ct.kernel
def low_precision_demo(input_f32, output_f16, output_bf16):
    """
    使用低精度类型节省内存和提高性能
    """
    pid = ct.bid(0)
    
    # 加载 float32
    tile_f32 = ct.load(input_f32, index=(pid,), shape=(16,))
    
    # 转换为 float16
    tile_f16 = ct.cast(tile_f32, ct.float16)
    
    # 转换为 bfloat16
    tile_bf16 = ct.cast(tile_f32, ct.bfloat16)
    
    # 存储
    ct.store(output_f16, index=(pid,), tile=tile_f16)
    ct.store(output_bf16, index=(pid,), tile=tile_bf16)

# 使用示例
input_f32 = cp.random.rand(256).astype(cp.float32)
output_f16 = cp.zeros(256, dtype=cp.float16)
output_bf16 = cp.zeros(256, dtype=cp.float16)  # bfloat16 在 CuPy 中作为 float16 处理

grid = (256 // 16, 1, 1)
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, low_precision_demo, (input_f32, output_f16, output_bf16))

print(f"float32 内存: {input_f32.nbytes} bytes")
print(f"float16 内存: {output_f16.nbytes} bytes")
print(f"节省: {(1 - output_f16.nbytes / input_f32.nbytes) * 100:.1f}%")
```

#### 示例 3：DType 属性查询

```python
import cuda.tile as ct

def dtype_info():
    """查询 DType 信息"""
    dtypes = [
        ct.bool_, ct.uint8, ct.uint16, ct.uint32, ct.uint64,
        ct.int8, ct.int16, ct.int32, ct.int64,
        ct.float16, ct.float32, ct.float64,
        ct.bfloat16, ct.tfloat32
    ]
    
    print("数据类型信息:")
    print(f"{'类型':<15} {'位宽':<10} {'字节':<10}")
    print("-" * 35)
    
    for dtype in dtypes:
        bitwidth = dtype.bitwidth
        bytes_size = bitwidth // 8
        print(f"{dtype.name:<15} {bitwidth:<10} {bytes_size:<10}")

dtype_info()
```

输出：
```
数据类型信息:
类型             位宽        字节      
-----------------------------------
bool_           8          1         
uint8           8          1         
uint16          16         2         
uint32          32         4         
uint64          64         8         
int8            8          1         
int16           16         2         
int32           32         4         
int64           64         8         
float16         16         2         
float32         32         4         
float64         64         8         
bfloat16        16         2         
tfloat32        32         4         
```

---

## 数值与算术类型

### 定义

**数值数据类型（Numeric Data Type）**
- 表示数字的数据类型

**算术数据类型（Arithmetic Data Type）**
- 支持通用算术操作（加、减、乘、除等）的数值数据类型

### 分类

cuTile 中的数据类型可分为三个类别（按优先级排序）：

```
boolean < integral < floating-point
布尔型 < 整数型 < 浮点型
```

---

## 算术类型提升

### 基本概念

当对两个不同数值类型的 Tile 或标量操作数执行二元运算时，系统会自动提升到公共类型。

### 提升规则

#### 规则 1：松散类型常量

```python
# 两个操作数都是松散类型常量
result1 = 5 + 7       # 结果：松散类型整数常量 12
result2 = 5 + 3.0     # 结果：松散类型浮点常量 8.0
```

#### 规则 2：类别提升

如果操作数不是松散类型常量，提升过程：

1. **分类**：每个操作数被分为三类之一
   - 布尔型 (boolean)
   - 整数型 (integral)  
   - 浮点型 (floating-point)

2. **松散类型具体化**：
   - 整数常量 → `int32`、`int64` 或 `uint64`（取决于值）
   - 浮点常量 → `float32`

3. **类别优先级**：
   - 如果一个操作数类别高于另一个，选择高类别的具体类型

4. **同类别处理**：
   - 如果一个是松散类型常量，选择另一个的类型
   - 否则，按照类型提升表处理

### 类型提升表

```python
# 简化版提升规则示例

# 布尔型 + 整数型 → 整数型
bool_ + int32 → int32

# 整数型 + 浮点型 → 浮点型
int32 + float32 → float32

# 无符号 + 有符号（同宽度）→ 错误
uint8 + int8 → ERROR

# 不同宽度整数 → 较宽的类型
int16 + int32 → int32

# 不同浮点类型 → 较高精度
float16 + float32 → float32
```

### 完整类型提升表

下表显示了两种类型之间的提升结果：

| 类型1 | 类型2 | 结果 | 说明 |
|-------|-------|------|------|
| `bool_` | `int8` | `int8` | 布尔提升为整数 |
| `uint8` | `uint16` | `uint16` | 提升为更宽的无符号 |
| `int8` | `int16` | `int16` | 提升为更宽的有符号 |
| `uint8` | `int8` | `ERROR` | 有符号和无符号不能混合 |
| `int32` | `float32` | `float32` | 整数提升为浮点 |
| `float16` | `float32` | `float32` | 提升为更高精度 |
| `bfloat16` | `float32` | `float32` | BF16 提升为 FP32 |
| `tfloat32` | 任何 | `ERROR` | TF32 不支持隐式提升 |
| `float8_*` | 任何 | `ERROR` | FP8 不支持隐式提升 |

### 代码示例

#### 示例 1：基本类型提升

```python
import cuda.tile as ct
import cupy as cp

@ct.kernel
def type_promotion_demo(arr_i16, arr_i32, arr_f32, output):
    """演示类型自动提升"""
    pid = ct.bid(0)
    
    # 加载不同类型的 Tile
    tile_i16 = ct.load(arr_i16, index=(pid,), shape=(16,))  # int16
    tile_i32 = ct.load(arr_i32, index=(pid,), shape=(16,))  # int32
    tile_f32 = ct.load(arr_f32, index=(pid,), shape=(16,))  # float32
    
    # 类型提升示例
    result1 = tile_i16 + tile_i32  # int16 + int32 → int32
    result2 = tile_i32 + tile_f32  # int32 + float32 → float32
    result3 = tile_i16 + 1.0       # int16 + float → float32
    
    # 最终结果类型是 float32
    result = ct.cast(result1, ct.float32) + result2 + result3
    
    ct.store(output, index=(pid,), tile=result)

# 测试
arr_i16 = cp.arange(64, dtype=cp.int16)
arr_i32 = cp.arange(64, dtype=cp.int32)
arr_f32 = cp.random.rand(64).astype(cp.float32)
output = cp.zeros(64, dtype=cp.float32)

grid = (64 // 16, 1, 1)
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, type_promotion_demo, 
          (arr_i16, arr_i32, arr_f32, output))
```

#### 示例 2：显式类型转换

```python
@ct.kernel
def explicit_cast_demo(arr_u8, arr_i8, output):
    """
    当隐式提升不可用时，使用显式转换
    """
    pid = ct.bid(0)
    
    # 加载
    tile_u8 = ct.load(arr_u8, index=(pid,), shape=(16,))  # uint8
    tile_i8 = ct.load(arr_i8, index=(pid,), shape=(16,))  # int8
    
    # 错误：uint8 + int8 不能隐式提升
    # result = tile_u8 + tile_i8  # ERROR!
    
    # 正确：显式转换到公共类型
    tile_u8_as_i16 = ct.cast(tile_u8, ct.int16)
    tile_i8_as_i16 = ct.cast(tile_i8, ct.int16)
    result = tile_u8_as_i16 + tile_i8_as_i16  # int16
    
    # 转换回 int8 存储
    result_i8 = ct.cast(result, ct.int8)
    ct.store(output, index=(pid,), tile=result_i8)
```

#### 示例 3：混合精度计算

```python
@ct.kernel
def mixed_precision_demo(arr_f16, arr_f32, output_f32, output_f16):
    """
    混合精度计算示例
    """
    pid = ct.bid(0)
    
    # 加载不同精度
    tile_f16 = ct.load(arr_f16, index=(pid,), shape=(16,))  # float16
    tile_f32 = ct.load(arr_f32, index=(pid,), shape=(16,))  # float32
    
    # float16 + float32 → float32（自动提升）
    result_f32 = tile_f16 + tile_f32
    
    # 高精度计算
    result_high = result_f32 * 2.0 + 1.0
    
    # 转换回低精度（节省内存）
    result_f16 = ct.cast(result_high, ct.float16)
    
    # 存储两个版本
    ct.store(output_f32, index=(pid,), tile=result_high)
    ct.store(output_f16, index=(pid,), tile=result_f16)
```

---

## 标量 (Scalars)

### 基本概念

**标量（Scalar）** 是特定数据类型的单个不可变值。

### 关键特性

- 标量和 0 维 Tile 可以互换使用
- 可以作为内核参数
- 可以在 Tile Code 中使用

### 标量类型规则

#### 规则 1：常量标量（松散类型）

```python
# 字面量常量
x = 2           # 松散类型整数
y = 3.14        # 松散类型浮点

# Tile 属性（常量）
ndim = tile.ndim      # 松散类型整数
shape = tile.shape    # 松散类型整数元组
```

#### 规则 2：数组形状和步长（默认 int32）

```python
# Array.shape 和 Array.stride 默认是 int32
# 这样可以提高性能，但限制最大可表示形状
# 注意：此限制将在未来版本中解除
```

### 代码示例

#### 示例 1：标量作为内核参数

```python
import cuda.tile as ct
import cupy as cp

@ct.kernel
def scalar_param_demo(arr, output, scale: ct.float32, offset: ct.int32):
    """
    标量作为内核参数
    """
    pid = ct.bid(0)
    
    # 加载 Tile
    tile = ct.load(arr, index=(pid,), shape=(16,))
    
    # 使用标量参数
    result = tile * scale + offset
    
    ct.store(output, index=(pid,), tile=result)

# 传递标量参数
arr = cp.arange(64, dtype=cp.float32)
output = cp.zeros(64, dtype=cp.float32)

scale = 2.5      # 标量
offset = 10      # 标量

grid = (64 // 16, 1, 1)
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, scalar_param_demo, (arr, output, scale, offset))
```

#### 示例 2：标量与 0 维 Tile

```python
@ct.kernel
def scalar_tile_demo(arr, output):
    """
    标量和 0 维 Tile 的互换使用
    """
    pid = ct.bid(0)
    
    # 标量
    scalar_value = 5.0
    
    # 0 维 Tile（从标量创建）
    tile_0d = ct.full(shape=(), fill_value=5.0, dtype=ct.float32)
    
    # 加载 1D Tile
    tile_1d = ct.load(arr, index=(pid,), shape=(16,))
    
    # 标量和 0 维 Tile 可以互换
    result1 = tile_1d + scalar_value  # 使用标量
    result2 = tile_1d + tile_0d       # 使用 0 维 Tile（相同效果）
    
    ct.store(output, index=(pid,), tile=result1)
```

#### 示例 3：Tile 属性作为标量

```python
@ct.kernel
def tile_properties_demo(arr, output):
    """
    Tile 属性返回松散类型标量常量
    """
    pid = ct.bid(0)
    
    # 加载 2D Tile
    tile = ct.load(arr, index=(pid, 0), shape=(16, 8))
    
    # Tile 属性（松散类型常量标量）
    ndim = tile.ndim           # 2（松散类型整数）
    shape0 = tile.shape[0]     # 16（松散类型整数）
    shape1 = tile.shape[1]     # 8（松散类型整数）
    size = tile.size           # 128（松散类型整数）
    
    # 可以在编译时常量表达式中使用
    # 例如：用于循环展开
    result = tile
    for i in range(3):  # 编译时常量
        result = result + 1.0
    
    ct.store(output, index=(pid, 0), tile=result)
```

---

## 元组 (Tuples)

### 基本概念

**元组（Tuples）** 可以在 Tile Code 中使用，但**不能**作为内核参数。

### 使用场景

1. **形状和索引表示**
2. **多个返回值**
3. **参数打包**

### 代码示例

#### 示例 1：形状和索引元组

```python
import cuda.tile as ct

@ct.kernel
def tuple_index_demo(arr, output):
    """使用元组表示索引和形状"""
    pid = ct.bid(0)
    
    # 元组表示 2D 索引
    index = (pid, 0)
    
    # 元组表示 2D 形状
    shape = (16, 16)
    
    # 使用元组加载
    tile = ct.load(arr, index=index, shape=shape)
    
    # 访问元组元素
    row_idx = index[0]
    col_idx = index[1]
    
    # 处理
    result = tile * 2.0
    
    ct.store(output, index=index, tile=result)
```

#### 示例 2：多返回值函数

```python
@ct.function
def compute_stats(tile):
    """
    返回多个统计值（使用元组）
    """
    tile_sum = ct.sum(tile)
    tile_mean = tile_sum / tile.size
    tile_max = ct.max(tile)
    tile_min = ct.min(tile)
    
    # 返回元组
    return (tile_mean, tile_max, tile_min)

@ct.kernel
def stats_demo(arr, mean_out, max_out, min_out):
    """使用返回元组的函数"""
    pid = ct.bid(0)
    
    tile = ct.load(arr, index=(pid,), shape=(16,))
    
    # 接收元组返回值
    mean, max_val, min_val = compute_stats(tile)
    
    # 使用这些值
    # 注意：这里简化了标量存储的处理
    # 实际使用中需要适当的存储机制
```

#### 示例 3：元组解包

```python
@ct.kernel
def tuple_unpacking_demo(arr, output):
    """元组解包示例"""
    pid = ct.bid(0)
    
    # 创建元组
    shape_tuple = (16, 8)
    
    # 解包元组
    height, width = shape_tuple
    
    # 使用解包的值
    tile = ct.load(arr, index=(pid, 0), shape=(height, width))
    
    # Tile 的 shape 属性也是元组
    tile_shape = tile.shape
    dim0, dim1 = tile_shape  # 解包
    
    result = tile * 2.0
    ct.store(output, index=(pid, 0), tile=result)
```

---

## 舍入模式 (Rounding Modes)

### 基本概念

```python
class cuda.tile.RoundingMode
```

**舍入模式（Rounding Mode）** 控制浮点操作的舍入行为。

### 舍入模式列表

| 模式 | 常量 | 说明 |
|------|------|------|
| **最近偶数** | `RN` = `'nearest_even'` | 舍入到最近的值（平局取偶数）|
| **向零** | `RZ` = `'zero'` | 向零舍入（截断）|
| **向负无穷** | `RM` = `'negative_inf'` | 向负无穷方向舍入 |
| **向正无穷** | `RP` = `'positive_inf'` | 向正无穷方向舍入 |
| **完全精度** | `FULL` = `'full'` | 完全精度舍入模式 |
| **近似** | `APPROX` = `'approx'` | 近似舍入模式 |
| **向零最近整数** | `RZI` = `'nearest_int_to_zero'` | 向零方向舍入到最近整数 |

### 代码示例

#### 示例 1：舍入模式的影响

```python
import cuda.tile as ct
import cupy as cp

@ct.kernel
def rounding_mode_demo(arr, out_rn, out_rz, out_rm, out_rp):
    """
    演示不同舍入模式的效果
    """
    pid = ct.bid(0)
    
    # 加载浮点数据
    tile = ct.load(arr, index=(pid,), shape=(16,))
    
    # 不同舍入模式的除法
    # 注意：具体 API 可能因版本而异
    # result_rn = ct.divide(tile, 3.0, rounding=ct.RoundingMode.RN)
    # result_rz = ct.divide(tile, 3.0, rounding=ct.RoundingMode.RZ)
    # result_rm = ct.divide(tile, 3.0, rounding=ct.RoundingMode.RM)
    # result_rp = ct.divide(tile, 3.0, rounding=ct.RoundingMode.RP)
    
    # 简化示例：类型转换时的舍入
    result_rn = ct.cast(tile / 3.0, ct.int32)  # 默认舍入
    
    ct.store(out_rn, index=(pid,), tile=ct.cast(result_rn, ct.float32))
```

#### 示例 2：精度控制

```python
@ct.kernel
def precision_control_demo(arr_high, arr_low):
    """
    使用舍入模式控制精度
    """
    pid = ct.bid(0)
    
    # 高精度计算
    tile_f32 = ct.load(arr_high, index=(pid,), shape=(16,))
    
    # 转换为低精度时使用不同舍入模式
    # APPROX：更快但精度稍低
    # FULL：更精确但可能更慢
    tile_f16_approx = ct.cast(tile_f32, ct.float16)  # 使用近似舍入
    
    ct.store(arr_low, index=(pid,), tile=tile_f16_approx)
```

---

## 填充模式 (Padding Modes)

### 基本概念

```python
class cuda.tile.PaddingMode
```

**填充模式（Padding Mode）** 指定加载操作超出边界时的填充值。

### 填充模式列表

| 模式 | 常量 | 填充值 |
|------|------|--------|
| **不确定** | `UNDETERMINED` = `'undetermined'` | 填充值未确定 |
| **零** | `ZERO` = `'zero'` | 填充值为 0 |
| **负零** | `NEG_ZERO` = `'neg_zero'` | 填充值为 -0 |
| **NaN** | `NAN` = `'nan'` | 填充值为 NaN |
| **正无穷** | `POS_INF` = `'pos_inf'` | 填充值为 +∞ |
| **负无穷** | `NEG_INF` = `'neg_inf'` | 填充值为 -∞ |

### 代码示例

#### 示例 1：边界处理

```python
import cuda.tile as ct
import cupy as cp

@ct.kernel
def padding_demo(arr, output, n: ct.Constant[int]):
    """
    演示填充模式的使用
    """
    pid = ct.bid(0)
    tile_idx = pid * 16
    
    # 当 tile_idx 接近数组边界时，可能超出范围
    # 使用 padding_mode 指定填充行为
    
    # 零填充（最常用）
    tile_zero = ct.load(arr, index=(pid,), shape=(16,), 
                       padding_mode=ct.PaddingMode.ZERO)
    
    # NaN 填充（用于浮点数，便于检测边界）
    tile_nan = ct.load(arr, index=(pid,), shape=(16,),
                      padding_mode=ct.PaddingMode.NAN)
    
    # 使用零填充的结果
    ct.store(output, index=(pid,), tile=tile_zero * 2.0)
```

#### 示例 2：图像边界填充

```python
@ct.kernel
def image_padding_demo(image, output, height: ct.Constant[int], 
                       width: ct.Constant[int]):
    """
    图像处理中的边界填充
    """
    row = ct.bid(0)
    col = ct.bid(1)
    
    # 加载图像块，边界外填充零
    # 对于图像边缘，超出部分填充为 0
    tile = ct.load(image, index=(row, col), shape=(16, 16),
                  padding_mode=ct.PaddingMode.ZERO)
    
    # 应用滤波器（边界外的零不会影响结果）
    result = tile * 0.8 + 0.1
    
    ct.store(output, index=(row, col), tile=result)
```

#### 示例 3：不同填充模式的应用

```python
@ct.kernel
def various_padding_demo(arr, out_zero, out_nan, out_inf):
    """
    不同填充模式的比较
    """
    pid = ct.bid(0)
    
    # 零填充：适用于求和、平均等操作
    tile_zero = ct.load(arr, index=(pid,), shape=(16,),
                       padding_mode=ct.PaddingMode.ZERO)
    
    # NaN 填充：适用于需要标记无效数据的场景
    tile_nan = ct.load(arr, index=(pid,), shape=(16,),
                      padding_mode=ct.PaddingMode.NAN)
    
    # 正无穷填充：适用于求最小值的操作
    tile_inf = ct.load(arr, index=(pid,), shape=(16,),
                      padding_mode=ct.PaddingMode.POS_INF)
    
    # 存储
    ct.store(out_zero, index=(pid,), tile=tile_zero)
    ct.store(out_nan, index=(pid,), tile=tile_nan)
    ct.store(out_inf, index=(pid,), tile=tile_inf)

# 使用示例
n = 100  # 不是 16 的倍数，会有边界问题
arr = cp.arange(n, dtype=cp.float32)

# 创建输出数组（大小向上取整到 16 的倍数）
out_size = ((n + 15) // 16) * 16
out_zero = cp.zeros(out_size, dtype=cp.float32)
out_nan = cp.zeros(out_size, dtype=cp.float32)
out_inf = cp.zeros(out_size, dtype=cp.float32)

grid = ((n + 15) // 16, 1, 1)
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, various_padding_demo, 
          (arr, out_zero, out_nan, out_inf))

print("原始数组大小:", n)
print("最后一个 Tile 的边界外元素数:", out_size - n)
print("零填充:", out_zero[n:n+5])
print("NaN填充:", out_nan[n:n+5])
print("正无穷填充:", out_inf[n:n+5])
```

---

## 实战示例

### 示例 1：完整的图像处理管道

```python
import cupy as cp
import cuda.tile as ct

@ct.function
def rgb_to_grayscale(r, g, b):
    """RGB 转灰度（使用标准权重）"""
    return 0.299 * r + 0.587 * g + 0.114 * b

@ct.kernel
def image_process_kernel(rgb_image, gray_output, 
                        height: ct.Constant[int],
                        width: ct.Constant[int]):
    """
    图像处理：RGB 转灰度 + 归一化
    
    rgb_image: (H, W, 3) - RGB 图像
    gray_output: (H, W) - 灰度图像
    """
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)
    
    # 加载 RGB 通道（使用零填充处理边界）
    tile_r = ct.load(rgb_image, index=(row_tile, col_tile, 0), 
                    shape=(16, 16), padding_mode=ct.PaddingMode.ZERO)
    tile_g = ct.load(rgb_image, index=(row_tile, col_tile, 1), 
                    shape=(16, 16), padding_mode=ct.PaddingMode.ZERO)
    tile_b = ct.load(rgb_image, index=(row_tile, col_tile, 2), 
                    shape=(16, 16), padding_mode=ct.PaddingMode.ZERO)
    
    # 转换为灰度（形状广播自动处理）
    tile_gray = rgb_to_grayscale(tile_r, tile_g, tile_b)
    
    # 归一化到 [0, 1]
    tile_normalized = tile_gray / 255.0
    
    # 存储
    ct.store(gray_output, index=(row_tile, col_tile), tile=tile_normalized)

# 使用示例
def process_image(image_rgb):
    """
    处理 RGB 图像
    
    image_rgb: CuPy array (H, W, 3), dtype=uint8
    返回: CuPy array (H, W), dtype=float32
    """
    H, W, C = image_rgb.shape
    assert C == 3, "需要 RGB 图像"
    
    # 转换为 float32
    image_float = image_rgb.astype(cp.float32)
    
    # 创建输出
    gray_output = cp.zeros((H, W), dtype=cp.float32)
    
    # 计算 Grid
    tile_h, tile_w = 16, 16
    grid_h = (H + tile_h - 1) // tile_h
    grid_w = (W + tile_w - 1) // tile_w
    grid = (grid_h, grid_w, 1)
    
    # 启动内核
    stream = cp.cuda.get_current_stream()
    ct.launch(stream, grid, image_process_kernel,
              (image_float, gray_output, H, W))
    
    return gray_output

# 测试
image = cp.random.randint(0, 256, (480, 640, 3), dtype=cp.uint8)
gray = process_image(image)
print(f"输入形状: {image.shape}, 输出形状: {gray.shape}")
print(f"灰度范围: [{gray.min():.3f}, {gray.max():.3f}]")
```

### 示例 2：矩阵运算与类型提升

```python
@ct.kernel
def matrix_operations(A_f16, B_f32, C_output):
    """
    混合精度矩阵运算
    
    A: float16
    B: float32
    C: float32 (output)
    """
    row = ct.bid(0)
    col = ct.bid(1)
    
    # 加载不同精度的 Tile
    tile_a = ct.load(A_f16, index=(row, col), shape=(16, 16))  # float16
    tile_b = ct.load(B_f32, index=(row, col), shape=(16, 16))  # float32
    
    # 自动类型提升：float16 + float32 → float32
    tile_sum = tile_a + tile_b
    
    # 矩阵运算
    result = tile_sum * 2.0 + 1.0
    
    # 存储为 float32
    ct.store(C_output, index=(row, col), tile=result)

# 使用
A = cp.random.rand(256, 256).astype(cp.float16)
B = cp.random.rand(256, 256).astype(cp.float32)
C = cp.zeros((256, 256), dtype=cp.float32)

grid = (256 // 16, 256 // 16, 1)
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, matrix_operations, (A, B, C))
```

### 示例 3：数据类型转换管道

```python
@ct.kernel
def precision_conversion_pipeline(input_f64, 
                                  output_f32, 
                                  output_f16, 
                                  output_i32):
    """
    精度转换管道：float64 → float32 → float16 → int32
    """
    pid = ct.bid(0)
    
    # 加载高精度数据
    tile_f64 = ct.load(input_f64, index=(pid,), shape=(16,))
    
    # 转换为 float32
    tile_f32 = ct.cast(tile_f64, ct.float32)
    ct.store(output_f32, index=(pid,), tile=tile_f32)
    
    # 转换为 float16（降低精度）
    tile_f16 = ct.cast(tile_f32, ct.float16)
    ct.store(output_f16, index=(pid,), tile=tile_f16)
    
    # 转换为整数（截断）
    tile_i32 = ct.cast(tile_f32, ct.int32)
    ct.store(output_i32, index=(pid,), tile=tile_i32)

# 测试精度损失
input_data = cp.array([1.123456789, 2.987654321, 3.141592653, 4.567890123] * 16,
                      dtype=cp.float64)
output_f32 = cp.zeros(64, dtype=cp.float32)
output_f16 = cp.zeros(64, dtype=cp.float16)
output_i32 = cp.zeros(64, dtype=cp.int32)

grid = (64 // 16, 1, 1)
stream = cp.cuda.get_current_stream()
ct.launch(stream, grid, precision_conversion_pipeline,
          (input_data, output_f32, output_f16, output_i32))

print("原始 (float64):", input_data[:4])
print("float32:      ", output_f32[:4])
print("float16:      ", output_f16[:4])
print("int32:        ", output_i32[:4])
```

---

## 总结

### 核心概念回顾

1. **数组模型**
   - 全局数组（Array）：存储在内存中，可变，全局可见
   - Tile 数组（Tile）：可能不在内存中，不可变，Block 局部

2. **空间概念**
   - 元素空间：数组元素的多维空间
   - Tile 空间：特定形状的 Tile 的多维空间

3. **形状广播**
   - 遵循 NumPy 语义
   - 尾部对齐，自动扩展

4. **类型系统**
   - 丰富的数据类型支持
   - 自动类型提升
   - 显式类型转换

5. **特殊模式**
   - 舍入模式：控制浮点精度
   - 填充模式：处理边界情况

### 最佳实践

**数据类型选择：**
- 使用 `float16` 或 `bfloat16` 节省内存和提高性能
- 关键计算使用 `float32` 保证精度
- 整数运算注意有符号/无符号混合

**Tile 形状：**
- 所有维度必须是 2 的幂
- 选择合适的 Tile 大小以平衡并行度和资源使用

**边界处理：**
- 使用适当的填充模式处理数组边界
- 零填充最常用，NaN 填充便于调试

**类型转换：**
- 优先使用自动类型提升
- 必要时使用显式转换避免精度损失
- 注意 tfloat32 和 float8 不支持隐式提升

---

**版权声明**: Copyright © 2025, NVIDIA Corporation.  
**许可证**: Apache-2.0  
**官方文档**: https://docs.nvidia.com/cuda/cutile-python/data-model.html


