# cuTile Python 快速入门教程

## 目录
1. [简介](#简介)
2. [前置要求](#前置要求)
3. [安装步骤](#安装步骤)
4. [核心概念](#核心概念)
5. [示例代码详解](#示例代码详解)
6. [运行与测试](#运行与测试)
7. [性能分析工具](#性能分析工具)

---

## 简介

cuTile Python 是 NVIDIA 推出的基于 Tile（数据块）编程模型的 CUDA Python 框架。它提供了一种更高层次的抽象，让开发者能够以数据块为单位进行并行计算，而不是传统的线程级编程。

**cuTile 编程的典型流程：**
1. 从 GPU 内存加载一个或多个 Tile（数据块）
2. 对 Tile 执行计算操作，生成新的 Tile
3. 将结果 Tile 写回 GPU 内存

---

## 前置要求

在开始使用 cuTile Python 之前，请确保您的系统满足以下要求：

### 硬件要求
- **操作系统**: Linux x86_64、Linux aarch64 或 Windows x86_64
- **GPU**: 计算能力 10.x 或 12.x 的 NVIDIA GPU
- **驱动**: NVIDIA Driver r580 或更高版本

### 软件要求
- **CUDA Toolkit**: 13.1 或更高版本
- **Python**: 3.10、3.11、3.12 或 3.13

---

## 安装步骤

### 1. 安装 cuTile Python

使用 pip 安装 cuTile Python：

```bash
pip install cuda-tile
```

### 2. 安装依赖包

#### 安装 CuPy（快速入门示例需要）

```bash
pip install cupy-cuda13x
```

#### 安装其他常用包（可选）

如果您想运行更多示例，可以安装以下包：

```bash
# 安装 PyTorch（请参考 https://pytorch.org/get-started/locally/）
pip install torch

# 安装 pytest 和 numpy
pip install pytest numpy
```

---

## 核心概念

### 1. Tile（数据块）
Tile 是 cuTile 中的基本数据单元，代表一块连续的数据。在并行计算中，通过将大数据划分为多个 Tile，每个处理器处理一个 Tile，从而实现高效的并行处理。

### 2. 内核函数（Kernel）
使用 `@ct.kernel` 装饰器定义的 GPU 计算函数。

### 3. 基本操作

#### `ct.load(array, index, shape)`
从 GPU 内存加载数据到 Tile：
- **`array`**: 源数组
- **`index`**: Tile 索引（不是元素索引！）指定加载第几个数据块
- **`shape`**: Tile 的形状，指定每个数据块的大小

#### `ct.store(array, index, tile)`
将 Tile 数据存储到 GPU 内存：
- **`array`**: 目标数组
- **`index`**: Tile 索引，指定存储到第几个数据块的位置
- **`tile`**: 要存储的 Tile 数据

#### `ct.bid(dim)`
获取当前处理器在指定维度上的索引（Block ID）：
- **`dim`**: 维度（0=x, 1=y, 2=z）

#### `ct.launch(stream, grid, kernel, args)`
启动 GPU 内核函数：
- **`stream`**: CUDA 流
- **`grid`**: 网格大小（处理器数量）
- **`kernel`**: 内核函数
- **`args`**: 参数元组

---

## 示例代码详解

下面我们通过一个向量加法的例子来学习 cuTile 编程：

### 完整代码

```python
# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
向量加法示例
演示如何对向量执行逐元素操作
"""

import cupy as cp
import numpy as np
import cuda.tile as ct


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    """
    向量加法内核函数
    
    参数:
        a: 输入向量 A
        b: 输入向量 B
        c: 输出向量 C
        tile_size: 每个 tile 的大小（编译时常量）
    """
    # 获取当前处理器的 1D 索引
    pid = ct.bid(0)

    # 从全局内存加载 tile
    # index=(pid,) 指定从哪个位置加载
    # shape=(tile_size,) 指定 tile 的形状
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # 执行逐元素加法
    result = a_tile + b_tile

    # 将结果存储回全局内存
    ct.store(c, index=(pid,), tile=result)


def test():
    """测试函数"""
    # 创建输入数据
    vector_size = 2**12  # 4096 个元素
    tile_size = 2**4     # 每个 tile 16 个元素
    
    # 计算需要的网格大小（向上取整）
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)  # (256, 1, 1)

    # 创建测试数据（使用 CuPy 在 GPU 上创建）
    a = cp.random.uniform(-1, 1, vector_size)
    b = cp.random.uniform(-1, 1, vector_size)
    c = cp.zeros_like(a)

    # 启动内核
    ct.launch(
        cp.cuda.get_current_stream(),  # CUDA 流
        grid,                           # 1D 处理器网格
        vector_add,                     # 内核函数
        (a, b, c, tile_size)           # 参数
    )

    # 将结果复制到 CPU 进行验证
    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)

    # 验证结果
    expected = a_np + b_np
    np.testing.assert_array_almost_equal(c_np, expected)

    print("✓ vector_add_example passed!")


if __name__ == "__main__":
    test()
```

### 代码解析

#### 1. 内核函数定义

```python
@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
```

- `@ct.kernel` 装饰器将 Python 函数标记为 GPU 内核
- `tile_size: ct.Constant[int]` 表示这是一个编译时常量

#### 2. 获取处理器索引

```python
pid = ct.bid(0)
```

- `ct.bid(0)` 获取当前块在第 0 维（x 维度）的索引
- 每个块处理向量的一个 tile

#### 3. 加载数据到 Tile

```python
a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
```

**参数详解：**
- `index=(pid,)`: 指定要加载的 **Tile 索引**（第几个数据块）
- `shape=(tile_size,)`: 指定每个 Tile 的形状（1D，大小为 tile_size）

**关键理解：** `index` 不是元素索引，而是 **Tile 索引**！

让我们通过具体例子理解：

```
假设：
- vector_size = 16（向量总长度）
- tile_size = 4（每个 Tile 包含 4 个元素）
- grid = (4, 1, 1)（需要 4 个处理器）

向量 a 的内存布局：
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │13 │14 │15 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
  └─ Tile 0 ─┘   └─ Tile 1 ─┘   └─ Tile 2 ─┘   └─ Tile 3 ─┘

处理器分配：
- 处理器 0: pid=0 → 加载 index=(0,) → 加载元素 [0:4]   → Tile 0
- 处理器 1: pid=1 → 加载 index=(1,) → 加载元素 [4:8]   → Tile 1
- 处理器 2: pid=2 → 加载 index=(2,) → 加载元素 [8:12]  → Tile 2
- 处理器 3: pid=3 → 加载 index=(3,) → 加载元素 [12:16] → Tile 3
```

**实际内存地址计算：**
```
起始地址 = base_address + (index[0] * shape[0]) * element_size
         = base_address + (pid * tile_size) * element_size
```

**例如：** 当 `pid=2`, `tile_size=4` 时：
- `index=(2,)` 表示加载第 2 个 Tile
- 实际加载的元素范围：`[2*4 : 2*4+4]` = `[8:12]`

#### 4. Tile 计算

```python
result = a_tile + b_tile
```

- 直接使用 Python 运算符进行 tile 级别的操作
- 运算是逐元素进行的

#### 5. 存储结果

```python
ct.store(c, index=(pid,), tile=result)
```

- 将结果 tile 写回全局内存的对应位置

#### 6. 启动内核

```python
ct.launch(
    cp.cuda.get_current_stream(),  # CUDA 流
    grid,                           # 网格大小
    vector_add,                     # 内核函数
    (a, b, c, tile_size)           # 参数元组
)
```

---

## 运行与测试

### 运行单个示例

将代码保存为 `vector_add.py`，然后运行：

```bash
python3 vector_add.py
```

**预期输出：**
```
✓ vector_add_example passed!
```

### 运行多个示例

如果您有 samples 目录，可以使用以下方式运行：

```bash
# 运行单个示例
python3 samples/FFT.py

# 使用 pytest 运行所有测试
pytest samples
```

**pytest 输出示例：**
```
========================= test session starts =========================
platform linux -- Python 3.12.3, pytest-9.0.1, pluggy-1.6.0
rootdir: /home/user/cutile-python
configfile: pytest.ini
collected 6 items

samples/test_samples.py ......                                  [100%]

========================= 6 passed in 30.74s ==========================
```

---

## 性能分析工具

### 使用 NVIDIA Nsight Compute 进行性能分析

NVIDIA Nsight Compute 可以像分析传统 CUDA 内核一样分析 cuTile Python 内核。

#### 1. 生成性能分析文件

```bash
ncu -o VecAddProfile --set detailed python3 vector_add.py
```

**参数说明：**
- `-o VecAddProfile`: 指定输出文件名
- `--set detailed`: 收集详细的性能指标

#### 2. 查看分析结果

使用 Nsight Compute GUI 打开生成的 `VecAddProfile.ncu-rep` 文件，可以查看：
- 内核执行时间
- 内存带宽利用率
- 计算单元利用率
- 详细的性能指标

**注意：** 收集 cuTile Python 内核的详细统计信息需要 NVIDIA Driver r590 或更高版本。

---

## 常见问题

### Q1: `ct.load()` 中的 `index` 和实际内存地址是什么关系？

**简单记忆：**
- `index` = **Tile 的编号**（第几个数据块）
- 实际内存起始位置 = `index[0] * shape[0]`

**示例：**
```python
# 如果 tile_size = 16
pid = 5  # 第 5 个处理器
a_tile = ct.load(a, index=(5,), shape=(16,))
# 实际加载：a[5*16 : 5*16+16] = a[80:96]
```

### Q2: 如何选择合适的 tile_size？

**建议：**
- 通常选择 2 的幂次方（如 16, 32, 64, 128）
- 考虑 GPU 的共享内存大小
- 通过性能测试找到最优值

### Q3: cuTile 与传统 CUDA 编程有什么区别？

**主要区别：**
| 特性 | 传统 CUDA | cuTile |
|------|-----------|--------|
| 编程粒度 | 线程级 | Tile（块）级 |
| 抽象层次 | 低层 | 高层 |
| 代码复杂度 | 较高 | 较低 |
| 性能优化 | 手动 | 部分自动 |

### Q4: 支持哪些操作？

cuTile 支持常见的数组操作：
- 算术运算：`+`, `-`, `*`, `/`
- 比较运算：`>`, `<`, `==` 等
- 数学函数：`sin`, `cos`, `exp`, `log` 等
- Tile 转换和重塑操作

---

## 下一步学习

1. **执行模型**: 了解 cuTile 的并行执行机制
2. **数据模型**: 深入学习 Tile 的数据结构
3. **内存模型**: 掌握内存层次和优化技巧
4. **互操作性**: 学习如何与 PyTorch、NumPy 等库集成
5. **性能调优**: 掌握高级性能优化技术

---

## 参考资源

- [cuTile Python 官方文档](https://docs.nvidia.com/cuda/cutile-python/)
- [CUDA Toolkit 下载](https://developer.nvidia.com/cuda-downloads)
- [PyTorch 官方网站](https://pytorch.org/)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)

---

## 总结

通过本教程，您已经学会了：
- ✅ 安装和配置 cuTile Python 环境
- ✅ 理解 cuTile 的核心概念和编程模型
- ✅ 编写和运行第一个 cuTile 内核（向量加法）
- ✅ 使用性能分析工具优化代码

cuTile Python 为 GPU 编程提供了一个更高层次的抽象，让您能够专注于算法逻辑而不是底层细节。继续探索更多高级特性，充分发挥 GPU 的强大计算能力！

---

**版权声明**: Copyright © 2025, NVIDIA Corporation.
**许可证**: Apache-2.0

