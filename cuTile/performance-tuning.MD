# cuTile Python 性能调优详解

## 目录
1. [概述](#概述)
2. [架构特定配置（ByTarget）](#架构特定配置bytarget)
3. [加载/存储性能提示](#加载存储性能提示)
4. [性能调优策略](#性能调优策略)
5. [实战示例](#实战示例)
6. [最佳实践](#最佳实践)

---

## 概述

cuTile Python 提供了多种性能调优技术，帮助开发者充分发挥 GPU 硬件的潜力。主要包括：

### 核心调优手段

1. **架构特定配置**：使用 `ByTarget` 针对不同 GPU 架构定制参数
2. **内存访问提示**：通过 `latency` 和 `allow_tma` 优化内存操作
3. **内核配置优化**：调整 `num_ctas`、`occupancy` 和 `opt_level` 等参数

### 为什么需要性能调优？

不同的 GPU 架构（如 SM_90、SM_100、SM_120）具有不同的硬件特性：
- **计算单元数量**不同
- **内存层次结构**差异
- **特殊硬件加速器**（如 TMA）的支持情况

通过性能调优，可以：
- ✅ 让同一份代码在不同架构上都达到最优性能
- ✅ 避免硬件资源浪费
- ✅ 减少内存访问延迟
- ✅ 提高整体吞吐量

---

## 架构特定配置（ByTarget）

### 什么是 ByTarget？

`ByTarget` 是一个类型类，允许你为不同的 GPU 架构指定不同的配置值。

```python
class cuda.tile.ByTarget(*, default=UNSPECIFIED, **value_by_target)
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `default` | Any | 回退值，当目标架构未显式列出时使用 |
| `value_by_target` | dict | 架构到值的映射，键格式为 `"sm_<major><minor>"` |

### 支持的架构标识

| 架构标识 | 对应硬件 | 特点 |
|----------|----------|------|
| `sm_90` | H100 (Hopper) | TMA 支持，线程块集群 |
| `sm_100` | B100 系列 (Blackwell) | 增强的 TMA，更大的共享内存 |
| `sm_120` | 未来架构 | 待定特性 |

### 基础用法

#### 1. 统一配置（所有架构相同）

```python
from cuda.tile import kernel

@kernel(num_ctas=8)
def simple_kernel(x):
    """所有架构都使用 8 个 CTA"""
    pass
```

#### 2. 架构特定配置（无回退值）

```python
from cuda.tile import kernel, ByTarget

@kernel(num_ctas=ByTarget(sm_90=16, sm_100=8, sm_120=4))
def arch_specific_kernel(x):
    """
    - Hopper (sm_90): 16 个 CTA
    - Blackwell (sm_100): 8 个 CTA  
    - 未来架构 (sm_120): 4 个 CTA
    """
    pass
```

#### 3. 架构特定配置（含回退值）

```python
from cuda.tile import kernel, ByTarget

@kernel(num_ctas=ByTarget(sm_100=8, sm_120=4, default=2))
def kernel_with_fallback(x):
    """
    - Blackwell (sm_100): 8 个 CTA
    - 未来架构 (sm_120): 4 个 CTA
    - 其他所有架构: 2 个 CTA（回退值）
    """
    pass
```

### 可配置的内核参数

`ByTarget` 可用于以下所有内核配置参数：

#### 1. num_ctas（CTA 数量）

控制同时运行的协作线程数组（Cooperative Thread Arrays）数量。

```python
@kernel(num_ctas=ByTarget(sm_90=16, sm_100=8, default=4))
def matmul_kernel(A, B, C):
    pass
```

**调优建议：**
- 更新的架构通常能支持更多并发 CTA
- 平衡并发度与资源消耗（寄存器、共享内存）

#### 2. occupancy（占用率）

控制每个 SM 上活跃的线程数量。

```python
@kernel(occupancy=ByTarget(sm_90="max", sm_100=0.75, default=0.5))
def high_occupancy_kernel(x):
    pass
```

**调优建议：**
- `"max"`：最大化占用率（可能增加寄存器溢出）
- `0.5-0.75`：平衡占用率与寄存器使用
- 较低值：留更多资源给单个线程

#### 3. opt_level（优化级别）

控制编译器优化强度。

```python
@kernel(opt_level=ByTarget(sm_100=3, default=2))
def optimized_kernel(x):
    pass
```

**优化级别：**
- `0`：无优化（调试用）
- `1`：基础优化
- `2`：标准优化（默认）
- `3`：激进优化（可能增加编译时间）

### 实战案例：矩阵乘法优化

```python
import cuda.tile as ct
from cuda.tile import ByTarget

# 根据架构优化 tile 大小
TILE_M = ByTarget(sm_90=128, sm_100=256, default=64)
TILE_N = ByTarget(sm_90=128, sm_100=256, default=64)
TILE_K = 32

@ct.kernel(
    num_ctas=ByTarget(sm_90=16, sm_100=8, default=4),
    occupancy=ByTarget(sm_90="max", sm_100=0.75, default=0.5),
    opt_level=ByTarget(sm_100=3, default=2)
)
def optimized_matmul(A, B, C, M, N, K):
    """
    架构感知的矩阵乘法内核
    - sm_90: 较小 tile，高并发
    - sm_100: 更大 tile，中等并发
    - 其他: 保守配置
    """
    # 获取当前架构的 tile 大小
    tile_m = ct.config.get_value(TILE_M)
    tile_n = ct.config.get_value(TILE_N)
    
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    
    # 使用动态 tile 大小
    tile_a = ct.load(A, index=(bid_x, 0), shape=(tile_m, TILE_K))
    tile_b = ct.load(B, index=(0, bid_y), shape=(TILE_K, tile_n))
    
    tile_c = ct.matmul(tile_a, tile_b)
    ct.store(C, index=(bid_x, bid_y), tile=tile_c)
```

---

## 加载/存储性能提示

### 概述

`load()` 和 `store()` 操作支持可选的性能提示参数，帮助编译器优化内存访问。

### 参数详解

#### 1. latency（延迟提示）

**类型：** `int` 或 `None`

**取值范围：** 1（低延迟）到 10（高延迟）

**含义：** 指示此操作的 DRAM 流量密集程度

```python
tile = ct.load(
    x, 
    index=(bid,), 
    shape=(TILE_SIZE,),
    latency=8  # 提示：这是一个高延迟的内存访问
)
```

**典型值：**
| latency 值 | 场景 | 说明 |
|------------|------|------|
| 1-3 | 低延迟访问 | 数据可能在缓存中，或访问频繁 |
| 4-6 | 中等延迟 | 普通 DRAM 访问 |
| 7-10 | 高延迟访问 | 大量数据从全局内存读取 |
| None | 自动推断 | 让编译器决定（默认） |

#### 2. allow_tma（TMA 使用控制）

**类型：** `bool` 或 `None`

**含义：** 控制是否允许使用 TMA（Tensor Memory Accelerator）

```python
ct.store(
    y,
    index=(bid,),
    tile=tile_result,
    allow_tma=False  # 禁止使用 TMA
)
```

**TMA 特性：**
- ✅ **优势**：硬件加速的张量内存操作，更高带宽
- ✅ **适用**：大块连续内存传输
- ❌ **限制**：仅在支持的架构上可用（sm_90+）
- ❌ **不适用**：小规模、不规则访问模式

**取值：**
| 值 | 行为 |
|----|------|
| `True` | 允许使用 TMA（如果架构支持） |
| `False` | 强制不使用 TMA |
| `None` | 默认行为（通常允许 TMA） |

### 性能提示策略

#### 策略 1：根据数据大小调整

```python
import cuda.tile as ct

def adaptive_load(data, index, shape, data_size_mb):
    """根据数据大小自适应选择 latency"""
    if data_size_mb < 1:
        latency_hint = 2  # 小数据，可能在缓存中
    elif data_size_mb < 10:
        latency_hint = 5  # 中等数据
    else:
        latency_hint = 9  # 大数据，高延迟
    
    return ct.load(data, index=index, shape=shape, latency=latency_hint)
```

#### 策略 2：访问模式优化

```python
@ct.kernel
def pattern_aware_kernel(input_data, temp_data, output_data):
    bid = ct.bid(0)
    
    # 策略 1：首次从全局内存加载 - 高延迟
    tile_input = ct.load(
        input_data,
        index=(bid,),
        shape=(256,),
        latency=8,        # 预期高延迟
        allow_tma=True    # 允许 TMA 加速
    )
    
    # 中间计算...
    tile_temp = ct.some_operation(tile_input)
    
    # 策略 2：写入临时数据 - 中等延迟
    ct.store(
        temp_data,
        index=(bid,),
        tile=tile_temp,
        latency=5,        # 中等延迟
        allow_tma=False   # 小数据不用 TMA
    )
    
    # 策略 3：最终结果写回 - 低延迟（可能被缓存）
    ct.store(
        output_data,
        index=(bid,),
        tile=tile_temp,
        latency=3,        # 较低延迟
        allow_tma=True
    )
```

#### 策略 3：架构感知的 TMA 使用

```python
from cuda.tile import ByTarget

@ct.kernel
def architecture_aware_tma_kernel(x, y):
    bid = ct.bid(0)
    
    # 在支持 TMA 的架构上使用
    use_tma = ct.config.get_compute_capability() >= 90  # sm_90+
    
    tile_x = ct.load(
        x,
        index=(bid,),
        shape=(1024,),
        latency=7,
        allow_tma=use_tma  # 动态决定
    )
    
    # 处理...
    ct.store(y, index=(bid,), tile=tile_x, allow_tma=use_tma)
```

### 完整示例：带性能提示的内核

```python
import cuda.tile as ct

TILE_SIZE = 16

@ct.kernel
def load_store_with_hints_kernel(x, y):
    """演示性能提示的完整使用"""
    bid = ct.bid(0)
    
    # 高延迟 DRAM 加载
    tx = ct.load(
        x,
        index=(bid,),
        shape=(TILE_SIZE,),
        latency=8,        # 提示：高延迟访问
        allow_tma=True    # 允许 TMA 加速
    )
    
    # 可能的处理...
    tx = tx * 2.0
    
    # 较低延迟写入
    ct.store(
        y,
        index=(bid,),
        tile=tx,
        latency=2,        # 提示：较便宜的写入
        allow_tma=False   # 禁用 TMA（小数据）
    )

# 使用示例
import numpy as np
from cuda import Device

# 创建测试数据
size = 1024
x_host = np.random.randn(size).astype(np.float32)
y_host = np.zeros(size, dtype=np.float32)

# 分配设备内存
x_device = Device.mem_alloc(x_host.nbytes)
y_device = Device.mem_alloc(y_host.nbytes)

# 拷贝数据
Device.copy_htod(x_device, x_host)

# 启动内核
num_blocks = size // TILE_SIZE
load_store_with_hints_kernel[num_blocks](x_device, y_device)

# 拷贝结果
Device.copy_dtoh(y_host, y_device)
```

---

## 性能调优策略

### 调优流程

```
1. 基准测试
   ↓
2. 识别瓶颈
   ↓
3. 应用优化
   ↓
4. 验证效果
   ↓
5. 迭代改进
```

### 策略矩阵

| 瓶颈类型 | 优化手段 | ByTarget 配置 | 内存提示 |
|----------|----------|---------------|----------|
| 计算受限 | 提高占用率 | `occupancy="max"` | - |
| 内存受限 | TMA + latency 优化 | - | `latency=8, allow_tma=True` |
| 并发度不足 | 增加 CTA | `num_ctas` 增大 | - |
| 寄存器溢出 | 降低占用率 | `occupancy=0.5` | - |

### 常见优化模式

#### 模式 1：内存密集型应用

```python
@ct.kernel(
    num_ctas=ByTarget(sm_90=8, sm_100=4, default=2),
    occupancy=ByTarget(sm_90=0.5, default=0.25),  # 降低占用率
    opt_level=3
)
def memory_intensive_kernel(large_array, result):
    """
    内存密集型：
    - 较少 CTA（减少内存竞争）
    - 较低占用率（更多缓存/寄存器）
    - 高优化级别
    """
    bid = ct.bid(0)
    
    # 大块数据加载
    tile = ct.load(
        large_array,
        index=(bid,),
        shape=(4096,),
        latency=9,        # 高延迟
        allow_tma=True    # 启用 TMA
    )
    
    # 处理...
    processed = ct.reduce_sum(tile)
    
    # 写回小结果
    ct.store(
        result,
        index=(bid,),
        tile=processed,
        latency=1,
        allow_tma=False
    )
```

#### 模式 2：计算密集型应用

```python
@ct.kernel(
    num_ctas=ByTarget(sm_90=32, sm_100=16, default=8),
    occupancy="max",  # 最大化占用率
    opt_level=ByTarget(sm_100=3, default=2)
)
def compute_intensive_kernel(data, result):
    """
    计算密集型：
    - 更多 CTA（提高并发）
    - 最大占用率
    - 较小的内存访问
    """
    bid = ct.bid(0)
    
    # 小数据加载
    tile = ct.load(
        data,
        index=(bid,),
        shape=(128,),
        latency=3,        # 低延迟（数据小）
        allow_tma=False   # 不需要 TMA
    )
    
    # 大量计算
    for _ in range(100):
        tile = ct.complex_computation(tile)
    
    ct.store(result, index=(bid,), tile=tile, latency=2)
```

#### 模式 3：混合负载

```python
@ct.kernel(
    num_ctas=ByTarget(sm_90=16, sm_100=12, default=6),
    occupancy=ByTarget(sm_90=0.75, sm_100=0.5, default=0.5),
    opt_level=2
)
def mixed_workload_kernel(input_data, weights, output):
    """
    混合负载：平衡内存和计算
    """
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    
    # 输入数据 - 中高延迟
    tile_input = ct.load(
        input_data,
        index=(bid_x,),
        shape=(512,),
        latency=6,
        allow_tma=True
    )
    
    # 权重数据 - 可能被缓存
    tile_weights = ct.load(
        weights,
        index=(bid_y,),
        shape=(512,),
        latency=3,  # 假设权重会被重用
        allow_tma=False
    )
    
    # 计算
    result = ct.dot(tile_input, tile_weights)
    
    # 输出
    ct.store(
        output,
        index=(bid_x, bid_y),
        tile=result,
        latency=4,
        allow_tma=True
    )
```

---

## 实战示例

### 示例 1：向量加法优化

```python
import cuda.tile as ct
from cuda.tile import ByTarget
import numpy as np

# 架构感知的 tile 大小
TILE_SIZE = ByTarget(sm_90=256, sm_100=512, default=128)

@ct.kernel(
    num_ctas=ByTarget(sm_90=16, sm_100=8, default=4),
    occupancy=ByTarget(sm_90="max", default=0.75)
)
def optimized_vector_add(a, b, c, n):
    """优化的向量加法"""
    bid = ct.bid(0)
    tile_size = ct.config.get_value(TILE_SIZE)
    
    # 计算当前块的索引范围
    start_idx = bid * tile_size
    
    # 加载输入 - 并行内存访问
    tile_a = ct.load(
        a,
        index=(start_idx,),
        shape=(tile_size,),
        latency=5,
        allow_tma=True
    )
    
    tile_b = ct.load(
        b,
        index=(start_idx,),
        shape=(tile_size,),
        latency=5,
        allow_tma=True
    )
    
    # 计算
    tile_c = tile_a + tile_b
    
    # 存储结果
    ct.store(
        c,
        index=(start_idx,),
        tile=tile_c,
        latency=3,
        allow_tma=True
    )

# 性能测试
def benchmark_vector_add():
    n = 1024 * 1024  # 1M 元素
    
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    
    # 启动优化内核
    # num_blocks 会根据架构自动调整
    optimized_vector_add[n // 256](a, b, c, n)
    
    return c
```

### 示例 2：矩阵转置优化

```python
import cuda.tile as ct
from cuda.tile import ByTarget

@ct.kernel(
    num_ctas=ByTarget(sm_90=16, sm_100=12, default=8),
    occupancy=0.75,
    opt_level=ByTarget(sm_100=3, default=2)
)
def optimized_transpose(input_matrix, output_matrix, rows, cols):
    """
    优化的矩阵转置
    - 使用共享内存避免 bank conflicts
    - 架构特定的 tile 大小
    """
    TILE_DIM = ByTarget(sm_90=32, sm_100=64, default=16)
    
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    tid_x = ct.tid(0)
    tid_y = ct.tid(1)
    
    tile_dim = ct.config.get_value(TILE_DIM)
    
    # 读取输入块 - 连续访问
    x_index = bid_x * tile_dim + tid_x
    y_index = bid_y * tile_dim + tid_y
    
    # 高延迟加载（大矩阵）
    tile = ct.load(
        input_matrix,
        index=(x_index, y_index),
        shape=(tile_dim, tile_dim),
        latency=7,
        allow_tma=True  # 大块传输用 TMA
    )
    
    # 转置 tile
    tile_transposed = ct.transpose(tile)
    
    # 写入输出 - 交换坐标
    out_x = bid_y * tile_dim + tid_x
    out_y = bid_x * tile_dim + tid_y
    
    ct.store(
        output_matrix,
        index=(out_x, out_y),
        tile=tile_transposed,
        latency=6,
        allow_tma=True
    )
```

### 示例 3：归约（Reduction）优化

```python
import cuda.tile as ct
from cuda.tile import ByTarget

@ct.kernel(
    num_ctas=ByTarget(sm_90=64, sm_100=32, default=16),
    occupancy="max",  # 归约受益于高占用率
    opt_level=3
)
def optimized_reduction(input_data, output, n):
    """
    优化的全局归约
    - 多阶段归约
    - 充分利用共享内存
    """
    BLOCK_SIZE = ByTarget(sm_90=512, sm_100=1024, default=256)
    
    bid = ct.bid(0)
    tid = ct.tid(0)
    block_size = ct.config.get_value(BLOCK_SIZE)
    
    # 第一阶段：加载并进行块内归约
    idx = bid * block_size + tid
    
    # 加载数据 - 跨度访问
    tile = ct.load(
        input_data,
        index=(idx,),
        shape=(block_size,),
        latency=8,        # 大量数据
        allow_tma=True
    )
    
    # 块内归约
    partial_sum = ct.reduce_sum(tile)
    
    # 使用原子操作累加到全局结果
    if tid == 0:
        # 写入部分结果 - 低延迟（小数据）
        ct.atomic_add(
            output,
            index=(0,),
            value=partial_sum
        )
```

### 示例 4：卷积操作优化

```python
import cuda.tile as ct
from cuda.tile import ByTarget

@ct.kernel(
    num_ctas=ByTarget(sm_90=24, sm_100=16, default=8),
    occupancy=ByTarget(sm_90=0.5, sm_100=0.75, default=0.5),
    opt_level=3
)
def optimized_conv2d(input_img, kernel, output, 
                     in_h, in_w, k_h, k_w, out_h, out_w):
    """
    优化的 2D 卷积
    - 重用输入数据
    - TMA 加载大块数据
    - 权重缓存优化
    """
    TILE_H = ByTarget(sm_90=16, sm_100=32, default=8)
    TILE_W = ByTarget(sm_90=16, sm_100=32, default=8)
    
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    
    tile_h = ct.config.get_value(TILE_H)
    tile_w = ct.config.get_value(TILE_W)
    
    # 加载输入 tile（包含 halo 区域）
    in_tile = ct.load(
        input_img,
        index=(bid_x * tile_h, bid_y * tile_w),
        shape=(tile_h + k_h - 1, tile_w + k_w - 1),
        latency=7,        # 图像数据通常较大
        allow_tma=True    # 连续内存，适合 TMA
    )
    
    # 加载卷积核 - 小数据，可能被缓存
    kernel_tile = ct.load(
        kernel,
        index=(0, 0),
        shape=(k_h, k_w),
        latency=2,        # 小内核，可能在缓存中
        allow_tma=False   # 太小，不需要 TMA
    )
    
    # 执行卷积计算
    out_tile = ct.conv2d(in_tile, kernel_tile)
    
    # 存储输出
    ct.store(
        output,
        index=(bid_x * tile_h, bid_y * tile_w),
        tile=out_tile,
        latency=5,
        allow_tma=True
    )
```

---

## 最佳实践

### 1. 架构配置最佳实践

#### ✅ 推荐做法

```python
# 1. 始终提供 default 值，确保向后兼容
@ct.kernel(num_ctas=ByTarget(sm_100=8, default=4))
def safe_kernel(x):
    pass

# 2. 为关键架构显式配置
@ct.kernel(
    num_ctas=ByTarget(sm_90=16, sm_100=8, default=4),
    occupancy=ByTarget(sm_90="max", sm_100=0.75, default=0.5)
)
def well_configured_kernel(x):
    pass

# 3. 使用配置常量，便于维护
SM90_CTAS = 16
SM100_CTAS = 8
DEFAULT_CTAS = 4

@ct.kernel(num_ctas=ByTarget(sm_90=SM90_CTAS, sm_100=SM100_CTAS, default=DEFAULT_CTAS))
def maintainable_kernel(x):
    pass
```

#### ❌ 避免的做法

```python
# 1. 不提供 default，可能在新架构上失败
@ct.kernel(num_ctas=ByTarget(sm_90=8))  # 危险！
def unsafe_kernel(x):
    pass

# 2. 所有架构使用相同配置（失去了 ByTarget 的意义）
@ct.kernel(num_ctas=ByTarget(sm_90=8, sm_100=8, sm_120=8, default=8))
def pointless_kernel(x):
    pass

# 3. 硬编码魔数
@ct.kernel(num_ctas=ByTarget(sm_90=16, sm_100=8, default=4))
@ct.kernel(num_ctas=ByTarget(sm_90=16, sm_100=8, default=4))  # 重复配置
def redundant_kernels(x):
    pass
```

### 2. 内存访问最佳实践

#### ✅ 推荐做法

```python
@ct.kernel
def good_memory_patterns(large_data, small_data, output):
    bid = ct.bid(0)
    
    # 1. 大数据：高 latency + TMA
    large_tile = ct.load(
        large_data,
        index=(bid,),
        shape=(4096,),
        latency=8,
        allow_tma=True
    )
    
    # 2. 小数据/频繁访问：低 latency，不用 TMA
    small_tile = ct.load(
        small_data,
        index=(bid,),
        shape=(16,),
        latency=2,
        allow_tma=False
    )
    
    # 3. 根据访问模式调整
    result = large_tile[:16] * small_tile
    
    ct.store(
        output,
        index=(bid,),
        tile=result,
        latency=3,
        allow_tma=False  # 小结果
    )
```

#### ❌ 避免的做法

```python
@ct.kernel
def bad_memory_patterns(data, output):
    bid = ct.bid(0)
    
    # 1. 大数据不使用 TMA（浪费性能）
    tile = ct.load(
        data,
        index=(bid,),
        shape=(4096,),
        allow_tma=False  # 不好！
    )
    
    # 2. 所有操作使用相同的 latency（不精确）
    ct.store(output, index=(bid,), tile=tile, latency=5)  # 太笼统
    
    # 3. 小数据使用 TMA（开销大于收益）
    small = ct.load(
        data,
        index=(bid,),
        shape=(4,),
        allow_tma=True  # 浪费！
    )
```

### 3. 性能调优检查清单

#### 调优前

- [ ] 建立基准性能（未优化版本）
- [ ] 识别瓶颈（计算 vs 内存）
- [ ] 了解目标架构特性
- [ ] 分析数据访问模式

#### 调优中

- [ ] 应用架构特定配置（ByTarget）
- [ ] 设置合适的 latency 提示
- [ ] 决定是否使用 TMA
- [ ] 调整 occupancy 和 num_ctas
- [ ] 选择合适的优化级别

#### 调优后

- [ ] 测量性能提升
- [ ] 验证结果正确性
- [ ] 在多个架构上测试
- [ ] 文档化配置选择的原因

### 4. 调试技巧

```python
import cuda.tile as ct
from cuda.tile import ByTarget

# 技巧 1：使用配置日志
@ct.kernel(
    num_ctas=ByTarget(sm_90=16, sm_100=8, default=4),
    debug_info=True  # 启用调试信息
)
def debug_kernel(x):
    # 运行时打印配置
    if ct.tid(0) == 0 and ct.bid(0) == 0:
        print(f"Running on architecture: {ct.config.get_arch()}")
        print(f"Number of CTAs: {ct.config.get_num_ctas()}")
    pass

# 技巧 2：性能计数器
@ct.kernel
def profiled_kernel(x, y):
    bid = ct.bid(0)
    
    # 记录加载时间
    start = ct.clock64()
    
    tile = ct.load(
        x,
        index=(bid,),
        shape=(256,),
        latency=5,
        allow_tma=True
    )
    
    load_time = ct.clock64() - start
    
    # 可以将性能数据写回
    # ...

# 技巧 3：渐进式优化
def progressive_optimization():
    """从简单到复杂，逐步优化"""
    
    # 阶段 1：基础版本
    @ct.kernel
    def v1_basic(x, y):
        pass
    
    # 阶段 2：添加架构配置
    @ct.kernel(num_ctas=ByTarget(sm_90=8, default=4))
    def v2_arch_aware(x, y):
        pass
    
    # 阶段 3：添加内存提示
    @ct.kernel(num_ctas=ByTarget(sm_90=8, default=4))
    def v3_memory_hints(x, y):
        # 添加 latency 和 allow_tma
        pass
    
    # 阶段 4：全面优化
    @ct.kernel(
        num_ctas=ByTarget(sm_90=16, sm_100=8, default=4),
        occupancy=ByTarget(sm_90="max", default=0.75),
        opt_level=3
    )
    def v4_fully_optimized(x, y):
        # 完整的性能提示
        pass
    
    # 在每个阶段测量性能
```

### 5. 常见陷阱

#### 陷阱 1：过度优化

```python
# ❌ 不好：过于复杂，难以维护
@ct.kernel(
    num_ctas=ByTarget(
        sm_80=32, sm_86=28, sm_87=30, sm_89=26,
        sm_90=24, sm_100=20, sm_120=16, default=8
    ),
    occupancy=ByTarget(
        sm_80=0.9, sm_86=0.85, sm_87=0.87, sm_89=0.83,
        sm_90=0.8, sm_100=0.75, sm_120=0.7, default=0.5
    )
)
def over_optimized_kernel(x):
    pass

# ✅ 好：简洁，关注主要架构
@ct.kernel(
    num_ctas=ByTarget(sm_90=24, sm_100=20, default=8),
    occupancy=ByTarget(sm_90=0.8, sm_100=0.75, default=0.5)
)
def well_balanced_kernel(x):
    pass
```

#### 陷阱 2：忽略数据大小

```python
# ❌ 不好：小数据使用高延迟提示
def process_small_array(data):
    tile = ct.load(
        data,
        index=(0,),
        shape=(16,),  # 很小
        latency=9,    # 不合适！
        allow_tma=True  # 浪费
    )

# ✅ 好：根据数据大小调整
def process_array_adaptively(data, size):
    if size < 64:
        latency_hint = 2
        use_tma = False
    elif size < 1024:
        latency_hint = 5
        use_tma = False
    else:
        latency_hint = 8
        use_tma = True
    
    tile = ct.load(
        data,
        index=(0,),
        shape=(size,),
        latency=latency_hint,
        allow_tma=use_tma
    )
```

#### 陷阱 3：不测试多架构

```python
# ✅ 好：多架构测试框架
def test_across_architectures():
    import cuda.tile as ct
    
    @ct.kernel(num_ctas=ByTarget(sm_90=16, sm_100=8, default=4))
    def kernel(x, y):
        pass
    
    # 模拟不同架构
    architectures = ["sm_90", "sm_100", "sm_80"]
    
    for arch in architectures:
        print(f"Testing on {arch}...")
        # 设置目标架构
        ct.config.set_target_arch(arch)
        
        # 运行测试
        test_data = create_test_data()
        result = kernel(test_data)
        
        # 验证结果
        assert verify_result(result), f"Failed on {arch}"
        
        print(f"✓ {arch} passed")
```

### 6. 性能测量模板

```python
import time
import numpy as np
import cuda.tile as ct
from cuda.tile import ByTarget

def benchmark_kernel(kernel_fn, input_data, num_iterations=100):
    """性能测量通用模板"""
    
    # 预热
    for _ in range(10):
        kernel_fn(input_data)
    
    # 同步
    ct.device_synchronize()
    
    # 计时
    start = time.perf_counter()
    
    for _ in range(num_iterations):
        kernel_fn(input_data)
    
    ct.device_synchronize()
    end = time.perf_counter()
    
    # 计算统计
    avg_time_ms = (end - start) / num_iterations * 1000
    
    # 计算带宽（如果适用）
    data_size_gb = input_data.nbytes / 1e9
    bandwidth_gb_s = data_size_gb / (avg_time_ms / 1000)
    
    return {
        'avg_time_ms': avg_time_ms,
        'bandwidth_gb_s': bandwidth_gb_s,
        'iterations': num_iterations
    }

# 使用示例
@ct.kernel(num_ctas=ByTarget(sm_90=16, sm_100=8, default=4))
def my_optimized_kernel(x):
    pass

data = np.random.randn(1024 * 1024).astype(np.float32)
results = benchmark_kernel(my_optimized_kernel, data)

print(f"Average time: {results['avg_time_ms']:.3f} ms")
print(f"Bandwidth: {results['bandwidth_gb_s']:.2f} GB/s")
```

---

## 总结

### 关键要点

1. **ByTarget** 是实现架构感知优化的核心工具
2. **latency** 和 **allow_tma** 帮助编译器优化内存访问
3. **性能调优是迭代过程**：测量 → 优化 → 验证
4. **平衡是关键**：不要过度优化，保持代码可维护性

### 优化优先级

```
1. 确保正确性（首要）
   ↓
2. 识别主要瓶颈
   ↓
3. 应用架构特定配置
   ↓
4. 优化内存访问模式
   ↓
5. 微调细节参数
```

### 学习路径

1. **入门阶段**：理解 `ByTarget` 基础用法
2. **进阶阶段**：掌握 `latency` 和 `allow_tma` 的使用
3. **高级阶段**：综合运用所有优化技术
4. **专家阶段**：针对特定应用深度定制

### 进一步学习

- 📖 [cuTile 执行模型](execution-model.MD) - 了解线程和块的组织
- 📖 [cuTile 内存模型](memory-model.MD) - 深入理解内存同步
- 📖 [cuTile 数据模型](data-model.MD) - 掌握数据布局优化
- 📖 [cuTile 快速入门](quick-start.MD) - 基础概念回顾

---

**最后更新：** 2025-12-29

