# LMCache 适配 NPU 和国产 AI 加速器完全指南

## 目录
1. [概述](#概述)
2. [NPU/国产加速器生态](#npu国产加速器生态)
3. [Kernel 层适配](#kernel-层适配)
4. [通信层适配](#通信层适配)
5. [完整适配案例](#完整适配案例)
6. [性能优化建议](#性能优化建议)
7. [常见问题](#常见问题)

---

## 概述

### 为什么需要适配？

LMCache 原生基于 **CUDA/HIP** 实现，使用了大量 NVIDIA/AMD 特定的 API 和编程模型。国产 AI 加速器（NPU）有自己的：
- **编程模型**：AscendC、Bang C 等，不兼容 CUDA
- **运行时 API**：不同于 CUDA Runtime
- **通信库**：HCCL、CNCL 等，不同于 NCCL
- **内存模型**：不同的内存层次结构

### 适配工作量评估

| 层级 | 工作量 | 难度 | 说明 |
|-----|--------|------|------|
| Kernel 重写 | ⭐⭐⭐⭐⭐ | 高 | 需要深入了解目标硬件 |
| 通信层适配 | ⭐⭐⭐⭐ | 中高 | 替换 NCCL/NIXL |
| Python 接口 | ⭐⭐ | 低 | 主要是条件编译 |
| 构建系统 | ⭐⭐⭐ | 中 | 修改 setup.py |

### 需要适配的核心模块

```
LMCache 核心组件
├── csrc/                         # 需要完全重写
│   ├── mem_kernels.cu           # 内存操作 Kernel
│   ├── ac_enc.cu                # 算术编码 Kernel
│   ├── ac_dec.cu                # 算术解码 Kernel
│   ├── pos_kernels.cu           # 位置编码 Kernel
│   └── pybind.cpp               # Python 绑定层
│
├── 通信层（需要替换）
│   ├── P2P 通信                  # NIXL → HCCL/CNCL
│   └── 集群通信                  # NCCL → HCCL/CNCL
│
└── 构建系统
    └── setup.py                  # 添加 NPU 支持
```

---

## NPU/国产加速器生态

### 主流国产 AI 加速器对比

| 厂商 | 产品 | 编程模型 | 通信库 | PyTorch 支持 | 成熟度 |
|-----|------|---------|--------|-------------|--------|
| 华为 | 昇腾 (Ascend) | AscendC / CANN | HCCL | PyTorch + Ascend | ⭐⭐⭐⭐⭐ |
| 寒武纪 | 思元 (MLU) | Bang C / CNRT | CNCL | PyTorch + Cambricon | ⭐⭐⭐⭐ |
| 海光 | DCU | HIP (类 CUDA) | RCCL | PyTorch + ROCm | ⭐⭐⭐⭐ |
| 壁仞 | BR100/BR200 | - | - | PyTorch | ⭐⭐⭐ |
| 燧原 | 云燧 | - | - | PyTorch | ⭐⭐⭐ |

### 各平台特点

#### 1. 华为昇腾 (Ascend)

**硬件**：
- Atlas 系列训练卡（如 910B）
- Atlas 系列推理卡（如 310P）

**软件栈**：
```
应用层：PyTorch, TensorFlow
   ↓
框架层：Torch-NPU
   ↓
编译层：ATC (Ascend Tensor Compiler)
   ↓
运行时：CANN (Compute Architecture for Neural Networks)
   ↓
驱动层：Ascend Driver
   ↓
硬件层：Ascend NPU
```

**编程模型**：
- **AscendC**：类似 CUDA C++，用于算子开发
- **AscendCL**：类似 CUDA Runtime，提供运行时 API

**通信库**：
- **HCCL** (Huawei Collective Communication Library)：类似 NCCL

#### 2. 寒武纪思元 (MLU)

**硬件**：
- MLU370 系列（推理+训练）
- MLU590 系列（训练优化）

**软件栈**：
```
应用层：PyTorch, TensorFlow
   ↓
框架层：Catch (PyTorch 适配)
   ↓
编译层：CNCC
   ↓
运行时：CNRT (Cambricon Neuware Runtime)
   ↓
驱动层：CNDRV
   ↓
硬件层：MLU
```

**编程模型**：
- **Bang C**：算子编程语言
- **CNRT API**：运行时 API

**通信库**：
- **CNCL** (Cambricon Collective Communication Library)

#### 3. 海光 DCU

**说明**：基于 AMD 架构，兼容 HIP/ROCm，适配相对简单（类似 LMCache 已有的 ROCm 支持）

---

## Kernel 层适配

### 华为昇腾 (Ascend) Kernel 开发

#### 步骤 1：环境准备

```bash
# 安装 CANN 开发套件
# 假设已安装 Ascend 驱动和 CANN

# 安装 PyTorch + Torch-NPU
pip install torch==2.1.0
pip install torch-npu

# 验证环境
python -c "import torch; import torch_npu; print(torch_npu.npu.is_available())"
```

#### 步骤 2：创建 AscendC Kernel

**目录结构**：
```
LMCache/
├── csrc_ascend/              # 新建目录
│   ├── mem_kernels_ascend.cpp
│   ├── ac_enc_ascend.cpp
│   ├── pybind_ascend.cpp
│   └── CMakeLists.txt
└── setup_ascend.py           # NPU 构建脚本
```

**示例：内存拷贝 Kernel（AscendC）**

```cpp
// csrc_ascend/mem_kernels_ascend.cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_copy.h"
#include <torch/extension.h>

// 使用 AscendCL API 进行内存操作
void multi_layer_kv_transfer_ascend(
    torch::Tensor& key_value,           // [2, num_layer, num_tokens, ...]
    const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping,
    const int page_buffer_size,
    const bool direction) {
    
    // 获取 NPU Stream
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    
    int num_layers = key_value.size(1);
    int num_tokens = slot_mapping.size(0);
    int num_elements = key_value.size(3);
    
    // 获取数据指针
    void* kv_ptr = key_value.data_ptr();
    const int64_t* slot_ptr = slot_mapping.data_ptr<int64_t>();
    
    // 调用自定义算子或使用 AscendCL API
    // 方法 1：使用 AscendCL 提供的基础操作
    for (int layer = 0; layer < num_layers; layer++) {
        for (int k_or_v = 0; k_or_v < 2; k_or_v++) {
            // 计算源地址和目标地址
            void* src = /* ... */;
            void* dst = /* ... */;
            size_t size = num_tokens * num_elements * key_value.element_size();
            
            // 异步拷贝
            aclError ret = aclrtMemcpyAsync(
                dst, size, src, size,
                direction ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_DEVICE,
                stream
            );
            
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMemcpyAsync failed");
            }
        }
    }
    
    // 方法 2：调用自定义 AscendC 算子（推荐）
    // 见下文 "编写 AscendC 算子"
}
```

**编写 AscendC 算子**（高性能版本）

```cpp
// csrc_ascend/kv_transfer_kernel.cpp
#include "kernel_operator.h"

using namespace AscendC;

// 全局内存（类似 CUDA Global Memory）
__global__ __aicore__ void kv_transfer_kernel(
    GM_ADDR src,           // Global Memory 地址
    GM_ADDR dst,
    GM_ADDR slot_mapping,
    int num_tokens,
    int num_heads,
    int head_size) {
    
    // 获取核心 ID（类似 CUDA blockIdx/threadIdx）
    int block_idx = GetBlockIdx();
    
    // 本地内存（类似 CUDA Shared Memory）
    __ubuf__ float local_buf[256];  // UB (Unified Buffer)
    
    // 数据搬运：Global Memory → Local Memory
    DataCopy(local_buf, src + block_idx * 256, 256);
    
    // 计算和处理
    for (int i = 0; i < 256; i++) {
        // 你的处理逻辑
    }
    
    // 写回：Local Memory → Global Memory
    DataCopy(dst + block_idx * 256, local_buf, 256);
}

// Host 端调用
extern "C" void launch_kv_transfer(
    void* src, void* dst, void* slot_mapping,
    int num_tokens, int num_heads, int head_size,
    aclrtStream stream) {
    
    // 配置 Kernel 参数
    uint32_t block_dim = 32;  // 核数
    
    // 调用 Kernel
    aclrtLaunchKernel(
        (void*)kv_transfer_kernel,
        block_dim,
        nullptr,  // 参数结构体
        0,
        stream
    );
}
```

**PyTorch C++ Extension 绑定**

```cpp
// csrc_ascend/pybind_ascend.cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

void multi_layer_kv_transfer_ascend(
    torch::Tensor& key_value,
    const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping,
    const int page_buffer_size,
    const bool direction);

PYBIND11_MODULE(c_ops_npu, m) {
    m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer_ascend,
          "Multi-layer KV transfer for Ascend NPU");
    
    // 其他函数...
}
```

#### 步骤 3：构建系统

```python
# setup_ascend.py
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension
import os

# 检测是否在昇腾环境
ASCEND_HOME = os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest')

def get_ascend_extension():
    """构建昇腾 NPU 扩展"""
    include_dirs = [
        f"{ASCEND_HOME}/include",
        f"{ASCEND_HOME}/include/aclnn",
    ]
    
    library_dirs = [
        f"{ASCEND_HOME}/lib64",
    ]
    
    libraries = [
        "ascendcl",
        "acl_op_compiler",
        "nnopbase",
    ]
    
    sources = [
        "csrc_ascend/pybind_ascend.cpp",
        "csrc_ascend/mem_kernels_ascend.cpp",
        "csrc_ascend/ac_enc_ascend.cpp",
        # 其他源文件...
    ]
    
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
    }
    
    ext = Extension(
        name="lmcache.c_ops_npu",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        language="c++"
    )
    
    return ext

if __name__ == "__main__":
    setup(
        name="lmcache-npu",
        ext_modules=[get_ascend_extension()],
        cmdclass={"build_ext": BuildExtension},
    )
```

**构建命令**：

```bash
# 设置环境变量
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH

# 构建
python setup_ascend.py build_ext --inplace

# 安装
pip install -e .
```

### 寒武纪 MLU Kernel 开发

#### 环境准备

```bash
# 安装寒武纪 CNToolkit
# 假设已安装驱动

# 安装 PyTorch + Catch
pip install torch==1.13.0+cpu  # 或指定版本
pip install torch_mlu

# 验证
python -c "import torch; import torch_mlu; print(torch.mlu.is_available())"
```

#### Bang C Kernel 示例

```cpp
// csrc_mlu/mem_kernels_mlu.mlu
#include "cnrt.h"
#include "bang.h"

// Bang C Kernel（类似 CUDA Kernel）
__mlu_entry__ void kv_transfer_kernel_mlu(
    float* dst,
    const float* src,
    const int64_t* slot_mapping,
    int num_tokens,
    int num_heads,
    int head_size) {
    
    // 获取任务 ID（类似 CUDA blockIdx/threadIdx）
    int task_id = taskId;
    int task_dim = taskDim;
    
    // NRAM：芯片上高速内存（类似 Shared Memory）
    __nram__ float nram_buffer[512];
    
    // 数据加载：GDRAM → NRAM
    int token_idx = task_id;
    if (token_idx < num_tokens) {
        int slot_idx = slot_mapping[token_idx];
        
        // Bang C 的向量加载指令
        __bang_write_value(nram_buffer, 512, 0.0f);
        
        // 从 GDRAM 加载到 NRAM
        __memcpy(nram_buffer, src + slot_idx * num_heads * head_size,
                 num_heads * head_size * sizeof(float),
                 GDRAM2NRAM);
        
        // 处理数据（示例：简单拷贝）
        // 实际可以进行复杂计算
        
        // 写回：NRAM → GDRAM
        __memcpy(dst + token_idx * num_heads * head_size, nram_buffer,
                 num_heads * head_size * sizeof(float),
                 NRAM2GDRAM);
    }
}

// Host 端启动函数
extern "C" void launch_kv_transfer_mlu(
    void* dst, void* src, void* slot_mapping,
    int num_tokens, int num_heads, int head_size,
    cnrtQueue_t queue) {
    
    // 配置 Kernel 参数
    cnrtDim3_t dim = {num_tokens, 1, 1};
    cnrtFunctionType_t func_type = CNRT_FUNC_TYPE_UNION1;  // 计算核心类型
    
    // 调用 Kernel
    kv_transfer_kernel_mlu<<<dim, func_type, queue>>>(
        (float*)dst,
        (const float*)src,
        (const int64_t*)slot_mapping,
        num_tokens,
        num_heads,
        head_size
    );
}
```

**C++ 封装**

```cpp
// csrc_mlu/mem_kernels_mlu_wrapper.cpp
#include <torch/extension.h>
#include "cnrt.h"

extern "C" void launch_kv_transfer_mlu(
    void* dst, void* src, void* slot_mapping,
    int num_tokens, int num_heads, int head_size,
    cnrtQueue_t queue);

void multi_layer_kv_transfer_mlu(
    torch::Tensor& key_value,
    const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping,
    const int page_buffer_size,
    const bool direction) {
    
    // 获取 MLU Queue（类似 CUDA Stream）
    cnrtQueue_t queue = torch_mlu::getCurrentQueue().queue();
    
    void* kv_ptr = key_value.data_ptr();
    const int64_t* slot_ptr = slot_mapping.data_ptr<int64_t>();
    
    int num_tokens = slot_mapping.size(0);
    int num_heads = key_value.size(3) / key_value.size(4);
    int head_size = key_value.size(4);
    
    // 调用 Bang C Kernel
    launch_kv_transfer_mlu(
        kv_ptr, kv_ptr,  // 简化示例，实际需要正确的地址
        (void*)slot_ptr,
        num_tokens, num_heads, head_size,
        queue
    );
    
    // 同步（可选）
    // cnrtQueueSync(queue);
}

PYBIND11_MODULE(c_ops_mlu, m) {
    m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer_mlu);
}
```

### 海光 DCU（类似 AMD ROCm）

**说明**：海光 DCU 基于 AMD 架构，可以直接复用 LMCache 的 HIP 代码！

```bash
# 构建命令（类似 ROCm）
PYTORCH_ROCM_ARCH="gfx906" \
CXX=hipcc \
BUILD_WITH_HIP=1 \
python setup.py install
```

只需确保：
1. 安装 DCU 驱动和 DTK（DCU Toolkit）
2. 设置正确的 `ROCM_PATH`
3. 使用 `hipcc` 编译

---

## 通信层适配

### NCCL → HCCL/CNCL 替换

LMCache 的 P2P 通信主要通过 **NIXL**（基于 NCCL）实现。需要替换为国产通信库。

#### 1. 华为 HCCL 适配

**原始通信代码**（基于 NIXL）：
```python
# lmcache/v1/transfer_channel/nixl_channel.py
from nixl import NixlChannel

class LMCacheNixlChannel:
    def __init__(self, config):
        self.channel = NixlChannel(
            rank=config.rank,
            world_size=config.world_size
        )
    
    def send(self, tensor, dst_rank):
        self.channel.send(tensor, dst_rank)
    
    def recv(self, tensor, src_rank):
        self.channel.recv(tensor, src_rank)
```

**适配 HCCL**：

```python
# lmcache/v1/transfer_channel/hccl_channel.py
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

class LMCacheHCCLChannel:
    """华为昇腾 HCCL 通信通道"""
    
    def __init__(self, config):
        self.rank = config.rank
        self.world_size = config.world_size
        
        # 初始化 HCCL
        torch.distributed.init_process_group(
            backend='hccl',  # 使用 HCCL 后端
            init_method=config.init_method,
            world_size=self.world_size,
            rank=self.rank
        )
        
        # 创建通信组
        self.process_group = torch.distributed.new_group(
            ranks=list(range(self.world_size))
        )
    
    def send(self, tensor, dst_rank):
        """发送 tensor 到目标 rank"""
        # 确保 tensor 在 NPU 上
        if not tensor.is_npu:
            tensor = tensor.to(f'npu:{self.rank}')
        
        # 使用 HCCL P2P Send
        torch.distributed.send(
            tensor=tensor,
            dst=dst_rank,
            group=self.process_group
        )
    
    def recv(self, tensor, src_rank):
        """从源 rank 接收 tensor"""
        if not tensor.is_npu:
            tensor = tensor.to(f'npu:{self.rank}')
        
        # 使用 HCCL P2P Recv
        torch.distributed.recv(
            tensor=tensor,
            src=src_rank,
            group=self.process_group
        )
        
        return tensor
    
    def broadcast(self, tensor, src_rank):
        """广播"""
        if not tensor.is_npu:
            tensor = tensor.to(f'npu:{self.rank}')
        
        torch.distributed.broadcast(
            tensor=tensor,
            src=src_rank,
            group=self.process_group
        )
        
        return tensor
    
    def all_reduce(self, tensor, op='sum'):
        """AllReduce 操作"""
        if not tensor.is_npu:
            tensor = tensor.to(f'npu:{self.rank}')
        
        op_map = {
            'sum': torch.distributed.ReduceOp.SUM,
            'max': torch.distributed.ReduceOp.MAX,
            'min': torch.distributed.ReduceOp.MIN,
        }
        
        torch.distributed.all_reduce(
            tensor=tensor,
            op=op_map[op],
            group=self.process_group
        )
        
        return tensor
```

**注册通信后端**：

```python
# lmcache/v1/transfer_channel/__init__.py
from .abstract import TransferChannel
from .nixl_channel import LMCacheNixlChannel

# 新增
from .hccl_channel import LMCacheHCCLChannel
from .cncl_channel import LMCacheCNCLChannel

CHANNEL_REGISTRY = {
    "nixl": LMCacheNixlChannel,
    "hccl": LMCacheHCCLChannel,  # 华为昇腾
    "cncl": LMCacheCNCLChannel,  # 寒武纪
    "tcp": LMCacheTCPChannel,    # 通用 TCP（低性能）
}

def get_transfer_channel(backend: str, config):
    channel_cls = CHANNEL_REGISTRY.get(backend)
    if not channel_cls:
        raise ValueError(f"Unknown transfer channel: {backend}")
    return channel_cls(config)
```

#### 2. 寒武纪 CNCL 适配

```python
# lmcache/v1/transfer_channel/cncl_channel.py
import torch
import torch_mlu

class LMCacheCNCLChannel:
    """寒武纪 MLU CNCL 通信通道"""
    
    def __init__(self, config):
        self.rank = config.rank
        self.world_size = config.world_size
        
        # 初始化 CNCL
        torch.distributed.init_process_group(
            backend='cncl',  # 使用 CNCL 后端
            init_method=config.init_method,
            world_size=self.world_size,
            rank=self.rank
        )
        
        self.process_group = torch.distributed.new_group(
            ranks=list(range(self.world_size))
        )
    
    def send(self, tensor, dst_rank):
        """发送 tensor"""
        # 确保在 MLU 上
        if not tensor.is_mlu:
            tensor = tensor.to(f'mlu:{self.rank}')
        
        torch.distributed.send(
            tensor=tensor,
            dst=dst_rank,
            group=self.process_group
        )
    
    def recv(self, tensor, src_rank):
        """接收 tensor"""
        if not tensor.is_mlu:
            tensor = tensor.to(f'mlu:{self.rank}')
        
        torch.distributed.recv(
            tensor=tensor,
            src=src_rank,
            group=self.process_group
        )
        
        return tensor
    
    # broadcast, all_reduce 等类似实现...
```

#### 3. 配置文件适配

```yaml
# config_ascend.yaml
chunk_size: 256
local_device: "npu"  # 使用 NPU
max_local_cache_size: 10

# P2P 通信配置
enable_p2p: True
transfer_channel: "hccl"  # 使用 HCCL

# HCCL 特定配置
hccl_config:
  init_method: "env://"
  rank: 0
  world_size: 2
```

### 低级别通信优化

#### 使用 RDMA（如果硬件支持）

```python
# lmcache/v1/transfer_channel/rdma_channel.py
import socket
import struct

class LMCacheRDMAChannel:
    """基于 RDMA 的高性能通信（需要硬件支持）"""
    
    def __init__(self, config):
        self.rank = config.rank
        self.world_size = config.world_size
        
        # 初始化 RDMA 设备
        # 这里需要根据具体硬件使用相应的 RDMA 库
        # 例如：libibverbs (Infiniband), RoCE 等
        
        self.device = self._init_rdma_device()
        self.qp = self._create_queue_pair()
    
    def _init_rdma_device(self):
        """初始化 RDMA 设备"""
        # 伪代码，实际需要根据硬件 API
        pass
    
    def send(self, tensor, dst_rank):
        """RDMA Send"""
        # 注册内存区域
        mr = self._register_memory(tensor)
        
        # RDMA Write
        self._rdma_write(mr, dst_rank)
    
    def recv(self, tensor, src_rank):
        """RDMA Recv"""
        mr = self._register_memory(tensor)
        
        # RDMA Read
        self._rdma_read(mr, src_rank)
```

---

## 完整适配案例

### 案例：适配华为昇腾的完整流程

#### 1. 项目结构

```
LMCache/
├── lmcache/
│   ├── __init__.py
│   ├── config.py
│   ├── cache_engine.py
│   └── v1/
│       └── transfer_channel/
│           ├── __init__.py
│           ├── abstract.py
│           ├── nixl_channel.py      # 原有 NVIDIA
│           └── hccl_channel.py      # 新增昇腾
│
├── csrc/                            # 原有 CUDA
│   ├── mem_kernels.cu
│   └── ...
│
├── csrc_ascend/                     # 新增昇腾
│   ├── mem_kernels_ascend.cpp
│   ├── ac_enc_ascend.cpp
│   ├── pybind_ascend.cpp
│   └── CMakeLists.txt
│
├── setup.py                         # 原有构建
├── setup_ascend.py                  # 昇腾构建
└── README_ASCEND.md                 # 昇腾文档
```

#### 2. 修改 Python 层（设备抽象）

```python
# lmcache/device.py
import torch

class DeviceManager:
    """设备管理器，自动检测和切换设备"""
    
    @staticmethod
    def get_device_type():
        """检测当前设备类型"""
        if torch.cuda.is_available():
            return "cuda"
        
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                return "npu"
        except ImportError:
            pass
        
        try:
            import torch_mlu
            if torch_mlu.is_available():
                return "mlu"
        except ImportError:
            pass
        
        return "cpu"
    
    @staticmethod
    def get_device(device_id=0):
        """获取设备对象"""
        device_type = DeviceManager.get_device_type()
        if device_type == "cuda":
            return torch.device(f"cuda:{device_id}")
        elif device_type == "npu":
            return torch.device(f"npu:{device_id}")
        elif device_type == "mlu":
            return torch.device(f"mlu:{device_id}")
        else:
            return torch.device("cpu")
    
    @staticmethod
    def synchronize():
        """同步设备"""
        device_type = DeviceManager.get_device_type()
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "npu":
            import torch_npu
            torch_npu.npu.synchronize()
        elif device_type == "mlu":
            import torch_mlu
            torch_mlu.synchronize()
```

#### 3. 修改 Cache Engine

```python
# lmcache/v1/cache_engine.py
from lmcache.device import DeviceManager

class LMCacheEngine:
    def __init__(self, config, metadata):
        self.config = config
        self.metadata = metadata
        
        # 自动检测设备
        self.device_type = DeviceManager.get_device_type()
        self.device = DeviceManager.get_device()
        
        # 加载对应的 C++ 扩展
        if self.device_type == "cuda":
            from lmcache import c_ops  # CUDA 版本
            self.ops = c_ops
        elif self.device_type == "npu":
            from lmcache import c_ops_npu  # NPU 版本
            self.ops = c_ops_npu
        elif self.device_type == "mlu":
            from lmcache import c_ops_mlu  # MLU 版本
            self.ops = c_ops_mlu
        else:
            raise RuntimeError(f"Unsupported device: {self.device_type}")
        
        # 初始化通信通道
        if config.enable_p2p:
            from lmcache.v1.transfer_channel import get_transfer_channel
            self.transfer_channel = get_transfer_channel(
                config.transfer_channel_backend,
                config
            )
    
    def store(self, tokens, kv_cache):
        """存储 KV Cache"""
        # 确保 tensor 在正确的设备上
        if not kv_cache.device.type == self.device_type:
            kv_cache = kv_cache.to(self.device)
        
        # 调用设备特定的 Kernel
        self.ops.multi_layer_kv_transfer(
            kv_cache,
            # ... 其他参数
        )
    
    def retrieve(self, tokens):
        """检索 KV Cache"""
        # ... 类似实现
        pass
```

#### 4. 构建和测试

```bash
# 1. 设置环境变量
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH
export PATH=$ASCEND_HOME_PATH/bin:$PATH

# 2. 构建 NPU 扩展
python setup_ascend.py build_ext --inplace

# 3. 安装
pip install -e .

# 4. 验证
python -c "from lmcache import c_ops_npu; print('NPU ops loaded successfully!')"

# 5. 运行测试
pytest tests/test_npu_ops.py
```

#### 5. 完整测试脚本

```python
# tests/test_npu_ops.py
import torch
import torch_npu
from lmcache import c_ops_npu

def test_multi_layer_kv_transfer():
    """测试多层 KV 传输"""
    num_layers = 32
    num_tokens = 128
    num_heads = 32
    head_size = 128
    
    # 创建测试数据（在 NPU 上）
    key_value = torch.randn(
        2, num_layers, num_tokens, num_heads * head_size,
        device='npu:0',
        dtype=torch.float16
    )
    
    slot_mapping = torch.arange(num_tokens, device='npu:0', dtype=torch.int64)
    
    # 调用 NPU Kernel
    c_ops_npu.multi_layer_kv_transfer(
        key_value,
        None,  # key_value_ptrs
        slot_mapping,
        1024,  # page_buffer_size
        True   # direction
    )
    
    # 同步并验证
    torch_npu.npu.synchronize()
    
    print("✅ NPU KV transfer test passed!")

if __name__ == "__main__":
    test_multi_layer_kv_transfer()
```

---

## 性能优化建议

### 1. Kernel 优化

#### 华为昇腾优化技巧

```cpp
// 优化 1：使用向量化指令
__global__ __aicore__ void optimized_kernel(...) {
    // 使用 Vector 计算单元
    __ubuf__ half16 vec_buf[64];  // 使用 FP16 向量
    
    // 向量化加载
    VecLoad(vec_buf, src, 64);
    
    // 向量化计算
    VecMul(vec_buf, vec_buf, scale, 64);
    
    // 向量化存储
    VecStore(dst, vec_buf, 64);
}

// 优化 2：流水线并行
__global__ __aicore__ void pipelined_kernel(...) {
    // Stage 1: Load
    DataCopyEnque(local_buf_1, global_src_1, size);
    
    // Stage 2: Compute (与 Load 并行)
    ProcessData(local_buf_2);
    
    // Stage 3: Store (与 Compute 并行)
    DataCopyEnque(global_dst_3, local_buf_3, size);
    
    // 流水线同步
    DataCopyWait();
}
```

#### 寒武纪 MLU 优化

```cpp
// 优化：使用 NRAM + WRAM 双缓冲
__mlu_entry__ void double_buffer_kernel(...) {
    __nram__ float nram_buf[2][512];  // 双缓冲
    __wram__ float wram_buf[512];     // Weight RAM
    
    int ping = 0;
    
    for (int i = 0; i < num_iters; i++) {
        // 当前缓冲处理
        int pong = 1 - ping;
        
        // 异步加载下一批数据到 pong
        __memcpy_async(nram_buf[pong], src + (i+1) * 512, 
                       512 * sizeof(float), GDRAM2NRAM);
        
        // 处理当前 ping 缓冲
        __bang_mul(nram_buf[ping], nram_buf[ping], wram_buf, 512);
        
        // 等待加载完成
        __sync();
        
        // 交换缓冲
        ping = pong;
    }
}
```

### 2. 内存优化

```python
# 使用 Pinned Memory（如果 NPU 支持）
class OptimizedCacheEngine:
    def __init__(self, config):
        # 分配锁页内存
        if config.device_type == "npu":
            # 昇腾：使用 Host Pinned Memory
            self.cpu_buffer = torch.empty(
                buffer_size,
                dtype=torch.float16,
                pin_memory=True  # NPU 也支持 pin_memory
            )
        elif config.device_type == "mlu":
            # 寒武纪：类似实现
            self.cpu_buffer = torch.empty(
                buffer_size,
                dtype=torch.float16,
                pin_memory=True
            )
```

### 3. 通信优化

```python
# 使用通信-计算重叠
class OverlappedTransfer:
    def transfer_with_compute(self, kv_cache, next_layer_compute):
        # 启动异步传输
        send_handle = self.transfer_channel.isend(kv_cache, dst_rank=1)
        
        # 同时进行下一层计算
        next_layer_compute()
        
        # 等待传输完成
        send_handle.wait()
```

---

## 常见问题

### Q1: Kernel 移植后性能下降怎么办？

**A**: 可能原因和解决方案：

1. **未使用硬件特定指令**
   - 华为昇腾：使用 Vector/Cube 计算单元
   - 寒武纪：使用 NRAM/WRAM 高速内存

2. **内存访问模式不优化**
   - 检查数据对齐（128 字节对齐）
   - 使用合并访问（Coalesced Access）

3. **未充分流水线化**
   - 使用双缓冲/多缓冲
   - 计算-访存重叠

### Q2: 如何调试 NPU Kernel？

**华为昇腾**：
```bash
# 1. 设置日志级别
export ASCEND_GLOBAL_LOG_LEVEL=1  # DEBUG
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 2. 使用 Profiler
import torch_npu
from torch_npu.profiler import profile

with profile(activities=[torch_npu.ProfilerActivity.NPU]) as prof:
    # 你的代码
    pass

print(prof.key_averages().table(sort_by="npu_time_total"))
```

**寒武纪 MLU**：
```bash
# 使用 CNProf 工具
cnprof --mode trace python your_script.py

# 查看性能报告
cnprof_parser trace_output.txt
```

### Q3: 通信库不支持某些操作怎么办？

**A**: 使用 TCP 作为 Fallback：

```python
class FallbackChannel:
    def __init__(self, config):
        # 尝试使用硬件通信库
        try:
            if config.device_type == "npu":
                from .hccl_channel import LMCacheHCCLChannel
                self.backend = LMCacheHCCLChannel(config)
            else:
                raise ImportError
        except (ImportError, RuntimeError):
            # Fallback 到 TCP
            from .tcp_channel import LMCacheTCPChannel
            self.backend = LMCacheTCPChannel(config)
            logger.warning("Using TCP fallback for communication")
```

### Q4: 如何验证 Kernel 正确性？

```python
# tests/test_kernel_correctness.py
def test_kernel_correctness():
    """对比 CPU 参考实现和 NPU Kernel"""
    
    # 1. CPU 参考实现
    def cpu_reference(src, slot_mapping):
        result = torch.zeros_like(src)
        for i, slot in enumerate(slot_mapping):
            result[i] = src[slot]
        return result
    
    # 2. NPU Kernel
    def npu_kernel(src, slot_mapping):
        src_npu = src.to('npu:0')
        slot_npu = slot_mapping.to('npu:0')
        result = c_ops_npu.kv_transfer(src_npu, slot_npu)
        return result.cpu()
    
    # 3. 比较结果
    src = torch.randn(100, 256)
    slot_mapping = torch.randperm(100)
    
    cpu_result = cpu_reference(src, slot_mapping)
    npu_result = npu_kernel(src, slot_mapping)
    
    # 允许微小误差（FP16）
    assert torch.allclose(cpu_result, npu_result, atol=1e-3)
    print("✅ Kernel correctness verified!")
```

### Q5: 不同 NPU 之间如何做性能对比？

**性能测试脚本**：

```python
# benchmark_npu.py
import time
import torch

def benchmark_kernel(device, num_runs=100):
    # 预热
    for _ in range(10):
        run_kernel(device)
    
    # 同步
    if device.type == 'npu':
        torch_npu.npu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测试
    start = time.time()
    for _ in range(num_runs):
        run_kernel(device)
    
    # 同步
    if device.type == 'npu':
        torch_npu.npu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    
    end = time.time()
    
    avg_time = (end - start) / num_runs * 1000  # ms
    return avg_time

# 运行对比
devices = [
    torch.device('cuda:0'),  # NVIDIA A100
    torch.device('npu:0'),   # 华为 910B
    torch.device('mlu:0'),   # 寒武纪 MLU370
]

for device in devices:
    try:
        avg_time = benchmark_kernel(device)
        print(f"{device.type.upper()}: {avg_time:.3f} ms")
    except Exception as e:
        print(f"{device.type.upper()}: Not available ({e})")
```

---

## 总结

### 适配检查清单

- [ ] **Kernel 层**
  - [ ] 内存操作 Kernel（`mem_kernels`）
  - [ ] 算术编码 Kernel（`ac_enc/ac_dec`）
  - [ ] 位置编码 Kernel（`pos_kernels`）
  - [ ] Python 绑定（`pybind`）

- [ ] **通信层**
  - [ ] P2P 通信（HCCL/CNCL）
  - [ ] 集群通信（AllReduce 等）
  - [ ] Fallback 到 TCP

- [ ] **构建系统**
  - [ ] 设备检测
  - [ ] 条件编译
  - [ ] 依赖管理

- [ ] **测试**
  - [ ] 单元测试
  - [ ] 正确性验证
  - [ ] 性能基准测试

### 参考资源

#### 华为昇腾
- [CANN 开发文档](https://www.hiascend.com/document)
- [AscendC 编程指南](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/apiref/aclcppdevg/aclcppdevg_000001.html)
- [Torch-NPU GitHub](https://github.com/Ascend/pytorch)

#### 寒武纪
- [CNToolkit 文档](https://www.cambricon.com/docs/)
- [Bang C 编程指南](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/programming_guide_1.7.0/index.html)
- [Catch PyTorch 适配](https://github.com/Cambricon/catch)

#### 海光 DCU
- [DTK 文档](https://www.hygon.cn/)
- [HIP 编程指南](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)

---

**祝你适配顺利！如有问题，欢迎参与社区讨论或提 Issue。**

