# LMCache 框架深度解析：架构与二次开发指南

## 目录
1. [LMCache 是什么](#lmcache-是什么)
2. [工程还是算法？与 GPU 的关系](#工程还是算法与-gpu-的关系)
3. [框架适配情况](#框架适配情况)
4. [核心架构解析](#核心架构解析)
5. [二次开发指南](#二次开发指南)
6. [总结](#总结)

---

## LMCache 是什么

### 1.1 核心定位

**LMCache 是一个 LLM（大语言模型）推理加速的引擎扩展**，它的核心目标是：

- **减少 TTFT（Time To First Token）**：首个 token 的生成时间
- **提升吞吐量**：整体推理性能
- **优化长上下文场景**：特别适合多轮对话、RAG（检索增强生成）等场景

### 1.2 工作原理

LMCache 的核心思想是 **KV Cache 的复用**：

在大语言模型推理过程中，每个 token 都会产生 Key 和 Value 张量（KV Cache）。传统方法中，每次推理都需要重新计算这些 KV Cache，即使输入的文本部分重复。

**LMCache 的创新**：
- 将任何可重用的文本片段的 KV Cache 存储起来
- 不仅限于前缀（prefix）匹配，任何重复出现的文本段都可以复用
- 跨请求、跨实例共享 KV Cache

**性能提升**：
根据官方数据，结合 vLLM 使用时，可以在多轮 QA 和 RAG 场景下实现：
- **3-10x 延迟降低**
- **3-10x GPU 计算周期节省**

### 1.3 典型应用场景

1. **多轮对话（Multi-round QA）**
   - 系统提示词（System Prompt）每次都一样
   - 历史对话内容可以复用
   - 大幅减少重复计算

2. **RAG（检索增强生成）**
   - 检索到的文档可以缓存
   - 相同查询可以直接复用 KV Cache
   - 加速文档理解和问答

3. **长文档处理**
   - 长文档只需 prefill 一次
   - 后续查询直接复用
   - 节省大量 GPU 显存和计算

4. **Agent 系统**
   - 工具描述、函数签名等固定内容
   - 复用公共知识库内容

---

## 工程，算法，与 GPU 的关系

### 2.1 项目定位：工程为主，算法为辅

**LMCache 是一个以工程实现为核心的项目**，但也融合了算法创新：

#### 工程方面（主体）

1. **多层级存储架构**
   - GPU Memory（活跃 KV Cache）
   - CPU DRAM（热缓存，使用 pinned memory 加速传输）
   - Local Storage（本地磁盘、NVMe GDS）
   - Remote Storage（Redis、Mooncake、InfiniStore 等）

2. **高性能 C++/CUDA 实现**
   ```
   csrc/
   ├── ac_dec.cu          # 算术编码解码
   ├── ac_enc.cu          # 算术编码
   ├── cal_cdf.cu         # CDF 计算
   ├── mem_kernels.cu     # 内存操作核函数
   ├── pos_kernels.cu     # 位置编码核函数
   └── ...
   ```

3. **异步化设计**
   - 异步 Offload：GPU → CPU → Disk/Remote
   - 异步 Prefetch：Disk/Remote → CPU → GPU
   - 不阻塞推理主线程

4. **分布式协调**
   - P2P KV Cache 共享
   - Disaggregated Prefill（Prefill-Decode 分离）
   - 跨实例缓存同步

#### 算法方面（增强）

1. **CacheGen 压缩算法**
   - KV Cache 压缩和流式传输
   - 基于算术编码的无损/有损压缩
   - 论文：*CacheGen: KV Cache Compression and Streaming for Fast LLM Serving* (SIGCOMM 2024)

2. **CacheBlend 融合技术**
   - 缓存知识融合
   - 智能 KV Cache 混合
   - 论文：*CacheBlend: Fast LLM Serving with Cached Knowledge Fusion* (EuroSys 2025)

3. **缓存策略**
   - LRU（Least Recently Used）驱逐策略
   - Layerwise 缓存优化
   - Token 序列哈希和索引

### 2.2 与 GPU 的深度关系

**LMCache 与 GPU 高度相关，但支持多种硬件**：

#### NVIDIA GPU 支持（主流）

- **要求**：Compute Capability 7.0+
- **支持卡型**：V100, T4, RTX 20xx, A100, L4, H100, B200 等
- **CUDA 版本**：12.1+
- **构建方式**：
  ```bash
  # 使用 CUDA 构建
  uv pip install lmcache
  ```

#### AMD GPU 支持（ROCm）

- **支持卡型**：MI300X 等 AMD Instinct 系列
- **ROCm 支持**：通过 HIP 编译
- **构建方式**：
  ```bash
  PYTORCH_ROCM_ARCH="gfx942" \
  TORCH_DONT_CHECK_COMPILER_ABI=1 \
  CXX=hipcc \
  BUILD_WITH_HIP=1 \
  python3 -m pip install --no-build-isolation -e .
  ```

#### 设备层级划分

LMCache 的设计天然需要**区分不同设备**：

| 存储层级 | 容量 | 速度 | 用途 |
|---------|------|------|------|
| GPU Memory | 小（40GB-80GB） | 极快 | 活跃 KV Cache |
| CPU DRAM | 中（数百 GB） | 快 | 热缓存，pinned memory |
| Local Disk | 大（数 TB） | 中 | 长文档、历史缓存 |
| Remote Storage | 极大 | 慢 | 持久化、跨实例共享 |

**关键技术**：
- **Pinned Memory**：CPU 侧使用锁页内存，加速 GPU-CPU 传输
- **NUMA-aware Allocation**：NUMA 感知的内存分配
- **Async Copy**：异步内存拷贝，不阻塞 CUDA Stream

### 2.3 底层实现语言

```python
# Python 层（lmcache/）
- 配置管理、存储后端抽象
- 与推理框架集成（vLLM, SGLang）
- 缓存策略、控制逻辑

# C++/CUDA 层（csrc/）
- 高性能内存操作
- CUDA 核函数（压缩、编码、内存拷贝）
- 位置编码计算

# HIP 层（ROCm 支持）
- AMD GPU 适配
- 通过 hipify 自动转换
```

---

## 框架适配情况

### 3.1 并非开箱即用所有框架

**LMCache 不是任意推理框架都能直接使用的**，它需要与特定框架深度集成。

### 3.2 官方支持的框架

#### 1. vLLM（主要集成）

**vLLM v1 集成（推荐）**

- **集成方式**：通过 KV Connector
- **支持特性**：
  - ✅ 高性能 CPU KV Cache offloading
  - ✅ Disaggregated prefill（Prefill-Decode 分离）
  - ✅ P2P KV Cache 共享
  - ✅ 多种存储后端

**使用示例**：
```python
from vllm import LLM, SamplingParams

# 创建 vLLM 实例，通过配置启用 LMCache
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_transfer_config={
        "kv_connector": "LMCacheConnector",
        "kv_buffer_size": 1e9,  # 1GB
    }
)

# 正常推理即可，LMCache 自动管理 KV Cache
outputs = llm.generate(prompts, sampling_params)
```

**vLLM v0 支持（旧版）**

- 通过 `examples/others/lmcache/cpu_offload_lmcache.py`
- 功能相对受限

**兼容性矩阵**（核心版本）：

| vLLM 版本 | LMCache 0.3.7 | LMCache 0.3.6 | LMCache 0.3.5 |
|-----------|---------------|---------------|---------------|
| 0.10.2.x  | ✅ | ✅ | 🕯️ (torch 不兼容) |
| 0.10.1.x  | 🕯️ | ❌ (API 不兼容) | ✅ |
| 0.10.0.x  | 🕯️ | ❌ | ✅ |

🕯️ 表示需要 `--no-build-isolation` 解决 torch 版本冲突

#### 2. SGLang 集成

- **支持程度**：KV cache offloading
- **集成路径**：`lmcache/integration/sglang/`
- **功能**：CPU offload 为主

**使用示例**：
```python
# 通过配置文件启用 LMCache
import sglang as sgl

# 配置 LMCache
lmcache_config = {
    "local_device": "cpu",
    "remote_url": None,
}

# SGLang 引擎会自动使用 LMCache
```

#### 3. Production Stack 支持

LMCache 已被官方集成到：
- **vLLM Production Stack**：企业级部署方案
- **llm-d**：部署工具链
- **KServe**：Kubernetes 原生模型服务

### 3.3 集成架构（Connector 模式）

LMCache 使用 **Connector 模式** 与推理框架集成：

```
┌─────────────────────────────────────────┐
│   Inference Engine (vLLM/SGLang)       │
│                                         │
│   ┌─────────────────────────────────┐  │
│   │   KV Memory Manager             │  │
│   └──────────────┬──────────────────┘  │
│                  │                      │
│   ┌──────────────▼──────────────────┐  │
│   │   LMCache Connector             │◄─┼── Integration Layer
│   │   - Cache Lookup                │  │
│   │   - Cache Store                 │  │
│   │   - Async Operations            │  │
│   └──────────────┬──────────────────┘  │
└──────────────────┼──────────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │   LMCache Storage Backend       │
    │   - GPU → CPU → Disk → Remote   │
    │   - Multi-tier Cache Management │
    └─────────────────────────────────┘
```

**关键 Connector 实现**：

```python
# lmcache/integration/vllm/lmcache_connector_v1.py
class LMCacheConnector:
    """vLLM v1 的 KV Connector"""
    
    def __init__(self, config):
        self.cache_engine = LMCacheEngine(config)
    
    def lookup(self, token_ids):
        """查询 KV Cache"""
        return self.cache_engine.retrieve(token_ids)
    
    def store(self, token_ids, kv_cache):
        """存储 KV Cache"""
        self.cache_engine.store(token_ids, kv_cache)
```

### 3.4 如何支持新框架？

要将 LMCache 集成到新的推理框架，需要：

1. **实现 Connector 接口**
   - `lookup(token_ids)`: 查询缓存
   - `store(token_ids, kv_cache)`: 存储缓存
   - `exists(token_ids)`: 检查是否存在

2. **Hook 到框架的 KV Cache 管理**
   - Prefill 阶段：检查缓存命中
   - Decode 阶段：存储新生成的 KV

3. **处理分布式场景**
   - Tensor Parallel：跨 GPU 协调
   - Pipeline Parallel：跨 stage 传输

参考实现：`lmcache/integration/vllm/` 和 `lmcache/integration/sglang/`

---

## 核心架构解析

### 4.1 系统架构图

```
                   ┌────────────────────────────────────────┐
                   │  LMCache Controller (管理 API)         │
                   │  - Lookup / Clear / Compress / Move   │
                   └──────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────▼───────┐        ┌────────▼────────┐      ┌────────▼────────┐
│  Worker 1     │        │  Worker 2       │      │  Worker N       │
│               │        │                 │      │                 │
│  ┌─────────┐  │        │  ┌─────────┐    │      │  ┌─────────┐    │
│  │ GPU Mem │  │        │  │ GPU Mem │    │      │  │ GPU Mem │    │
│  └────┬────┘  │        │  └────┬────┘    │      │  └────┬────┘    │
│       │       │        │       │         │      │       │         │
│  ┌────▼────┐  │        │  ┌────▼────┐    │      │  ┌────▼────┐    │
│  │ CPU Mem │  │        │  │ CPU Mem │    │      │  │ CPU Mem │    │
│  │ (pinned)│  │        │  │ (pinned)│    │      │  │ (pinned)│    │
│  └────┬────┘  │        │  └────┬────┘    │      │  └────┬────┘    │
└───────┼───────┘        └────────┼─────────┘      └────────┼────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
     ┌────────▼────────┐  ┌───────▼───────┐  ┌───────▼────────┐
     │  Local Disk     │  │ Redis/KV Store│  │ Remote Storage │
     │  (NVMe, GDS)    │  │               │  │ (Mooncake, etc)│
     └─────────────────┘  └───────────────┘  └────────────────┘
```

### 4.2 核心组件详解

#### 1. Cache Index（Token Database）

**功能**：维护 Token 序列到 KV Cache 的映射

```python
# 核心数据结构
{
    "token_hash_123": {
        "tokens": [1, 2, 3, ...],
        "location": "cpu",  # or "disk", "remote"
        "chunk_id": "chunk_abc",
        "metadata": {...}
    }
}
```

**哈希策略**：
- 默认按 256 tokens 分块
- 使用快速哈希算法（非加密哈希）
- 支持前缀树（Trie）优化查找

#### 2. Multi-tier Storage Backend

**GPU Memory**
- 存储：活跃的 KV Cache
- 容量：受限于 GPU 显存
- 访问：最快，直接参与计算

**CPU DRAM**
- 存储：热缓存，最近使用的 KV
- 容量：数百 GB
- 技术：
  - Pinned Memory（锁页内存）
  - NUMA-aware Allocation
  - LRU 驱逐策略

**Local Storage**
- 存储：长期缓存、大文档
- 容量：数 TB
- 支持：
  - 普通文件系统（file://）
  - NVMe + GDS（GPU Direct Storage）
  - 压缩存储

**Remote Storage**
- 存储：持久化、跨实例共享
- 支持后端：
  - Redis（redis://）
  - Mooncake
  - InfiniStore
  - S3-compatible
  - 自定义 Connector

#### 3. Async Pipeline（异步流水线）

```python
# 简化的异步流程
async def offload_pipeline():
    # 1. GPU → CPU (sync, fast)
    cpu_buffer = await gpu_to_cpu_async(kv_cache)
    
    # 2. CPU → Disk (async, non-blocking)
    asyncio.create_task(cpu_to_disk_async(cpu_buffer))
    
    # 3. Disk → Remote (async, optional)
    asyncio.create_task(disk_to_remote_async(key))

async def prefetch_pipeline():
    # 1. Remote → Disk (prefetch, async)
    await remote_to_disk_async(key)
    
    # 2. Disk → CPU (load, async)
    cpu_buffer = await disk_to_cpu_async(key)
    
    # 3. CPU → GPU (when needed, sync)
    kv_cache = await cpu_to_gpu_async(cpu_buffer)
```

**关键优化**：
- **双缓冲**：读写分离
- **批处理**：合并小请求
- **预取**：预测性加载

#### 4. Compression Module（CacheGen）

**算术编码压缩**：

```
原始 KV Cache (FP16)
    ↓
量化 (INT8/INT4)
    ↓
CDF 计算
    ↓
算术编码
    ↓
压缩数据 (2-4x 压缩比)
```

**特点**：
- 无损/有损可配置
- 流式压缩/解压
- CUDA 加速
- 压缩比：2-4x

#### 5. P2P Sharing（点对点共享）

**场景**：多个实例间共享 KV Cache

```
Instance A (Prefill)          Instance B (Decode)
     │                              │
     │  1. Compute KV Cache         │
     ├──────────────────────────────►
     │  2. Send via NIXL/TCP        │
     │                              │
     │                          3. Reuse KV
     │                              │
```

**技术栈**：
- **NIXL**：高性能网络传输库
- **TCP/RDMA**：可选传输协议
- **P2P Discovery**：实例发现机制

#### 6. Disaggregated Prefill

**架构**：

```
       Prefill Cluster            Decode Cluster
    ┌─────────────────┐        ┌────────────────┐
    │  GPU 1, 2, ..., N│        │ GPU 1', 2', ...│
    │                  │        │                │
    │  - Prefill       │───────►│ - Decode       │
    │  - KV Generation │  KV    │ - Generation   │
    └─────────────────┘  Cache  └────────────────┘
```

**优势**：
- **专业化**：Prefill 和 Decode 独立优化
- **弹性伸缩**：根据负载动态调整
- **成本优化**：不同算力需求配置不同硬件

### 4.3 两种工作模式

#### Mode 1: Storage Mode（存储模式）

**用途**：KV Cache 持久化和复用

**流程**：
1. 首次请求：计算 KV Cache → 存储到各级存储
2. 后续请求：查询 Cache Index → 命中则复用
3. 缓存管理：LRU 驱逐、定期压缩

**典型场景**：
- 多轮对话（System Prompt 复用）
- RAG（文档缓存）
- 长文档 QA

#### Mode 2: Transport Mode（传输模式）

**用途**：Disaggregated Prefill

**流程**：
1. Prefill Instance：计算 KV Cache
2. 传输：通过 NIXL/TCP 发送到 Decode Instance
3. Decode Instance：直接使用 KV Cache 生成

**典型场景**：
- 高吞吐推理服务
- Prefill-Decode 分离部署
- 弹性伸缩场景

---

## 二次开发指南

### 5.1 开发环境搭建

#### 从源码安装

```bash
# 1. 克隆仓库
git clone https://github.com/LMCache/LMCache.git
cd LMCache

# 2. 创建虚拟环境
uv venv --python 3.12
source .venv/bin/activate

# 3. 安装构建依赖
uv pip install -r requirements/build.txt

# 4. 安装 PyTorch（匹配推理引擎版本）
uv pip install torch==2.7.1  # 示例：vLLM 0.10.0 使用此版本

# 5. 安装 LMCache（开发模式）
uv pip install -e . --no-build-isolation

# 6. 验证安装
python3 -c "import lmcache.c_ops"  # 测试 CUDA 扩展
```

#### 项目结构

```
LMCache/
├── lmcache/                 # Python 主包
│   ├── __init__.py
│   ├── config.py           # 配置管理
│   ├── cache_engine.py     # 缓存引擎核心
│   ├── integration/        # 框架集成
│   │   ├── vllm/          # vLLM Connector
│   │   └── sglang/        # SGLang Adapter
│   ├── storage_backend/    # 存储后端
│   │   ├── connector/     # 远程连接器
│   │   ├── evictor/       # 驱逐策略
│   │   └── serde/         # 序列化
│   └── v1/                # 新版 API
│       ├── cache_engine.py
│       ├── storage_backend/
│       └── internal_api_server/
│
├── csrc/                    # C++/CUDA 源码
│   ├── ac_enc.cu           # 算术编码
│   ├── ac_dec.cu           # 算术解码
│   ├── mem_kernels.cu      # 内存核函数
│   ├── pos_kernels.cu      # 位置编码
│   └── *.h/*.cpp           # 头文件和辅助
│
├── examples/                # 示例代码
│   ├── basic_check/
│   ├── cache_controller/
│   ├── disagg_prefill/
│   └── kv_cache_reuse/
│
├── tests/                   # 测试
│   ├── v1/
│   └── benchmarks/
│
├── docs/                    # 文档（Sphinx）
├── benchmarks/              # 性能测试
└── setup.py                 # 构建脚本
```

### 5.2 扩展存储后端

#### 步骤 1：实现 Connector 接口

```python
# lmcache/storage_backend/connector/my_storage_connector.py

from lmcache.storage_backend.connector.base_connector import RemoteBytesConnector
from typing import Optional, List

class MyStorageConnector(RemoteBytesConnector):
    """自定义存储后端连接器"""
    
    def __init__(self, host: str, port: int, **kwargs):
        self.host = host
        self.port = port
        # 初始化你的客户端
        self.client = MyStorageClient(host, port)
    
    def exists(self, key: str) -> bool:
        """检查 key 是否存在"""
        return self.client.has(key)
    
    def set(self, key: str, obj: bytes) -> None:
        """存储数据"""
        self.client.put(key, obj)
    
    def get(self, key: str) -> Optional[bytes]:
        """获取数据"""
        try:
            return self.client.get(key)
        except KeyError:
            return None
    
    def list(self) -> List[str]:
        """列出所有 key"""
        return self.client.keys()
    
    def close(self) -> None:
        """关闭连接"""
        self.client.disconnect()
```

#### 步骤 2：注册 Connector

```python
# lmcache/storage_backend/connector/__init__.py

from .my_storage_connector import MyStorageConnector

CONNECTOR_REGISTRY = {
    "redis": RedisConnector,
    "lmcache_server": LMCServerConnector,
    "my_storage": MyStorageConnector,  # 注册你的 Connector
}

def get_connector(backend_type: str, **kwargs):
    connector_cls = CONNECTOR_REGISTRY.get(backend_type)
    if not connector_cls:
        raise ValueError(f"Unknown backend: {backend_type}")
    return connector_cls(**kwargs)
```

#### 步骤 3：配置使用

```yaml
# my_config.yaml
chunk_size: 256
local_device: "cpu"
max_local_cache_size: 10
remote_url: "mystorage://myhost:9000"
remote_serde: "torch"
```

```python
# 使用自定义存储
from lmcache import LMCacheEngine
from lmcache.config import LMCacheEngineConfig

config = LMCacheEngineConfig.from_file("my_config.yaml")
cache_engine = LMCacheEngine(config, metadata)
```

### 5.3 自定义驱逐策略

#### 实现 Evictor

```python
# lmcache/storage_backend/evictor/lfu_evictor.py

from lmcache.storage_backend.evictor.base_evictor import Evictor
from collections import defaultdict
import heapq

class LFUEvictor(Evictor):
    """LFU（Least Frequently Used）驱逐策略"""
    
    def __init__(self):
        self.freq = defaultdict(int)  # key -> 访问频率
        self.time = defaultdict(int)  # key -> 最后访问时间
        self.clock = 0
    
    def update(self, key: str) -> None:
        """更新访问信息"""
        self.freq[key] += 1
        self.time[key] = self.clock
        self.clock += 1
    
    def evict(self) -> str:
        """选择要驱逐的 key"""
        if not self.freq:
            raise ValueError("No keys to evict")
        
        # 选择频率最低的，如果频率相同则选择最旧的
        min_key = min(
            self.freq.keys(),
            key=lambda k: (self.freq[k], self.time[k])
        )
        
        del self.freq[min_key]
        del self.time[min_key]
        return min_key
```

### 5.4 集成新的推理框架

#### 步骤 1：实现 Adapter

```python
# lmcache/integration/myframework/myframework_adapter.py

class MyFrameworkLMCacheAdapter:
    """MyFramework 的 LMCache 适配器"""
    
    def __init__(self, config):
        self.cache_engine = LMCacheEngine(config)
        self.chunk_size = config.chunk_size
    
    def prefill_hook(self, token_ids, kv_cache):
        """Prefill 阶段的 Hook"""
        # 1. 查询缓存
        cached_kv, hit_tokens = self.cache_engine.retrieve(token_ids)
        
        if cached_kv is not None:
            # 缓存命中，返回已有的 KV Cache
            return cached_kv, hit_tokens
        
        # 2. 缓存未命中，返回 None 让框架计算
        return None, 0
    
    def post_prefill_hook(self, token_ids, kv_cache):
        """Prefill 后的 Hook，存储新生成的 KV Cache"""
        # 异步存储到 LMCache
        self.cache_engine.store(token_ids, kv_cache)
    
    def decode_hook(self, token_ids, kv_cache):
        """Decode 阶段的 Hook（可选）"""
        if self.config.save_decode_cache:
            self.cache_engine.store(token_ids, kv_cache)
```

#### 步骤 2：修改推理引擎

```python
# 在推理引擎的 Prefill 阶段插入
class MyFrameworkEngine:
    def __init__(self, model, lmcache_config=None):
        self.model = model
        self.lmcache = None
        
        if lmcache_config:
            self.lmcache = MyFrameworkLMCacheAdapter(lmcache_config)
    
    def prefill(self, token_ids):
        # 1. 尝试从 LMCache 获取
        if self.lmcache:
            cached_kv, hit_tokens = self.lmcache.prefill_hook(token_ids)
            if cached_kv is not None:
                return cached_kv  # 缓存命中，直接返回
        
        # 2. 缓存未命中，正常计算
        kv_cache = self.model.forward_prefill(token_ids)
        
        # 3. 存储到 LMCache
        if self.lmcache:
            self.lmcache.post_prefill_hook(token_ids, kv_cache)
        
        return kv_cache
```

### 5.5 添加 CUDA Kernel

#### 示例：自定义内存拷贝 Kernel

```cuda
// csrc/my_kernel.cu

#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void my_copy_kernel(
    const float* src,
    float* dst,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

torch::Tensor my_copy_cuda(torch::Tensor src) {
    auto dst = torch::empty_like(src);
    
    int size = src.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    my_copy_kernel<<<blocks, threads>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        size
    );
    
    return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_copy", &my_copy_cuda, "My custom copy (CUDA)");
}
```

#### 添加到 setup.py

```python
# setup.py
cuda_sources = [
    "csrc/pybind.cpp",
    "csrc/mem_kernels.cu",
    # ... 其他文件
    "csrc/my_kernel.cu",  # 添加你的 kernel
]
```

### 5.6 性能分析和调试

#### 启用 NVTX 标注

```python
# LMCache 内置 NVTX 支持
from lmcache.utils import _lmcache_nvtx_annotate

@_lmcache_nvtx_annotate
def my_function():
    # 你的代码
    pass
```

#### 使用 Nsight Systems 分析

```bash
# 使用 Nsight Systems 分析性能
nsys profile -o lmcache_profile python my_script.py

# 查看结果
nsys-ui lmcache_profile.nsys-rep
```

#### 调试配置

```python
# 启用调试模式
from lmcache.config import GlobalConfig
GlobalConfig.set_debug(True)

# 查看详细日志
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 5.7 测试你的扩展

```python
# tests/test_my_extension.py

import pytest
import torch
from lmcache.storage_backend.connector import get_connector

def test_my_storage_connector():
    """测试自定义存储 Connector"""
    connector = get_connector(
        "my_storage",
        host="localhost",
        port=9000
    )
    
    # 测试存储和读取
    test_data = b"hello lmcache"
    connector.set("test_key", test_data)
    
    assert connector.exists("test_key")
    retrieved = connector.get("test_key")
    assert retrieved == test_data
    
    connector.close()

def test_integration_with_vllm():
    """测试与 vLLM 的集成"""
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model="facebook/opt-125m",
        kv_transfer_config={
            "kv_connector": "LMCacheConnector",
            "lmcache_config_file": "my_config.yaml"
        }
    )
    
    # 首次推理
    output1 = llm.generate("Hello, how are you?")
    
    # 第二次推理，应该命中缓存
    output2 = llm.generate("Hello, how are you?")
    
    # 验证输出一致
    assert output1 == output2
```

---

## 总结

### 核心要点回顾

#### 1. LMCache 是什么
- **LLM 推理加速引擎扩展**，通过 KV Cache 复用减少 TTFT 和提升吞吐量
- 支持**任意文本片段**的 KV Cache 复用，不仅限于前缀
- 在多轮对话、RAG 等场景下可实现 **3-10x 性能提升**

#### 2. 工程 vs 算法
- **主要是工程项目**：多层级存储、异步流水线、分布式协调
- **融合算法优化**：CacheGen 压缩、CacheBlend 融合
- **与 GPU 深度相关**：支持 NVIDIA（CUDA）和 AMD（ROCm）
- **多级存储设计**：GPU → CPU → Disk → Remote，需要区分设备

#### 3. 框架适配
- **非开箱即用**：需要通过 Connector 集成
- **官方支持**：vLLM（v1 主推）、SGLang
- **企业级支持**：vLLM Production Stack、llm-d、KServe
- **扩展性强**：可以通过实现 Connector 接口支持新框架

#### 4. 二次开发方向

**存储层扩展**：
- 自定义存储后端（实现 RemoteBytesConnector）
- 新的驱逐策略（继承 Evictor）
- 优化序列化/反序列化（serde 模块）

**推理引擎集成**：
- 实现 Adapter 类
- Hook Prefill/Decode 阶段
- 处理分布式场景

**性能优化**：
- 添加 CUDA Kernel
- 优化异步流水线
- 改进缓存策略

**功能增强**：
- 新的压缩算法
- 智能预取策略
- 可观测性增强

### 推荐学习路径

1. **基础理解**（1-2 天）
   - 阅读 README 和文档
   - 运行 `examples/` 中的示例
   - 理解 KV Cache 原理

2. **深入架构**（3-5 天）
   - 阅读核心代码：`lmcache/cache_engine.py`
   - 理解存储后端：`lmcache/storage_backend/`
   - 研究 vLLM 集成：`lmcache/integration/vllm/`

3. **动手实践**（1-2 周）
   - 实现一个简单的存储 Connector
   - 尝试修改驱逐策略
   - 运行 benchmarks 并分析性能

4. **高级开发**（持续）
   - 参与社区讨论（Slack、GitHub）
   - 贡献代码和文档
   - 探索前沿优化方向

### 参考资源

#### 官方资源
- **文档**：https://docs.lmcache.ai/
- **博客**：https://blog.lmcache.ai/
- **Slack**：https://join.slack.com/t/lmcacheworkspace/...
- **GitHub**：https://github.com/LMCache/LMCache

#### 学术论文
1. CacheGen (SIGCOMM 2024)
2. CacheBlend (EuroSys 2025)
3. LLM Content Delivery Network (arXiv 2024)
4. LMCache Architecture (arXiv 2025)

#### 相关项目
- **vLLM**：https://github.com/vllm-project/vllm
- **SGLang**：https://github.com/sgl-project/sglang
- **NIXL**：https://github.com/ai-dynamo/nixl

---


