# LMCache 实现细节与数据流转完全解析

## 目录
1. [整体架构回顾](#整体架构回顾)
2. [核心实现细节](#核心实现细节)
3. [数据流转全过程](#数据流转全过程)
4. [上层使用方式](#上层使用方式)
5. [完整示例](#完整示例)
6. [性能分析](#性能分析)

---

## 整体架构回顾

### 系统分层

```
┌─────────────────────────────────────────────────────────────┐
│                   上层：推理框架                             │
│            (vLLM / SGLang / 其他 LLM Engine)                │
└────────────────────────┬────────────────────────────────────┘
                         │
                    集成层 (Connector)
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  LMCache Core                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Cache Engine│  │ GPU Connector│  │Storage Manager│      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Storage Backends                               │
│  ┌────────────┬──────────────┬────────────┬──────────────┐ │
│  │ Local CPU  │ Local Disk   │ Remote KV  │ P2P Backend  │ │
│  │ Backend    │ Backend      │ Store      │              │ │
│  └────────────┴──────────────┴────────────┴──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心实现细节

### 1. Cache Engine（核心引擎）

**文件位置**：`lmcache/v1/cache_engine.py`

#### 核心数据结构

```python
class LMCacheEngine:
    """LMCache 的核心引擎"""
    
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        # 1. 配置
        self.config = config
        self.metadata = metadata
        
        # 2. GPU Connector：处理 GPU 内存操作
        self.gpu_connector = self._create_gpu_connector()
        
        # 3. Storage Manager：管理多级存储
        self.storage_manager = StorageManager(config, metadata, self.loop)
        
        # 4. 缓存索引：Token Hash → CacheEngineKey
        self.cache_index = {}  # 内存中的哈希表
        
        # 5. Lookup Pins：用于 P2P 传输时的临时固定
        self.lookup_pins = {}  # lookup_id → [keys]
        
        # 6. 事件循环：用于异步操作
        self.loop = asyncio.new_event_loop()
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()
```

#### 关键方法

##### **1. store() - 存储 KV Cache**

```python
def store(
    self,
    tokens: Union[torch.Tensor, List[int]],
    kv_caches,  # vLLM/SGLang 的 KV Cache
    skip_query: bool = False
):
    """
    将 KV Cache 存储到 LMCache
    
    流程：
    1. Token 序列 → 哈希（分块）
    2. GPU Paged Memory → CPU Contiguous Memory
    3. CPU → Local Storage (CPU/Disk)
    4. Local → Remote Storage (异步)
    """
    
    # Step 1: 对 tokens 进行哈希和分块
    chunk_size = self.config.chunk_size  # 默认 256
    token_chunks = self._chunk_tokens(tokens, chunk_size)
    
    for chunk_tokens in token_chunks:
        # 计算哈希
        chunk_hash = self._compute_hash(chunk_tokens)
        
        # Step 2: 检查是否已存在
        key = CacheEngineKey(
            fmt=self.metadata.fmt,
            model=self.metadata.model_name,
            worker_id=self.metadata.worker_id,
            chunk_hash=chunk_hash
        )
        
        if not skip_query and self.storage_manager.contains(key):
            continue  # 已存在，跳过
        
        # Step 3: 从 GPU 提取到 CPU
        memory_obj = self._extract_from_gpu(kv_caches, chunk_tokens)
        
        # Step 4: 提交到 Storage Manager（异步存储）
        self.storage_manager.submit_put_task(key, memory_obj)
```

**关键点**：
- Token 序列按 `chunk_size`（默认 256）分块
- 每个块计算哈希作为 key
- GPU → CPU 同步（快）
- CPU → Disk/Remote 异步（不阻塞）

##### **2. retrieve() - 检索 KV Cache**

```python
def retrieve(
    self,
    tokens: Union[torch.Tensor, List[int]],
    kv_caches  # 目标：vLLM/SGLang 的 KV Cache（待填充）
) -> int:
    """
    从 LMCache 检索 KV Cache
    
    返回：命中的 token 数量
    """
    
    # Step 1: Token 序列分块和哈希
    chunk_size = self.config.chunk_size
    token_chunks = self._chunk_tokens(tokens, chunk_size)
    
    hit_tokens = 0
    
    for i, chunk_tokens in enumerate(token_chunks):
        chunk_hash = self._compute_hash(chunk_tokens)
        key = CacheEngineKey(...)
        
        # Step 2: 从 Storage Manager 获取 MemoryObj
        memory_obj = self.storage_manager.get(key)
        
        if memory_obj is None:
            break  # 未命中，停止（前缀匹配）
        
        # Step 3: 从 CPU 加载到 GPU
        self._load_to_gpu(memory_obj, kv_caches, chunk_tokens)
        
        hit_tokens += len(chunk_tokens)
    
    return hit_tokens
```

**关键点**：
- 前缀匹配：一旦某个块未命中，后续块也不检索
- CPU → GPU 同步加载
- 支持部分命中（前 N 个块命中）

---

### 2. GPU Connector（GPU 接口）

**文件位置**：`lmcache/v1/gpu_connector.py`

#### 作用

GPU Connector 负责：
1. **从 GPU 提取**：vLLM Paged Memory → LMCache Contiguous Memory
2. **加载到 GPU**：LMCache Contiguous Memory → vLLM Paged Memory

#### 为什么需要？

vLLM 的 KV Cache 是 **分页存储**（PagedAttention）：

```
vLLM Paged Memory（不连续）：
Page 0: [Token 0, Token 1, Token 3]
Page 1: [Token 5, Token 6]
Page 2: [Token 2, Token 4]
         ↑ 乱序、分散

需要转换为：

LMCache Contiguous Memory（连续）：
[Token 0][Token 1][Token 2][Token 3][Token 4][Token 5][Token 6]
    ↑ 有序、连续
```

#### 核心实现

```python
class VLLMPagedMemGPUConnectorV2(GPUConnectorInterface):
    """vLLM 的 GPU Connector"""
    
    def __init__(self, kvcaches, metadata, ...):
        self.kvcaches = kvcaches  # vLLM 的 paged KV cache
        self.num_layers = metadata.kv_shape[0]
        self.page_buffer_size = ...
        
        # 创建专用的 CUDA Stream
        self.store_stream = torch.cuda.Stream()  # 用于存储（GPU→CPU）
        self.load_stream = torch.cuda.Stream()   # 用于加载（CPU→GPU）
    
    def from_gpu(
        self,
        memory_obj: MemoryObj,  # 目标：CPU 上的连续内存
        start: int,             # 起始 token 索引
        end: int,               # 结束 token 索引
        slot_mapping: torch.Tensor  # Token → Page 的映射
    ):
        """
        从 vLLM Paged Memory 提取到 LMCache Contiguous Memory
        """
        
        # 获取所有层的 KV cache 指针
        kv_cache_pointers = self._initialize_pointers(self.kvcaches)
        
        # 使用专用 Stream（不阻塞主 Stream）
        with torch.cuda.stream(self.store_stream):
            # 调用 CUDA Kernel
            lmc_ops.multi_layer_kv_transfer(
                memory_obj.tensor,          # 目标（CPU pinned memory）
                kv_cache_pointers,          # 源（GPU paged memory）
                slot_mapping[start:end],    # 映射关系
                self.kvcaches[0].device,    # GPU 设备
                self.page_buffer_size,
                True,                       # 方向：paged → contiguous
                self.use_mla
            )
        
        # 如果目标不在 GPU 上，同步等待
        if not memory_obj.tensor.is_cuda:
            self.store_stream.synchronize()
    
    def to_gpu(
        self,
        memory_obj: MemoryObj,  # 源：CPU 上的连续内存
        start: int,
        end: int,
        slot_mapping: torch.Tensor
    ):
        """
        从 LMCache Contiguous Memory 加载到 vLLM Paged Memory
        """
        
        kv_cache_pointers = self._initialize_pointers(self.kvcaches)
        
        with torch.cuda.stream(self.load_stream):
            lmc_ops.multi_layer_kv_transfer(
                memory_obj.tensor,
                kv_cache_pointers,
                slot_mapping[start:end],
                self.kvcaches[0].device,
                self.page_buffer_size,
                False,  # 方向：contiguous → paged
                self.use_mla
            )
        
        self.load_stream.synchronize()
```

**关键优化**：
- **Pinned Memory**：CPU 侧使用锁页内存，加速 GPU-CPU 传输
- **专用 Stream**：避免阻塞主 Stream
- **批量处理**：一次处理所有层

---

### 3. Storage Manager（存储管理器）

**文件位置**：`lmcache/v1/storage_backend/storage_manager.py`

#### 核心职责

```python
class StorageManager:
    """
    管理多级存储后端
    
    层级：
    1. Local CPU Backend（热缓存，快）
    2. Local Disk Backend（冷缓存，中速）
    3. Remote Backend（持久化，慢）
    4. P2P Backend（跨实例共享）
    """
    
    def __init__(self, config, metadata, loop):
        self.storage_backends = {}
        
        # 1. Local CPU Backend（必须）
        self.storage_backends["LocalCPUBackend"] = LocalCPUBackend(
            config, metadata, loop
        )
        
        # 2. Local Disk Backend（可选）
        if config.local_device and config.local_device.startswith("/"):
            self.storage_backends["LocalDiskBackend"] = LocalDiskBackend(
                config, metadata, loop
            )
        
        # 3. Remote Backend（可选）
        if config.remote_url:
            self.storage_backends["RemoteBackend"] = RemoteBackend(
                config, metadata, loop
            )
        
        # 4. P2P Backend（可选）
        if config.enable_p2p:
            self.storage_backends["P2PBackend"] = P2PBackend(
                config, metadata, loop, ...
            )
        
        # 异步任务队列
        self.loop = loop
    
    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        提交存储任务（异步）
        
        流程：
        1. CPU Backend（同步）
        2. Disk Backend（异步）← 如果 CPU 已满，LRU 驱逐到 Disk
        3. Remote Backend（异步）
        """
        
        # 提交到 CPU Backend
        future = asyncio.run_coroutine_threadsafe(
            self.storage_backends["LocalCPUBackend"].async_submit_put_task(
                key, memory_obj
            ),
            self.loop
        )
        
        # 不等待完成，继续执行（异步）
    
    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        获取缓存（同步）
        
        查找顺序：
        1. Local CPU（最快）
        2. Local Disk（中速）
        3. Remote（最慢）
        4. P2P（跨实例）
        """
        
        # 1. 先查 CPU
        mem_obj = self.storage_backends["LocalCPUBackend"].get(key)
        if mem_obj:
            return mem_obj
        
        # 2. 再查 Disk
        if "LocalDiskBackend" in self.storage_backends:
            mem_obj = self.storage_backends["LocalDiskBackend"].get(key)
            if mem_obj:
                # 提升到 CPU（热度提升）
                self.storage_backends["LocalCPUBackend"].put(key, mem_obj)
                return mem_obj
        
        # 3. 最后查 Remote
        if "RemoteBackend" in self.storage_backends:
            mem_obj = self.storage_backends["RemoteBackend"].get(key)
            if mem_obj:
                # 提升到 CPU
                self.storage_backends["LocalCPUBackend"].put(key, mem_obj)
                return mem_obj
        
        return None
```

---

### 4. Local CPU Backend（本地 CPU 后端）

**文件位置**：`lmcache/v1/storage_backend/local_cpu_backend.py`

#### 核心实现

```python
class LocalCPUBackend(StorageBackendInterface):
    """
    本地 CPU 内存后端
    
    特点：
    - 使用 Pinned Memory（锁页内存）
    - LRU 驱逐策略
    - 高速访问
    """
    
    def __init__(self, config, metadata, loop):
        # 内存分配器
        self.memory_allocator = PagedCpuGpuMemoryAllocator(
            buffer_size=config.max_local_cache_size * 1024**3,  # GB → Bytes
            page_size=config.chunk_size * hidden_size * 2 * dtype_size,
            device="cpu",
            pin_memory=True  # 关键！使用锁页内存
        )
        
        # 缓存索引：key → memory_obj
        self.cache = {}
        
        # LRU 驱逐器
        self.evictor = LRUEvictor()
        
        # 异步任务队列
        self.put_queue = asyncio.Queue()
        self.loop = loop
    
    def allocate(self, shape, dtype, fmt) -> MemoryObj:
        """
        分配 CPU 内存（Pinned Memory）
        """
        
        # 如果内存不足，驱逐旧条目
        while not self.memory_allocator.has_space(shape):
            evict_key = self.evictor.evict()
            self._evict_to_disk(evict_key)  # 驱逐到 Disk
        
        # 从内存池分配
        tensor = self.memory_allocator.allocate(shape, dtype)
        
        # 创建 MemoryObj
        memory_obj = MemoryObj(
            tensor=tensor,
            metadata=MemoryMetadata(shape=shape, dtype=dtype, fmt=fmt)
        )
        
        return memory_obj
    
    def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        存储到 CPU 缓存
        """
        self.cache[key] = memory_obj
        self.evictor.update(key)  # 更新 LRU
    
    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        从 CPU 缓存读取
        """
        if key in self.cache:
            self.evictor.update(key)  # 更新 LRU
            return self.cache[key]
        return None
```

**关键优化**：
- **Pinned Memory**：GPU 可以直接 DMA 访问
- **LRU 驱逐**：内存满时自动驱逐最少使用的
- **异步写入**：不阻塞主线程

---

## 数据流转全过程

### 场景 1：首次存储 KV Cache

```
┌──────────────────────────────────────────────────────────────┐
│                    vLLM Inference                            │
│  1. 模型推理，生成 KV Cache（GPU Paged Memory）              │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              LMCache.store(tokens, kv_caches)                │
│                                                              │
│  Step 1: Token 序列分块和哈希                                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ tokens: [1, 2, 3, ..., 512]                           │ │
│  │    ↓ 按 chunk_size=256 分块                            │ │
│  │ chunks: [[1...256], [257...512]]                      │ │
│  │    ↓ 计算哈希                                          │ │
│  │ hashes: [hash1, hash2]                                │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Step 2: 从 GPU 提取（对每个 chunk）                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ GPU Connector.from_gpu()                              │ │
│  │   ├─ vLLM Paged Memory (GPU, 不连续)                  │ │
│  │   │     [Page 0][Page 2][Page 5]...                   │ │
│  │   ↓ CUDA Kernel: multi_layer_kv_transfer             │ │
│  │   ├─ LMCache Contiguous Memory (CPU Pinned, 连续)     │ │
│  │   │     [Token 0][Token 1][Token 2]...                │ │
│  │   └─ MemoryObj {tensor, metadata}                     │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│         ▼                                                    │
│  Step 3: 提交到 Storage Manager                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ storage_manager.submit_put_task(key, memory_obj)      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼ 异步处理
┌──────────────────────────────────────────────────────────────┐
│                  Storage Manager 异步任务                     │
│                                                              │
│  Step 4a: 存储到 Local CPU Backend                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ LocalCPUBackend.put(key, memory_obj)                  │ │
│  │   ├─ 检查内存空间                                      │ │
│  │   ├─ 如果满了，LRU 驱逐                                 │ │
│  │   └─ 存入 cache[key] = memory_obj                     │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│         ▼                                                    │
│  Step 4b: 异步 Offload 到 Disk（如果配置）                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ LocalDiskBackend.async_put(key, memory_obj)           │ │
│  │   ├─ 序列化 memory_obj                                 │ │
│  │   ├─ 压缩（可选）                                       │ │
│  │   └─ 写入磁盘文件                                       │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│         ▼                                                    │
│  Step 4c: 异步上传到 Remote（如果配置）                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ RemoteBackend.async_put(key, memory_obj)              │ │
│  │   ├─ 序列化                                            │ │
│  │   └─ 发送到 Redis/S3/...                               │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**时间线**：
- Step 1-3：**同步**（快，< 1ms）
- Step 4a：**同步**（快，内存操作）
- Step 4b-c：**异步**（慢，不阻塞推理）

---

### 场景 2：检索和复用 KV Cache

```
┌──────────────────────────────────────────────────────────────┐
│                    vLLM Inference                            │
│  收到新请求，tokens 前缀可能已缓存                            │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│           hit_tokens = LMCache.retrieve(tokens)              │
│                                                              │
│  Step 1: Token 序列分块和哈希                                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ tokens: [1, 2, 3, ..., 512]                           │ │
│  │    ↓ 分块                                              │ │
│  │ chunks: [[1...256], [257...512]]                      │ │
│  │    ↓ 哈希                                              │ │
│  │ hashes: [hash1, hash2]                                │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Step 2: 查询 Storage Manager（对每个 chunk）                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ For chunk in chunks:                                  │ │
│  │   memory_obj = storage_manager.get(key)               │ │
│  │                                                       │ │
│  │   查找顺序：                                           │ │
│  │   1. Local CPU Backend (命中率 80-90%)                │ │
│  │      └─ 直接返回 memory_obj                           │ │
│  │                                                       │ │
│  │   2. Local Disk Backend (命中率 10-20%)               │ │
│  │      ├─ 从磁盘读取                                     │ │
│  │      ├─ 解压缩                                         │ │
│  │      └─ 提升到 CPU Backend                            │ │
│  │                                                       │ │
│  │   3. Remote Backend (命中率 < 5%)                     │ │
│  │      ├─ 从 Redis/S3 获取                               │ │
│  │      └─ 提升到 CPU Backend                            │ │
│  │                                                       │ │
│  │   4. 未命中                                            │ │
│  │      └─ 停止查询（前缀匹配）                           │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│         ▼                                                    │
│  Step 3: 加载到 GPU（对命中的 chunk）                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ GPU Connector.to_gpu()                                │ │
│  │   ├─ LMCache Contiguous Memory (CPU Pinned)           │ │
│  │   │     [Token 0][Token 1][Token 2]...                │ │
│  │   ↓ CUDA Kernel: multi_layer_kv_transfer             │ │
│  │   └─ vLLM Paged Memory (GPU)                          │ │
│  │         [Page 0][Page 2][Page 5]...                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Return: hit_tokens (已复用的 token 数量)                   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    vLLM Inference                            │
│  跳过已复用的 tokens，只计算剩余部分                          │
│  ├─ 已复用：tokens[0:hit_tokens]  ← 直接使用缓存             │
│  └─ 需计算：tokens[hit_tokens:]   ← 模型推理                │
└──────────────────────────────────────────────────────────────┘
```

**性能关键点**：
- **CPU 命中**：< 1ms（内存读取）
- **Disk 命中**：10-50ms（磁盘 I/O）
- **Remote 命中**：50-200ms（网络 I/O）

---

### 场景 3：跨实例 P2P 共享

```
Worker A (有缓存)                        Worker B (需要缓存)
    │                                         │
    │ 1. Worker B 发起查询                     │
    │◄─────── P2P Lookup ──────────────────────┤
    │   { keys: [hash1, hash2] }               │
    │                                         │
    │ 2. Worker A 检查本地缓存                  │
    ├─ LocalCPUBackend.contains(hash1) ✓     │
    ├─ LocalCPUBackend.contains(hash2) ✓     │
    │                                         │
    │ 3. Worker A 响应                         │
    ├─────── P2P Lookup Response ────────────►│
    │   { hits: [hash1, hash2], location }     │
    │                                         │
    │ 4. Worker B 发起 P2P 传输                │
    │◄─────── BatchedLookupAndPutMsg ──────────┤
    │   { sender_id: A, mem_indexes }          │
    │                                         │
    │ 5. Worker B 分配本地内存                  │
    │                                         ├─ allocate()
    │                                         │
    │ 6. NIXL 直接传输（GPU → GPU）             │
    ├═══════════════════════════════════════►│
    │   GPU Direct Transfer (RDMA)             │
    ├═══════════════════════════════════════►│
    │                                         │
    │                                         │ 7. Worker B 存入本地缓存
    │                                         ├─ LocalCPUBackend.put()
    │                                         │
    │ 8. Worker B 响应完成                     │
    │◄─────── BatchedLookupAndPutRetMsg ───────┤
    │   { num_read_chunks: 2 }                 │
    │                                         │
    ▼                                         ▼
```

**性能**：
- 延迟：2-5 ms（GPU Direct Transfer）
- 带宽：10-25 GB/s（接近 NVLink/PCIe 带宽）

---

## 上层使用方式

### 1. vLLM 集成

**集成点**：`vllm/worker/model_runner.py`

#### 初始化

```python
# vLLM Worker 启动时
class ModelRunner:
    def __init__(self, ...):
        # 如果启用 LMCache
        if kv_transfer_config and kv_transfer_config["kv_connector"] == "LMCacheConnector":
            # 创建 LMCache Connector
            self.lmcache_connector = LMCacheConnector(
                config_file=kv_transfer_config.get("lmcache_config_file"),
                kvcaches=self.kv_caches,  # vLLM 的 KV Cache
                metadata=...
            )
```

#### Prefill 阶段（生成 KV Cache）

```python
def prefill(self, seq_group_metadata_list):
    """
    Prefill 阶段：处理输入 tokens，生成 KV Cache
    """
    
    # Step 1: 尝试从 LMCache 检索
    if self.lmcache_connector:
        for seq_group in seq_group_metadata_list:
            tokens = seq_group.token_ids
            
            # 检索缓存
            hit_tokens = self.lmcache_connector.retrieve(tokens)
            
            if hit_tokens > 0:
                # 更新 seq_group，标记哪些 tokens 已缓存
                seq_group.cached_tokens = hit_tokens
    
    # Step 2: 只计算未缓存的部分
    for seq_group in seq_group_metadata_list:
        if seq_group.cached_tokens < len(seq_group.token_ids):
            # 从 cached_tokens 开始计算
            new_tokens = seq_group.token_ids[seq_group.cached_tokens:]
            
            # 模型推理（生成新的 KV Cache）
            logits, new_kv = self.model.forward(new_tokens, ...)
    
    # Step 3: 存储新生成的 KV Cache
    if self.lmcache_connector:
        for seq_group in seq_group_metadata_list:
            if seq_group.cached_tokens < len(seq_group.token_ids):
                # 存储完整的 token 序列
                self.lmcache_connector.store(
                    seq_group.token_ids,
                    self.kv_caches
                )
```

**关键点**：
- **先检索**：每个请求先尝试从 LMCache 获取
- **部分计算**：只计算未缓存的 tokens
- **后存储**：新生成的 KV Cache 存入 LMCache

#### Decode 阶段（生成 token）

```python
def decode(self, seq_group_metadata_list):
    """
    Decode 阶段：生成下一个 token
    """
    
    # Decode 阶段通常不使用 LMCache
    # （除非 config.save_decode_cache = True）
    
    for seq_group in seq_group_metadata_list:
        # 正常 decode
        next_token = self.model.forward_decode(...)
```

---

### 2. 配置方式

#### 方式 1：命令行参数

```bash
vllm serve meta-llama/Llama-2-7b-hf \
  --kv-transfer-config '{
    "kv_connector": "LMCacheConnector",
    "kv_buffer_size": 1e9
  }'
```

#### 方式 2：配置文件

```yaml
# lmcache_config.yaml
chunk_size: 256
local_device: "cpu"
max_local_cache_size: 10  # GB

# 可选：Remote Storage
remote_url: "redis://localhost:6379"
remote_serde: "torch"

# 可选：P2P
enable_p2p: true
p2p_host: "localhost"
p2p_init_ports: [8200, 8202]
transfer_channel: "nixl"
```

```bash
export LMCACHE_CONFIG_FILE=lmcache_config.yaml

vllm serve meta-llama/Llama-2-7b-hf \
  --kv-transfer-config '{
    "kv_connector": "LMCacheConnector"
  }'
```

#### 方式 3：Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_transfer_config={
        "kv_connector": "LMCacheConnector",
        "kv_buffer_size": 1e9,
    },
    # 或通过环境变量指定配置文件
)

# 正常使用
outputs = llm.generate("Hello, how are you?", sampling_params)
```

---

### 3. SGLang 集成

```python
import sglang as sgl

# 设置 LMCache 配置
lmcache_config = {
    "local_device": "cpu",
    "max_local_cache_size": 10,
}

# 启动 SGLang Runtime
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-7b-hf",
    lmcache_config=lmcache_config
)

# 使用
@sgl.function
def chat(s, question):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer"))

state = chat.run(question="What is AI?")
```

---

## 完整示例

### 示例 1：多轮对话（System Prompt 复用）

```python
from vllm import LLM, SamplingParams

# 初始化 LMCache
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_transfer_config={
        "kv_connector": "LMCacheConnector",
    }
)

sampling_params = SamplingParams(max_tokens=100)

# System Prompt（固定）
system_prompt = "You are a helpful AI assistant. " * 100  # 假设很长

# Round 1：首次对话
prompt1 = system_prompt + "\nUser: What is AI?\nAssistant:"
output1 = llm.generate(prompt1, sampling_params)
# ↑ LMCache 存储了 system_prompt 的 KV Cache

# Round 2：第二次对话
prompt2 = system_prompt + "\nUser: Tell me about ML.\nAssistant:"
output2 = llm.generate(prompt2, sampling_params)
# ↑ LMCache 复用了 system_prompt 的 KV Cache（命中！）
#   只需要计算 "User: Tell me about ML." 的部分

# Round 3：第三次对话
prompt3 = system_prompt + "\nUser: Explain deep learning.\nAssistant:"
output3 = llm.generate(prompt3, sampling_params)
# ↑ 再次复用 system_prompt（命中！）
```

**性能提升**：
- Round 1：100% 计算（无缓存）
- Round 2：只计算 10%（system_prompt 占 90%）
- Round 3：只计算 10%

**TTFT 降低**：5-10x

---

### 示例 2：RAG（文档缓存）

```python
from vllm import LLM

llm = LLM(model="...", kv_transfer_config={"kv_connector": "LMCacheConnector"})

# 长文档（每次 RAG 都检索到）
long_document = "..." * 10000  # 10k tokens

# Query 1
query1 = f"Document: {long_document}\n\nQuestion: Who is the author?\nAnswer:"
answer1 = llm.generate(query1)
# ↑ 存储 long_document 的 KV Cache

# Query 2（不同问题，但同一文档）
query2 = f"Document: {long_document}\n\nQuestion: What is the main idea?\nAnswer:"
answer2 = llm.generate(query2)
# ↑ 复用 long_document 的 KV Cache（命中！）

# Query 3
query3 = f"Document: {long_document}\n\nQuestion: Summarize it.\nAnswer:"
answer3 = llm.generate(query3)
# ↑ 再次复用（命中！）
```

**性能提升**：
- Query 1：计算整个文档（10k tokens）
- Query 2-3：只计算问题部分（< 20 tokens）

**GPU 计算节省**：500x

---

## 性能分析

### 关键性能指标

#### 1. 缓存命中率

```
缓存命中率 = 复用的 tokens / 总 tokens
```

**典型值**：
- 多轮对话：80-95%（System Prompt 占大部分）
- RAG：70-90%（文档占大部分）
- 长文档 QA：90-95%

#### 2. TTFT（Time To First Token）

```
TTFT = Prefill 时间 + Decode 首 token 时间
```

**LMCache 优化**：

```
无 LMCache：
TTFT = T_prefill(all_tokens) + T_decode
     ≈ 1000ms + 50ms = 1050ms

有 LMCache（90% 命中）：
TTFT = T_retrieve + T_prefill(10% tokens) + T_decode
     ≈ 1ms + 100ms + 50ms = 151ms

提升：1050ms / 151ms ≈ 7x
```

#### 3. 吞吐量

```
吞吐量 = 请求数 / 时间
```

**提升原因**：
- Prefill 时间减少 → GPU 更快处理下一个请求
- GPU 利用率提高

**典型提升**：3-5x

---

### 性能瓶颈分析

#### 瓶颈 1：CPU-GPU 传输

```python
# 优化前（慢）
kv_cache = gpu_tensor.cpu()  # GPU → CPU（阻塞）

# 优化后（快）
kv_cache = torch.empty(..., pin_memory=True)  # Pinned Memory
gpu_tensor.copy_(kv_cache, non_blocking=True)  # 异步拷贝
```

**优化效果**：2-3x

#### 瓶颈 2：Disk I/O

```python
# 优化前（慢）
with open(path, 'wb') as f:
    torch.save(kv_cache, f)  # 同步写入

# 优化后（快）
async def async_write():
    # 异步写入，不阻塞主线程
    await asyncio.to_thread(torch.save, kv_cache, path)
```

**优化效果**：不阻塞推理

#### 瓶颈 3：Remote 网络延迟

```python
# 优化：Lazy Upload
# - CPU Backend：立即可用（< 1ms）
# - Remote Backend：异步上传（不阻塞）

storage_manager.submit_put_task(key, memory_obj)
# ↑ 立即返回，后台上传
```

---

## 总结

### 核心实现要点

1. **分层设计**：
   - Cache Engine：业务逻辑
   - GPU Connector：GPU 接口
   - Storage Manager：存储协调
   - Storage Backends：多级存储

2. **异步处理**：
   - Prefill：同步检索 + 异步存储
   - 不阻塞推理主线程

3. **多级缓存**：
   - CPU（热）→ Disk（温）→ Remote（冷）
   - 自动提升热度

### 数据流转关键路径

```
存储：GPU Paged → CPU Contiguous → Storage Backends
检索：Storage Backends → CPU Contiguous → GPU Paged
```

### 上层使用模式

```python
# 1. 配置 LMCache
kv_transfer_config = {"kv_connector": "LMCacheConnector"}

# 2. 创建 LLM
llm = LLM(model="...", kv_transfer_config=kv_transfer_config)

# 3. 正常推理（自动缓存复用）
outputs = llm.generate(prompt)
```

### 性能关键因素

| 因素 | 影响 | 优化方法 |
|-----|------|---------|
| 缓存命中率 | 直接影响 TTFT | 增加 cache_size、调整 chunk_size |
| CPU-GPU 传输 | Prefill 延迟 | Pinned Memory、异步拷贝 |
| 存储层级 | 命中延迟 | 优先 CPU、Lazy Upload |

---

**完整的 LMCache 实现细节和数据流转解析完成！** 🎉

希望这篇文档能帮你深入理解 LMCache 的工作原理！

