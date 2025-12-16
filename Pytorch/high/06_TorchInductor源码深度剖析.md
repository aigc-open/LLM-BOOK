# TorchInductor 源码深度剖析

## 为什么需要 TorchInductor？从 Dynamo 到机器码的桥梁

### TorchDynamo 的局限

在学习完 TorchDynamo 后，我们知道 Dynamo 通过字节码级追踪可以捕获完整的计算图（FX Graph），但它有一个关键问题：

```python
# TorchDynamo 的输出
@torch.compile
def model(x, y):
    z = x + y
    return z.relu()

# Dynamo 生成的 FX Graph
def forward(x, y):
    add_tensor = torch.ops.aten.add.Tensor(x, y)
    relu_default = torch.ops.aten.relu.default(add_tensor)
    return relu_default

# 问题：这只是一个计算图，还不是可执行的高效代码！
```

**TorchDynamo 只负责捕获图，不负责执行优化**：
- [X] 生成了 FX Graph（计算图的中间表示）
- [X] 但没有生成优化的 GPU/CPU 代码
- [X] 没有算子融合、内存优化等
- [X] 仍然需要逐个调用 PyTorch 算子

### TorchInductor 的使命

**TorchInductor 是 PyTorch 编译栈的"代码生成器"，负责将 FX Graph 编译为高效的机器码。**

```python
# 完整的编译流程
用户代码
    ↓
[TorchDynamo] 字节码追踪 → FX Graph
    ↓
[AOTAutograd] 前向/反向分离 → 优化的 FX Graph
    ↓
[TorchInductor] 代码生成 → 优化的机器码  ← 本文重点！
    ↓
GPU/CPU 执行
```

### 技术对比

| 维度 | TorchDynamo | TorchInductor |
|------|------------|--------------|
| **职责** | 捕获计算图 | 生成优化代码 |
| **输入** | Python 字节码 | FX Graph |
| **输出** | FX Graph | Triton/C++ 代码 |
| **核心技术** | 字节码分析 + Guard | Lowering + Fusion + CodeGen |
| **优化** | 无（只捕获） | 算子融合、内存规划、循环优化 |
| **性能提升** | 0x（只是捕获） | 1.5-5x（真正的加速） |

### 核心设计思想

```
TorchDynamo:  分析字节码 → 构建 FX Graph
              [静态分析]   [符号执行]

TorchInductor: 分析 FX Graph → 生成优化代码 → 编译执行
               [Lowering]      [Fusion]      [CodeGen]
```

---

## TorchInductor 是什么？

### 一句话总结

**TorchInductor 是 PyTorch 的 JIT 编译器后端，通过 Lowering、Fusion、Scheduling 等技术，将 FX Graph 编译为高效的 Triton/C++ 代码，实现端到端的性能优化。**

### 核心组成

```
TorchInductor 架构：
├─ Graph Lowering      [转换] FX Graph → IR (Intermediate Representation)
├─ Scheduler           [调度] 算子融合决策、内存规划
├─ Code Generation     [生成] IR → Triton/C++ 源代码
│   ├─ Triton CodeGen  (GPU)
│   ├─ C++ CodeGen     (CPU)
│   └─ Wrapper CodeGen (调用包装)
├─ Compilation         [编译] 源代码 → 机器码
│   ├─ Triton Compiler → PTX/CUBIN
│   └─ C++ Compiler    → .so
└─ Execution           [执行] 动态加载并运行
```

### 工作流程

```
FX Graph (来自 Dynamo)
    |
    v
[1] Graph Lowering
    |
    | 将高层算子转换为低层 IR
    |
    v
IR Nodes (Pointwise, Reduction, etc.)
    |
    v
[2] Scheduling & Fusion
    |
    | 决定哪些算子可以融合
    | 生成内存规划
    |
    v
Fused IR Groups
    |
    v
[3] Code Generation
    |
    | 为每个融合组生成 Triton/C++ 代码
    |
    v
Source Code (Triton/C++)
    |
    v
[4] Compilation
    |
    | Triton → PTX → CUBIN
    | C++ → .so
    |
    v
Machine Code
    |
    v
[5] Execution
    |
    | 动态加载并执行
    |
    v
Result
```

---

## 源码文件索引

TorchInductor 的核心代码位于 `torch/_inductor/` 目录：

```
torch/_inductor/
├── compile_fx.py              # [入口] 编译 FX Graph 的主入口
├── graph.py                   # [核心] GraphLowering - 图降低与转换
├── lowering.py                # [核心] FX 算子 → IR 的 lowering 规则
├── ir.py                      # [核心] IR 节点定义 (Pointwise, Reduction, etc.)
├── scheduler.py               # [核心] Scheduler - 融合决策与调度
├── dependencies.py            # 依赖分析
├── sizevars.py                # 符号化形状处理
├── codegen/                   # 代码生成
│   ├── common.py             # 公共代码生成工具
│   ├── triton.py             # [核心] Triton 代码生成
│   ├── cpp.py                # C++ 代码生成
│   ├── wrapper.py            # Wrapper 代码生成
│   └── simd.py               # SIMD 优化
├── kernel/                    # 特殊 Kernel
│   ├── mm.py                 # 矩阵乘法
│   ├── conv.py               # 卷积
│   └── ...
├── fx_passes/                 # FX Graph 优化 Pass
│   ├── joint_graph.py        # 联合图优化
│   ├── pre_grad.py           # 梯度前优化
│   └── post_grad.py          # 梯度后优化
├── runtime/                   # 运行时支持
│   ├── hints.py              # 启发式提示
│   └── triton_heuristics.py  # Triton 启发式
└── utils.py                   # 工具函数
```

---

## 第一部分：整体架构与编译流程

### 章节 1：从 FX Graph 到机器码的完整流程

#### 1.1 编译入口

```python
# torch/_inductor/compile_fx.py

def compile_fx(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    inner_compile=None,
):
    """
    编译 FX GraphModule
    
    Args:
        gm: FX GraphModule (来自 TorchDynamo)
        example_inputs: 示例输入（用于形状推导）
        inner_compile: 内部编译函数（可选）
    
    Returns:
        compiled_fn: 编译后的可执行函数
    """
    # [1] 创建 GraphLowering 实例
    with V.set_graph_handler(GraphLowering(gm, example_inputs=example_inputs)):
        # [2] 执行 Lowering（FX Graph → IR）
        graph_handler = V.graph
        graph_handler.run(*example_inputs)
        
        # [3] 编译为可执行函数
        compiled_fn = graph_handler.compile_to_fn()
    
    return compiled_fn
```

**执行流程时间线**：

```
t=0ms:   compile_fx() 被调用
         ↓
t=1ms:   创建 GraphLowering 实例
         ↓
t=2ms:   开始 Lowering (FX Graph → IR)
         ├─→ 遍历 FX Graph 的每个节点
         ├─→ 调用对应的 lowering 函数
         └─→ 生成 IR 节点
         ↓
t=50ms:  Lowering 完成
         ↓
t=51ms:  开始 Scheduling
         ├─→ 分析依赖关系
         ├─→ 融合决策
         └─→ 内存规划
         ↓
t=100ms: Scheduling 完成
         ↓
t=101ms: 开始 Code Generation
         ├─→ 为每个融合组生成 Triton 代码
         ├─→ 生成 Wrapper 代码
         └─→ 生成调用代码
         ↓
t=150ms: Code Generation 完成
         ↓
t=151ms: 开始 Compilation
         ├─→ Triton 编译为 PTX
         ├─→ PTX 编译为 CUBIN
         └─→ 动态加载
         ↓
t=300ms: Compilation 完成
         ↓
t=301ms: 返回 compiled_fn
```

#### 1.2 V.graph 上下文管理器

```python
# torch/_inductor/virtualized.py

class V:
    """
    全局上下文管理器
    
    提供对当前 GraphLowering 实例的访问
    类似于线程局部存储（Thread Local Storage）
    """
    _graph_handler = None
    
    @staticmethod
    def set_graph_handler(graph):
        """设置当前的 GraphLowering 实例"""
        V._graph_handler = graph
        return contextlib.contextmanager(lambda: (yield))()
    
    @property
    def graph(self):
        """获取当前的 GraphLowering 实例"""
        return V._graph_handler

# 使用方式
with V.set_graph_handler(GraphLowering(gm)):
    # 在这个上下文中，所有 lowering 函数都可以通过 V.graph 访问当前图
    V.graph.register_buffer("buf0", some_ir_node)
```

#### 1.3 GraphLowering - 核心控制器

```python
# torch/_inductor/graph.py

class GraphLowering:
    """
    图降低器 - TorchInductor 的核心控制器
    
    职责：
    1. 将 FX Graph 转换为 IR
    2. 管理缓冲区（buffers）
    3. 协调 Scheduler
    4. 生成最终代码
    """
    
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs=None,
    ):
        self.gm = gm
        self.graph = gm.graph
        self.example_inputs = example_inputs
        
        # 缓冲区管理
        self.buffers = {}  # name → IRNode
        self.constants = {}  # name → Tensor
        
        # 图输入/输出
        self.graph_inputs = {}
        self.graph_outputs = []
        
        # Scheduler（稍后创建）
        self.scheduler = None
    
    def run(self, *args):
        """
        执行 Lowering
        
        遍历 FX Graph，将每个节点转换为 IR
        """
        # 一次遍历处理所有节点
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                # [1] 处理输入节点
                self.graph_inputs[node.name] = self.wrap_input(node, args)
            
            elif node.op == 'call_function':
                # [2] 处理计算节点 - 调用对应的 lowering 函数
                self.call_function(node)
            
            elif node.op == 'call_method':
                self.call_method(node)
            
            elif node.op == 'get_attr':
                self.get_attr(node)
            
            elif node.op == 'output':
                # [3] 处理输出节点
                self.graph_outputs = node.args[0]
    
    def call_function(self, node):
        """
        处理 call_function 节点
        
        例如：torch.ops.aten.add.Tensor
        """
        target = node.target
        args = self.fetch_args(node.args)
        kwargs = self.fetch_kwargs(node.kwargs)
        
        # 查找对应的 lowering 函数
        # lowerings 是一个全局字典，映射 ATen 算子到 Inductor IR 的转换函数
        from torch._inductor import lowering
        lowering_fn = lowering.lowerings.get(target)
        
        if lowering_fn is None:
            raise NotImplementedError(f"No lowering for {target}")
        
        # 调用 lowering 函数，生成 IR
        result = lowering_fn(*args, **kwargs)
        
        # 保存结果
        self.register_buffer(node.name, result)
    
    def compile_to_fn(self):
        """
        编译为可执行函数
        
        流程：
        1. 创建 Scheduler
        2. 执行调度与融合
        3. 生成代码
        4. 编译并加载
        """
        # [1] 创建 Scheduler
        from torch._inductor.scheduler import Scheduler
        self.scheduler = Scheduler(self.buffers)
        
        # [2] 执行调度
        self.scheduler.codegen()
        
        # [3] 生成 Python 模块
        return self.compile_to_module().call
```

#### 1.4 完整示例：逐步追踪

让我们通过一个具体例子，完整追踪从 FX Graph 到机器码的每一步：

```python
# 用户代码
import torch

@torch.compile
def model(x, y):
    z = x + y
    return z.relu()

# 运行
x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')
result = model(x, y)
```

**步骤 1：FX Graph（来自 Dynamo）**

```python
# TorchDynamo 生成的 FX Graph
def forward(x, y):
    add_tensor = torch.ops.aten.add.Tensor(x, y)
    relu_default = torch.ops.aten.relu.default(add_tensor)
    return relu_default

# 图结构
# graph():
#     %x : Tensor(1024)
#     %y : Tensor(1024)
#     %add_tensor : Tensor(1024) = call_function[target=torch.ops.aten.add.Tensor](args=(%x, %y))
#     %relu_default : Tensor(1024) = call_function[target=torch.ops.aten.relu.default](args=(%add_tensor,))
#     return %relu_default
```

**步骤 2：Lowering（FX Graph → IR）**

```python
# torch/_inductor/lowering.py

@register_lowering(torch.ops.aten.add)
def add_tensor(x, y):
    """
    将 aten.add 转换为 Pointwise IR
    """
    def inner_fn(idx):
        return ops.add(
            ops.load(x, idx),
            ops.load(y, idx)
        )
    
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )

@register_lowering(torch.ops.aten.relu)
def relu(x):
    """
    将 aten.relu 转换为 Pointwise IR
    """
    def inner_fn(idx):
        return ops.maximum(
            ops.load(x, idx),
            ops.constant(0.0, x.get_dtype())
        )
    
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )

# 生成的 IR
# buf0 = Pointwise(inner_fn=lambda idx: load(x, idx) + load(y, idx), ranges=[1024])
# buf1 = Pointwise(inner_fn=lambda idx: max(load(buf0, idx), 0.0), ranges=[1024])
```

**步骤 3：Fusion（算子融合）**

```python
# torch/_inductor/scheduler.py

class Scheduler:
    def fusion_pass(self):
        """
        算子融合决策
        
        分析：
        - buf0 (add) 和 buf1 (relu) 都是 Pointwise
        - buf0 只被 buf1 使用
        - 形状相同
        → 可以融合！
        """
        # 融合后的 IR
        # buf_fused = Pointwise(
        #     inner_fn=lambda idx: max(load(x, idx) + load(y, idx), 0.0),
        #     ranges=[1024]
        # )
```

**步骤 4：Code Generation（生成 Triton 代码）**

```python
# torch/_inductor/codegen/triton.py

class TritonKernel:
    def codegen(self):
        """
        生成 Triton 代码
        """
        # 生成的 Triton 代码
        code = """
@triton.jit
def triton_poi_fused_add_relu_0(
    in_ptr0,  # x
    in_ptr1,  # y
    out_ptr0, # output
    xnumel,   # 1024
    XBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    
    # Load
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    tmp1 = tl.load(in_ptr1 + xindex, xmask)
    
    # Compute (融合！)
    tmp2 = tmp0 + tmp1      # add
    tmp3 = tl.maximum(tmp2, 0.0)  # relu
    
    # Store
    tl.store(out_ptr0 + xindex, tmp3, xmask)
"""
        return code
```

**步骤 5：Compilation（编译）**

```python
# Triton 编译流程
Triton Python Code
    ↓
Triton IR (TTIR)
    ↓
LLVM IR
    ↓
PTX (GPU Assembly)
    ↓
CUBIN (GPU Binary)
```

**步骤 6：Execution（执行）**

```python
# 生成的 Wrapper 代码
def call(args):
    x = args[0]  # input x
    y = args[1]  # input y
    
    # 分配输出缓冲区
    buf0 = torch.empty_strided(
        (1024,), (1,),
        device='cuda',
        dtype=torch.float32
    )
    
    # 调用 Triton Kernel
    grid = lambda meta: (triton.cdiv(1024, meta['XBLOCK']),)
    triton_poi_fused_add_relu_0[grid](
        x, y, buf0, 1024,
        XBLOCK=256,
        num_warps=4,
    )
    
    return (buf0,)
```

---

## 第二部分：Lowering 机制 - FX Graph 到 IR 的转换

### 章节 2：Lowering 的核心原理

#### 2.1 什么是 Lowering

**Lowering（降低）是将高层抽象（FX Graph 中的 ATen 算子）转换为低层中间表示（IR）的过程。**

```python
# 高层：FX Graph
torch.ops.aten.add.Tensor(x, y)

# ↓ Lowering

# 低层：IR
Pointwise(
    inner_fn=lambda idx: ops.add(ops.load(x, idx), ops.load(y, idx)),
    ranges=[N]
)
```

**为什么需要 Lowering？**

1. **统一表示**：将数百个 ATen 算子统一为少数几种 IR 类型
2. **便于优化**：IR 更容易分析和优化（融合、内存规划等）
3. **跨平台**：同一个 IR 可以生成不同后端的代码（Triton/C++/CUDA）

#### 2.2 IR 节点类型

```python
# torch/_inductor/ir.py

class IRNode:
    """IR 节点基类"""
    def get_size(self) -> List[Expr]:
        """返回张量形状（符号表达式）"""
        pass
    
    def get_dtype(self) -> torch.dtype:
        """返回数据类型"""
        pass
    
    def get_device(self) -> torch.device:
        """返回设备"""
        pass
    
    def get_reads(self) -> Set[str]:
        """返回读取的缓冲区"""
        pass

# 主要 IR 类型
class Pointwise(IRNode):
    """
    逐点操作
    
    特点：每个输出元素只依赖对应位置的输入元素
    例如：add, mul, relu, sigmoid
    """
    def __init__(self, inner_fn, ranges, **kwargs):
        self.inner_fn = inner_fn  # 计算逻辑
        self.ranges = ranges      # 输出形状
        ...

class Reduction(IRNode):
    """
    归约操作
    
    特点：多个输入元素归约为一个输出元素
    例如：sum, mean, max, min
    """
    def __init__(self, inner_fn, ranges, reduction_ranges, reduction_type, **kwargs):
        self.inner_fn = inner_fn
        self.ranges = ranges              # 输出形状
        self.reduction_ranges = reduction_ranges  # 归约维度
        self.reduction_type = reduction_type  # sum/max/min
        ...

class TensorBox(IRNode):
    """
    张量引用
    
    包装一个 IRNode，提供张量接口
    这是 Inductor 中最常用的包装类型
    """
    def __init__(self, data: IRNode):
        self.data = data
    
    def realize(self):
        """
        实体化：将延迟计算转换为实际的缓冲区
        
        调用时机：
        1. 当张量被多次使用时（避免重复计算）
        2. 当张量作为输出时
        3. 当无法融合时（如算子类型不兼容）
        
        实体化后会创建 ComputedBuffer
        """
        if isinstance(self.data, ComputedBuffer):
            return  # 已经实体化
        
        # 创建新的 ComputedBuffer
        name = V.graph.register_buffer(self.data)
        self.data = ComputedBuffer(
            name=name,
            layout=self.data.get_layout(),
            data=self.data
        )

class ComputedBuffer(IRNode):
    """
    计算缓冲区
    
    表示一个需要计算并存储的中间结果
    """
    def __init__(self, name, layout, data):
        self.name = name
        self.layout = layout
        self.data = data  # IRNode
        ...
```

#### 2.3 Lowering 规则注册

```python
# torch/_inductor/lowering.py

# 全局 lowering 注册表
lowerings = {}

def register_lowering(aten_op):
    """
    装饰器：注册 lowering 函数
    
    用法：
    @register_lowering(torch.ops.aten.add)
    def add_tensor(x, y):
        ...
    """
    def decorator(fn):
        lowerings[aten_op] = fn
        return fn
    return decorator

# 示例：add 的 lowering
@register_lowering(torch.ops.aten.add)
def add_tensor(x, y):
    """
    aten.add → Pointwise IR
    
    Args:
        x, y: TensorBox (包装的 IR 节点)
    
    Returns:
        TensorBox (包装的 Pointwise IR)
    """
    def inner_fn(idx):
        # idx 是符号索引，例如 (i, j) 对于 2D 张量
        x_val = ops.load(x, idx)
        y_val = ops.load(y, idx)
        return ops.add(x_val, y_val)
    
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )
```

#### 2.4 Lowering 执行流程

```python
# torch/_inductor/graph.py

class GraphLowering:
    def call_function(self, node):
        """
        处理 FX Graph 中的 call_function 节点
        
        例如：
        %add = call_function[target=torch.ops.aten.add.Tensor](args=(%x, %y))
        """
        # [1] 获取目标算子
        target = node.target  # torch.ops.aten.add.Tensor
        
        # [2] 获取参数（已经是 IR 节点）
        args = [self.get_buffer(arg) for arg in node.args]
        kwargs = {k: self.get_buffer(v) for k, v in node.kwargs.items()}
        
        # [3] 查找 lowering 函数
        from torch._inductor import lowering
        lowering_fn = lowering.lowerings.get(target)
        
        if lowering_fn is None:
            raise NotImplementedError(f"No lowering for {target}")
        
        # [4] 调用 lowering 函数
        result = lowering_fn(*args, **kwargs)
        
        # [5] 注册结果缓冲区
        self.register_buffer(node.name, result)
        
        return result
```

#### 2.5 深入理解：inner_fn 和 ops.load 的工作机制

**这是 TorchInductor 最精妙的设计 —— 延迟计算 + 符号化表达式树**

##### 问题的本质

```python
# 在 lowering 函数中，我们经常看到这样的代码：
def inner_fn(idx):
    return ops.maximum(
        ops.load(x, idx),
        ops.constant(0.0, x.get_dtype())
    )
```

**关键问题**：
1. `inner_fn` 为什么不立即执行？
2. `ops.load` 返回的是什么？
3. 这些如何转换成 Triton 代码？

##### inner_fn 的本质：计算逻辑的"模板"

```python
# ❌ 错误理解：认为 inner_fn 会立即计算
def inner_fn(idx):
    return ops.maximum(
        ops.load(x, idx),  # 马上加载 x[idx] 的值？NO!
        ops.constant(0.0, x.get_dtype())
    )

# ✅ 正确理解：inner_fn 是一个"计算模板"
# - 定义"如何计算"而不是"立即计算"
# - 类似于 SQL 查询语句（只是描述）
# - 类比：把计算逻辑序列化成数据结构
```

**为什么需要 inner_fn？**

```python
# 原因 1：延迟绑定 - 在不知道具体索引时定义计算逻辑
# 对于 1D 张量：idx 可能是 xindex
# 对于 2D 张量：idx 可能是 (i, j)
# 对于 3D 张量：idx 可能是 (i, j, k)

# 原因 2：便于融合 - 可以内联到其他计算中
# relu 的 inner_fn 可以内联到 add 的结果中

# 原因 3：便于优化 - 可以分析和重写表达式树
# 例如：识别 x * 0 并优化为 0
```

##### ops.load 返回的是"表达式节点"

```python
# torch/_inductor/ops.py (简化版)

class Load:
    """
    Load 操作的符号表示
    
    不会真正加载数据，只是构建一个"加载节点"
    """
    def __init__(self, name, index):
        self.name = name    # 缓冲区名称（如 "buf0", "in_ptr0"）
        self.index = index  # 索引表达式（如 "xindex", "(i, j)"）
    
    def __repr__(self):
        return f"Load({self.name}[{self.index}])"

class Maximum:
    """Maximum 操作的符号表示"""
    def __init__(self, lhs, rhs):
        self.lhs = lhs  # 左操作数（可能是 Load 节点）
        self.rhs = rhs  # 右操作数（可能是 Constant 节点）
    
    def __repr__(self):
        return f"Maximum({self.lhs}, {self.rhs})"

class Constant:
    """常量的符号表示"""
    def __init__(self, value, dtype):
        self.value = value
        self.dtype = dtype
    
    def __repr__(self):
        return f"Constant({self.value})"

# ops 模块提供构建这些节点的工厂函数
def load(buffer, index):
    return Load(buffer.name, index)

def maximum(lhs, rhs):
    return Maximum(lhs, rhs)

def constant(value, dtype):
    return Constant(value, dtype)
```

##### 执行 inner_fn 时构建表达式树

```python
# 假设我们有：
x = TensorBox(name="buf0", size=[1024])

# 定义 inner_fn
def inner_fn(idx):
    return ops.maximum(
        ops.load(x, idx),
        ops.constant(0.0, torch.float32)
    )

# 现在"符号执行" inner_fn
idx = "xindex"  # 这只是一个符号，不是具体的数值！
expr = inner_fn(idx)

# expr 的结果是一个表达式树（AST）：
#       Maximum
#       /     \
#   Load       Constant
#   /  \           |
#  buf0 xindex    0.0

print(expr)
# 输出：Maximum(Load(buf0[xindex]), Constant(0.0))
```

##### 完整示例：从表达式树到融合

```python
# 例子：(x + y).relu()

# [1] add 的 inner_fn
def add_inner_fn(idx):
    return ops.add(
        ops.load(x, idx),  # Load("x", "xindex")
        ops.load(y, idx)   # Load("y", "xindex")
    )

# 符号执行
add_expr = add_inner_fn("xindex")
# 结果表达式树：
#       Add
#       /   \
#   Load     Load
#   /  \     /  \
#  x  xindex y  xindex

# [2] relu 的 inner_fn（引用 add_buffer）
def relu_inner_fn(idx):
    return ops.maximum(
        ops.load(add_buffer, idx),  # 引用上一步的结果
        ops.constant(0.0, torch.float32)
    )

relu_expr = relu_inner_fn("xindex")
# 结果：Maximum(Load(add_buffer[xindex]), Constant(0.0))

# [3] 融合：内联 add_buffer 的计算！
def fused_inner_fn(idx):
    # 直接内联 add 的计算
    add_result = ops.add(
        ops.load(x, idx),
        ops.load(y, idx)
    )
    # 在 add 结果上应用 relu
    return ops.maximum(
        add_result,  # 不再需要 load(add_buffer)
        ops.constant(0.0, torch.float32)
    )

fused_expr = fused_inner_fn("xindex")
# 融合后的表达式树：
#           Maximum
#           /      \
#         Add       Constant
#        /   \          |
#     Load   Load      0.0
#     / \    / \
#    x idx  y idx

# 这样生成的 Triton 代码就只有一个 Kernel，
# 没有中间结果 add_buffer 写回内存！
```

##### 从表达式树到 Triton 代码

```python
# torch/_inductor/codegen/triton.py

class TritonKernel:
    def codegen_pointwise(self, pointwise_node):
        """
        为 Pointwise IR 生成 Triton 代码
        """
        # [1] 符号执行 inner_fn，获取表达式树
        symbolic_idx = "xindex"
        expr_tree = pointwise_node.inner_fn(symbolic_idx)
        
        # expr_tree = Maximum(Load(x[xindex]), Constant(0.0))
        
        # [2] 递归生成代码
        result_var = self.codegen_expr(expr_tree)
        
        # [3] 生成 store
        self.stores.writeline(
            f"tl.store(out_ptr0 + xindex, {result_var}, xmask)"
        )
    
    def codegen_expr(self, expr):
        """
        递归生成表达式的 Triton 代码
        """
        if isinstance(expr, ops.Load):
            # 生成 load 指令
            tmp_var = self.new_tmp_var()  # tmp0
            self.loads.writeline(
                f"{tmp_var} = tl.load({expr.name} + {expr.index}, xmask)"
            )
            return tmp_var
        
        elif isinstance(expr, ops.Maximum):
            # 递归生成左右操作数
            lhs_var = self.codegen_expr(expr.lhs)  # tmp0
            rhs_var = self.codegen_expr(expr.rhs)  # 0.0
            
            # 生成 maximum 指令
            tmp_var = self.new_tmp_var()  # tmp1
            self.compute.writeline(
                f"{tmp_var} = tl.maximum({lhs_var}, {rhs_var})"
            )
            return tmp_var
        
        elif isinstance(expr, ops.Constant):
            # 常量直接返回字符串表示
            return str(expr.value)
        
        else:
            raise NotImplementedError(f"Unknown expr: {type(expr)}")
```

##### 完整转换流程可视化

```
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 1：Lowering（定义 inner_fn）                                  │
└─────────────────────────────────────────────────────────────────┘

@register_lowering(torch.ops.aten.relu)
def relu(x):
    def inner_fn(idx):  ← 定义计算模板，不执行
        return ops.maximum(ops.load(x, idx), 0.0)
    return Pointwise.create(inner_fn=inner_fn, ...)

┌─────────────────────────────────────────────────────────────────┐
│ 阶段 2：Scheduler（符号执行，构建表达式树）                          │
└─────────────────────────────────────────────────────────────────┘

expr = inner_fn("xindex")  ← 符号执行，不求值

生成表达式树：
       Maximum
       /     \
   Load       Constant
   / \            |
  x xindex       0.0

┌─────────────────────────────────────────────────────────────────┐
│ 阶段 3：CodeGen（遍历表达式树，生成 Triton 代码）                    │
└─────────────────────────────────────────────────────────────────┘

遍历 Maximum 节点：
  └─ 遍历 Load 节点 → 生成: tmp0 = tl.load(in_ptr0 + xindex, xmask)
  └─ 遍历 Constant 节点 → 生成: 0.0
  └─ 生成: tmp1 = tl.maximum(tmp0, 0.0)

最终 Triton 代码：
@triton.jit
def kernel(...):
    xindex = pid * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, xmask)  ← 这时才真正加载
    tmp1 = tl.maximum(tmp0, 0.0)             ← 这时才真正计算
    tl.store(out_ptr0 + xindex, tmp1, xmask) ← 这时才真正写入
```

##### 设计优势总结

**1. 延迟绑定 - 灵活处理不同维度**

```python
# 同一个 inner_fn 可以处理不同维度的张量
inner_fn("xindex")        # 1D: idx = xindex
inner_fn(("i", "j"))      # 2D: idx = (i, j)
inner_fn(("i", "j", "k")) # 3D: idx = (i, j, k)
```

**2. 便于融合 - 内联表达式**

```python
# 融合前：两个独立的 Kernel（中间结果写回内存）
buf0 = Pointwise(inner_fn=lambda idx: ops.add(ops.load(x, idx), ops.load(y, idx)))
buf1 = Pointwise(inner_fn=lambda idx: ops.maximum(ops.load(buf0, idx), 0.0))

# 融合后：一个 Kernel（无中间结果）
buf_fused = Pointwise(
    inner_fn=lambda idx: ops.maximum(
        ops.add(ops.load(x, idx), ops.load(y, idx)),  # 内联！
        0.0
    )
)
```

**3. 便于优化 - 代数简化**

```python
# 优化器可以分析表达式树并优化
expr = ops.mul(ops.load(x, idx), ops.constant(0.0))
# 识别 x * 0 = 0
optimized = ops.constant(0.0)

expr = ops.add(ops.load(x, idx), ops.constant(0.0))
# 识别 x + 0 = x
optimized = ops.load(x, idx)
```

**4. 跨后端 - 统一的 IR**

```python
# 同一个表达式树可以生成不同后端的代码

# Triton 后端
triton_codegen.codegen_expr(expr)
# → tl.maximum(tmp0, 0.0)

# C++ 后端
cpp_codegen.codegen_expr(expr)
# → std::max(tmp0, 0.0f)

# CUDA 后端
cuda_codegen.codegen_expr(expr)
# → fmaxf(tmp0, 0.0f)
```

##### 类比理解

```
inner_fn        ≈  SQL 查询语句（描述操作，不执行）
ops.load        ≈  SQL 中的 SELECT（构建查询节点）
表达式树         ≈  SQL 查询计划（AST）
Pointwise IR    ≈  逻辑查询计划
CodeGen         ≈  物理查询计划 + 执行
Triton 代码     ≈  实际的数据库操作
```

**核心要点**：
- `inner_fn` 是延迟执行的计算配方
- `ops.load` 返回符号化的表达式节点（不加载数据）
- 转换过程：定义模板 → 构建表达式树 → 遍历树生成代码
- 这是编译器中经典的 **AST（抽象语法树）** 设计模式

---

### 章节 3：常见算子的 Lowering 实现

#### 3.1 Pointwise 算子

```python
# torch/_inductor/lowering.py

# ========== 一元算子 ==========

@register_lowering(torch.ops.aten.relu)
def relu(x):
    """ReLU: max(x, 0)"""
    def inner_fn(idx):
        x_val = ops.load(x, idx)
        zero = ops.constant(0.0, x.get_dtype())
        return ops.maximum(x_val, zero)
    
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )

@register_lowering(torch.ops.aten.sigmoid)
def sigmoid(x):
    """Sigmoid: 1 / (1 + exp(-x))"""
    def inner_fn(idx):
        x_val = ops.load(x, idx)
        neg_x = ops.neg(x_val)
        exp_neg_x = ops.exp(neg_x)
        one = ops.constant(1.0, x.get_dtype())
        denom = ops.add(one, exp_neg_x)
        return ops.truediv(one, denom)
    
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )

@register_lowering(torch.ops.aten.tanh)
def tanh(x):
    """Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    def inner_fn(idx):
        x_val = ops.load(x, idx)
        # 使用 libdevice.tanh（GPU 库函数）
        return ops.tanh(x_val)
    
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )

# ========== 二元算子 ==========

@register_lowering(torch.ops.aten.mul)
def mul_tensor(x, y):
    """Element-wise multiplication"""
    def inner_fn(idx):
        x_val = ops.load(x, idx)
        y_val = ops.load(y, idx)
        return ops.mul(x_val, y_val)
    
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )

@register_lowering(torch.ops.aten.maximum)
def maximum(x, y):
    """Element-wise maximum"""
    def inner_fn(idx):
        x_val = ops.load(x, idx)
        y_val = ops.load(y, idx)
        return ops.maximum(x_val, y_val)
    
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )
```

#### 3.2 Reduction 算子

```python
@register_lowering(torch.ops.aten.sum)
def sum_dim(x, dim=None, keepdim=False):
    """
    Sum reduction
    
    Args:
        x: 输入张量
        dim: 归约维度（None 表示全部）
        keepdim: 是否保持维度
    """
    if dim is None:
        # 全局归约
        dim = list(range(len(x.get_size())))
    
    if not isinstance(dim, (list, tuple)):
        dim = [dim]
    
    # 标准化维度（处理负数索引）
    ndim = len(x.get_size())
    dim = [d if d >= 0 else d + ndim for d in dim]
    
    # 计算输出形状
    output_size = []
    reduction_ranges = []
    for i, size in enumerate(x.get_size()):
        if i in dim:
            reduction_ranges.append(size)
            if keepdim:
                output_size.append(1)
        else:
            output_size.append(size)
    
    def inner_fn(idx, reduction_idx):
        """
        idx: 输出索引
        reduction_idx: 归约维度的索引
        """
        # 构建完整的输入索引
        full_idx = []
        idx_iter = iter(idx)
        reduction_iter = iter(reduction_idx)
        
        for i in range(ndim):
            if i in dim:
                full_idx.append(next(reduction_iter))
            else:
                full_idx.append(next(idx_iter))
        
        return ops.load(x, tuple(full_idx))
    
    return Reduction.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=output_size,
        reduction_ranges=reduction_ranges,
        reduction_type="sum",
    )

@register_lowering(torch.ops.aten.mean)
def mean_dim(x, dim=None, keepdim=False):
    """Mean = sum / count"""
    # 先计算 sum
    sum_result = sum_dim(x, dim, keepdim)
    
    # 计算元素数量
    if dim is None:
        numel = 1
        for size in x.get_size():
            numel *= size
    else:
        if not isinstance(dim, (list, tuple)):
            dim = [dim]
        numel = 1
        for d in dim:
            numel *= x.get_size()[d]
    
    # sum / numel
    def inner_fn(idx):
        sum_val = ops.load(sum_result, idx)
        count = ops.constant(numel, x.get_dtype())
        return ops.truediv(sum_val, count)
    
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(sum_result.get_size()),
    )
```

#### 3.3 复杂算子：LayerNorm

```python
@register_lowering(torch.ops.aten.native_layer_norm)
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """
    Layer Normalization
    
    公式：
    y = (x - mean) / sqrt(var + eps) * weight + bias
    
    其中 mean 和 var 在 normalized_shape 维度上计算
    """
    # [1] 计算归约维度
    ndim = len(input.get_size())
    norm_ndim = len(normalized_shape)
    reduction_dims = list(range(ndim - norm_ndim, ndim))
    
    # [2] 计算 mean
    mean = mean_dim(input, dim=reduction_dims, keepdim=True)
    
    # [3] 计算 variance
    # var = mean((x - mean)^2)
    def centered_squared(idx):
        x_val = ops.load(input, idx)
        mean_val = ops.load(mean, idx)  # 会自动广播
        centered = ops.sub(x_val, mean_val)
        return ops.mul(centered, centered)
    
    centered_sq = Pointwise.create(
        device=input.get_device(),
        dtype=input.get_dtype(),
        inner_fn=centered_squared,
        ranges=list(input.get_size()),
    )
    
    var = mean_dim(centered_sq, dim=reduction_dims, keepdim=True)
    
    # [4] 归一化
    def normalize(idx):
        x_val = ops.load(input, idx)
        mean_val = ops.load(mean, idx)
        var_val = ops.load(var, idx)
        
        # (x - mean) / sqrt(var + eps)
        centered = ops.sub(x_val, mean_val)
        eps_const = ops.constant(eps, input.get_dtype())
        var_eps = ops.add(var_val, eps_const)
        std = ops.sqrt(var_eps)
        normalized = ops.truediv(centered, std)
        
        # * weight + bias
        if weight is not None:
            weight_val = ops.load(weight, idx[-norm_ndim:])  # 只取最后几维
            normalized = ops.mul(normalized, weight_val)
        
        if bias is not None:
            bias_val = ops.load(bias, idx[-norm_ndim:])
            normalized = ops.add(normalized, bias_val)
        
        return normalized
    
    output = Pointwise.create(
        device=input.get_device(),
        dtype=input.get_dtype(),
        inner_fn=normalize,
        ranges=list(input.get_size()),
    )
    
    return output, mean, var  # 返回 (output, mean, rstd)
```

---

## 第三部分：Scheduler 调度系统

### 章节 4：Scheduler 的核心职责

#### 4.1 Scheduler 是什么

**Scheduler（调度器）负责决定如何执行 IR 节点，包括：**
1. **融合决策**：哪些节点可以融合成一个 Kernel
2. **执行顺序**：节点的执行顺序（拓扑排序）
3. **内存规划**：缓冲区的分配与复用

```python
# torch/_inductor/scheduler.py

class Scheduler:
    """
    调度器
    
    输入：IR 节点列表（来自 GraphLowering）
    输出：调度计划（融合组 + 执行顺序）
    """
    
    def __init__(self, buffers):
        """
        Args:
            buffers: Dict[str, IRNode] - 所有缓冲区
        """
        self.buffers = buffers
        self.nodes = []  # SchedulerNode 列表
        self.fused_nodes = []  # 融合后的节点组
    
    def codegen(self):
        """
        主调度流程
        
        1. 创建 SchedulerNode
        2. 分析依赖关系
        3. 融合决策
        4. 生成代码
        """
        # [1] 为每个缓冲区创建 SchedulerNode
        self.create_scheduler_nodes()
        
        # [2] 分析依赖关系
        self.compute_dependencies()
        
        # [3] 融合决策
        self.fusion_pass()
        
        # [4] 拓扑排序
        self.topological_sort()
        
        # [5] 生成代码
        self.generate_code()
```

#### 4.2 SchedulerNode - 调度节点

```python
class SchedulerNode:
    """
    调度节点
    
    包装一个 IR 节点，添加调度信息
    """
    
    def __init__(self, scheduler, node: IRNode):
        self.scheduler = scheduler
        self.node = node  # IR 节点
        
        # 依赖关系
        self.read_writes = self.node.get_read_writes()
        self.unmet_dependencies = set()  # 未满足的依赖
        self.users = []  # 使用该节点的节点列表
        
        # 融合信息
        self.group = None  # 所属融合组
        self.can_inplace = False  # 是否可以原地操作
    
    def can_fuse(self, other: 'SchedulerNode') -> bool:
        """
        判断是否可以与另一个节点融合
        
        条件：
        1. 都是 Pointwise 或 Reduction
        2. 设备相同
        3. 形状兼容
        4. 无循环依赖
        """
        # [1] 类型检查
        if not self.is_fusable_type() or not other.is_fusable_type():
            return False
        
        # [2] 设备检查
        if self.node.get_device() != other.node.get_device():
            return False
        
        # [3] 形状检查
        if not self.is_compatible_shape(other):
            return False
        
        # [4] 依赖检查
        if self.has_circular_dependency(other):
            return False
        
        return True
    
    def is_fusable_type(self) -> bool:
        """是否是可融合的类型"""
        return isinstance(self.node, (Pointwise, Reduction))
    
    def is_compatible_shape(self, other) -> bool:
        """形状是否兼容"""
        # 简化：要求形状完全相同
        return self.node.get_size() == other.node.get_size()
```

### 章节 5：融合决策算法

#### 5.1 融合的基本原则

```python
class Scheduler:
    def fusion_pass(self):
        """
        融合决策
        
        策略：
        1. 优先融合 Pointwise 操作（最容易）
        2. 尝试融合 Reduction + Pointwise
        3. 避免融合会增加内存使用的情况
        """
        # [1] 构建融合候选
        fusion_candidates = self.find_fusion_candidates()
        
        # [2] 贪心融合
        for producer, consumer in fusion_candidates:
            if self.should_fuse(producer, consumer):
                self.fuse_nodes(producer, consumer)
    
    def should_fuse(self, producer, consumer) -> bool:
        """
        决定是否融合两个节点
        
        考虑因素：
        1. 是否可以融合（类型、形状等）
        2. 融合后的收益（减少内存访问）
        3. 融合后的成本（增加寄存器压力）
        """
        # [1] 基本检查
        if not producer.can_fuse(consumer):
            return False
        
        # [2] 检查 producer 是否只被 consumer 使用
        if len(producer.users) != 1:
            # producer 有多个使用者，融合会导致重复计算
            return False
        
        # [3] 估算收益
        benefit = self.estimate_fusion_benefit(producer, consumer)
        cost = self.estimate_fusion_cost(producer, consumer)
        
        return benefit > cost
    
    def estimate_fusion_benefit(self, producer, consumer) -> float:
        """
        估算融合收益
        
        主要收益：减少内存访问
        """
        # producer 的输出不需要写回内存
        producer_size = producer.node.get_numel()
        elem_size = producer.node.get_dtype().itemsize
        
        # 节省的内存访问（字节）
        saved_bytes = producer_size * elem_size * 2  # 1次写 + 1次读
        
        return saved_bytes
    
    def estimate_fusion_cost(self, producer, consumer) -> float:
        """
        估算融合成本
        
        主要成本：增加寄存器使用
        """
        # 简化：假设每个操作需要固定数量的寄存器
        producer_regs = self.estimate_register_usage(producer)
        consumer_regs = self.estimate_register_usage(consumer)
        
        # 融合后的寄存器使用
        fused_regs = producer_regs + consumer_regs
        
        # 如果超过阈值，成本很高
        MAX_REGS = 64
        if fused_regs > MAX_REGS:
            return float('inf')
        
        return 0.0  # 否则成本可忽略
```

#### 5.2 融合实现

```python
class Scheduler:
    def fuse_nodes(self, producer, consumer):
        """
        融合两个节点
        
        策略：
        1. 内联 producer 的计算到 consumer
        2. 更新依赖关系
        3. 移除 producer
        """
        # [1] 创建融合后的 inner_fn
        def fused_inner_fn(idx):
            # 内联 producer 的计算
            producer_result = producer.node.inner_fn(idx)
            
            # 在 consumer 的 inner_fn 中，
            # 将对 producer 的 load 替换为直接使用 producer_result
            return consumer.node.inner_fn_with_inline(
                producer.name,
                producer_result,
                idx
            )
        
        # [2] 创建融合节点
        fused_node = Pointwise.create(
            device=consumer.node.get_device(),
            dtype=consumer.node.get_dtype(),
            inner_fn=fused_inner_fn,
            ranges=consumer.node.get_size(),
        )
        
        # [3] 更新图
        # 将 consumer 替换为 fused_node
        self.replace_node(consumer, fused_node)
        
        # 移除 producer
        self.remove_node(producer)
        
        # [4] 更新依赖关系
        # producer 的依赖 → fused_node 的依赖
        for dep in producer.unmet_dependencies:
            fused_node.unmet_dependencies.add(dep)
```

### 章节 6：内存规划

#### 6.1 缓冲区生命周期分析

```python
class Scheduler:
    def compute_buffer_lifetimes(self):
        """
        计算每个缓冲区的生命周期
        
        生命周期 = [first_use, last_use]
        """
        lifetimes = {}
        
        # 拓扑排序后的节点
        for i, node in enumerate(self.ordered_nodes):
            # 该节点读取的缓冲区
            for buf_name in node.read_writes.reads:
                if buf_name not in lifetimes:
                    lifetimes[buf_name] = [i, i]
                else:
                    lifetimes[buf_name][1] = i  # 更新 last_use
            
            # 该节点写入的缓冲区
            for buf_name in node.read_writes.writes:
                if buf_name not in lifetimes:
                    lifetimes[buf_name] = [i, i]
                # first_use 已经设置
        
        return lifetimes
    
    def allocate_buffers(self):
        """
        分配缓冲区
        
        策略：
        1. 复用生命周期不重叠的缓冲区
        2. 对齐内存以提高访问效率
        """
        lifetimes = self.compute_buffer_lifetimes()
        
        # 按大小排序（大的优先分配）
        buffers_by_size = sorted(
            self.buffers.items(),
            key=lambda x: x[1].get_numel(),
            reverse=True
        )
        
        # 内存池
        memory_pool = []
        allocations = {}
        
        for buf_name, buf_node in buffers_by_size:
            buf_lifetime = lifetimes[buf_name]
            buf_size = buf_node.get_numel() * buf_node.get_dtype().itemsize
            
            # 尝试从内存池中复用
            reused = False
            for pool_entry in memory_pool:
                pool_buf, pool_lifetime, pool_offset = pool_entry
                
                # 检查生命周期是否不重叠
                if (buf_lifetime[0] > pool_lifetime[1] or
                    buf_lifetime[1] < pool_lifetime[0]):
                    # 可以复用
                    allocations[buf_name] = (pool_buf, pool_offset)
                    # 更新生命周期
                    pool_entry[1] = [
                        min(pool_lifetime[0], buf_lifetime[0]),
                        max(pool_lifetime[1], buf_lifetime[1])
                    ]
                    reused = True
                    break
            
            if not reused:
                # 分配新缓冲区
                new_buf = f"buf_pool_{len(memory_pool)}"
                allocations[buf_name] = (new_buf, 0)
                memory_pool.append([new_buf, buf_lifetime, buf_size])
        
        return allocations
```

---

## 第四部分：CodeGen 代码生成

### 章节 7：Triton 代码生成原理

#### 7.1 IndentedBuffer - 代码缓冲区工具

```python
# torch/_inductor/codegen/common.py

class IndentedBuffer:
    """
    缩进缓冲区 - 用于生成格式化的代码
    
    特点：
    1. 自动管理缩进
    2. 支持嵌套
    3. 可以拼接其他 IndentedBuffer
    """
    
    def __init__(self, indent_level=0):
        self._lines = []
        self._indent_level = indent_level
        self._indent_str = "    "  # 4 空格
    
    def writeline(self, line):
        """写入一行（自动添加缩进）"""
        if line:
            self._lines.append(self._indent_str * self._indent_level + line)
        else:
            self._lines.append("")
    
    def indent(self):
        """增加缩进级别"""
        self._indent_level += 1
    
    def dedent(self):
        """减少缩进级别"""
        self._indent_level -= 1
    
    def splice(self, other_buffer):
        """拼接另一个缓冲区"""
        self._lines.extend(other_buffer._lines)
    
    def getvalue(self):
        """获取完整代码字符串"""
        return "\n".join(self._lines)

# 使用示例
code = IndentedBuffer()
code.writeline("def my_function():")
code.indent()
code.writeline("x = 1")
code.writeline("return x")
code.dedent()

print(code.getvalue())
# 输出：
# def my_function():
#     x = 1
#     return x
```

#### 7.2 Triton 简介

**Triton 是一种 Python DSL（领域特定语言），用于编写 GPU Kernel。**

**优势**：
- Python 语法，易于学习
- 自动内存管理
- 自动并行化
- 性能接近手写 CUDA（90-95%）

**示例**：

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,      # 输入指针
    y_ptr,      # 输入指针
    out_ptr,    # 输出指针
    n_elements, # 元素数量
    BLOCK_SIZE: tl.constexpr,  # 编译时常量
):
    # 获取当前线程块的 ID
    pid = tl.program_id(0)
    
    # 计算该线程块处理的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 边界检查
    mask = offsets < n_elements
    
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 计算
    output = x + y
    
    # 存储结果
    tl.store(out_ptr + offsets, output, mask=mask)

# 调用
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
```

#### 7.3 TritonKernel - 代码生成器

```python
# torch/_inductor/codegen/triton.py

class TritonKernel:
    """
    Triton Kernel 代码生成器
    
    职责：
    1. 将 IR 节点转换为 Triton 代码
    2. 管理参数、索引、加载、计算、存储
    3. 选择最优的 Kernel 参数（BLOCK_SIZE, num_warps 等）
    """
    
    def __init__(self, *groups):
        """
        Args:
            groups: 融合后的节点组
        """
        self.groups = groups
        
        # 代码缓冲区
        self.args = IndentedBuffer()      # 参数列表
        self.indexing = IndentedBuffer()  # 索引计算
        self.loads = IndentedBuffer()     # 加载操作
        self.compute = IndentedBuffer()   # 计算逻辑
        self.stores = IndentedBuffer()    # 存储操作
        
        # 临时变量计数器
        self.tmp_counter = 0
        
        # 参数配置
        self.block_size = None
        self.num_warps = None
    
    def codegen(self):
        """
        生成完整的 Triton Kernel 代码
        
        流程：
        1. 生成参数列表
        2. 生成索引计算
        3. 生成加载/计算/存储
        4. 组装完整代码
        """
        # [1] 生成参数
        self.codegen_args()
        
        # [2] 生成索引
        self.codegen_indexing()
        
        # [3] 生成计算
        self.codegen_body()
        
        # [4] 选择参数
        self.select_kernel_config()
        
        # [5] 组装代码
        return self.assemble()
    
    def codegen_args(self):
        """
        生成参数列表
        
        包括：
        - 输入缓冲区指针
        - 输出缓冲区指针
        - 元素数量
        - 编译时常量（BLOCK_SIZE 等）
        """
        # 输入缓冲区
        for i, inp in enumerate(self.get_inputs()):
            self.args.writeline(f"in_ptr{i},")
        
        # 输出缓冲区
        for i, out in enumerate(self.get_outputs()):
            self.args.writeline(f"out_ptr{i},")
        
        # 元素数量
        self.args.writeline("xnumel,")
        
        # 编译时常量
        self.args.writeline("XBLOCK: tl.constexpr,")
    
    def codegen_indexing(self):
        """
        生成索引计算代码
        
        对于 1D 张量：
        xindex = pid * XBLOCK + tl.arange(0, XBLOCK)
        xmask = xindex < xnumel
        """
        self.indexing.writeline("# 计算索引")
        self.indexing.writeline("pid = tl.program_id(0)")
        self.indexing.writeline("xoffset = pid * XBLOCK")
        self.indexing.writeline("xindex = xoffset + tl.arange(0, XBLOCK)")
        self.indexing.writeline("xmask = xindex < xnumel")
    
    def codegen_body(self):
        """
        生成计算主体
        
        遍历融合组中的所有节点，生成对应的代码
        """
        for group in self.groups:
            for node in group.nodes:
                self.codegen_node(node)
    
    def codegen_node(self, node):
        """
        为单个 IR 节点生成代码
        """
        if isinstance(node, Pointwise):
            self.codegen_pointwise(node)
        elif isinstance(node, Reduction):
            self.codegen_reduction(node)
        else:
            raise NotImplementedError(f"Unknown node type: {type(node)}")
    
    def codegen_pointwise(self, node):
        """
        生成 Pointwise 节点的代码
        
        策略：
        1. 符号执行 inner_fn
        2. 递归生成表达式树
        3. 生成 load/compute/store
        """
        # 符号执行
        symbolic_idx = sympy.Symbol('xindex')
        expr = node.inner_fn(symbolic_idx)
        
        # 生成表达式代码
        result_var = self.codegen_expr(expr)
        
        # 生成 store
        self.stores.writeline(
            f"tl.store(out_ptr0 + xindex, {result_var}, xmask)"
        )
    
    def codegen_expr(self, expr):
        """
        递归生成表达式的 Triton 代码
        
        Args:
            expr: 符号表达式（ops.Load, ops.Add, etc.）
        
        Returns:
            tmp_var: 临时变量名
        """
        if isinstance(expr, ops.Load):
            # 加载操作
            buffer_name = expr.name
            index = 'xindex'
            
            tmp_var = self.new_tmp_var()
            self.loads.writeline(
                f"{tmp_var} = tl.load({buffer_name} + {index}, xmask)"
            )
            return tmp_var
        
        elif isinstance(expr, ops.Add):
            # 加法操作
            lhs_var = self.codegen_expr(expr.lhs)
            rhs_var = self.codegen_expr(expr.rhs)
            
            tmp_var = self.new_tmp_var()
            self.compute.writeline(f"{tmp_var} = {lhs_var} + {rhs_var}")
            return tmp_var
        
        elif isinstance(expr, ops.Maximum):
            # Maximum 操作
            lhs_var = self.codegen_expr(expr.lhs)
            rhs_var = self.codegen_expr(expr.rhs)
            
            tmp_var = self.new_tmp_var()
            self.compute.writeline(f"{tmp_var} = tl.maximum({lhs_var}, {rhs_var})")
            return tmp_var
        
        elif isinstance(expr, ops.Constant):
            # 常量
            return str(expr.value)
        
        else:
            raise NotImplementedError(f"Unknown expr: {type(expr)}")
    
    def new_tmp_var(self):
        """分配新的临时变量"""
        var = f"tmp{self.tmp_counter}"
        self.tmp_counter += 1
        return var
    
    def assemble(self):
        """
        组装完整的 Kernel 代码
        """
        code = IndentedBuffer()
        
        # 函数签名
        code.writeline("@triton.jit")
        code.writeline(f"def {self.kernel_name}(")
        code.indent()
        code.splice(self.args)
        code.dedent()
        code.writeline("):")
        
        # 函数体
        code.indent()
        code.splice(self.indexing)
        code.writeline("")
        code.splice(self.loads)
        code.writeline("")
        code.splice(self.compute)
        code.writeline("")
        code.splice(self.stores)
        code.dedent()
        
        return code.getvalue()
```

### 章节 8：完整示例 - 从 IR 到 Triton 代码

让我们通过一个完整的例子，看看如何从 IR 生成 Triton 代码：

```python
# 输入 IR（融合后）
fused_ir = Pointwise(
    inner_fn=lambda idx: ops.maximum(
        ops.add(
            ops.load('in_ptr0', idx),  # x
            ops.load('in_ptr1', idx)   # y
        ),
        ops.constant(0.0, torch.float32)
    ),
    ranges=[1024],
    name='buf_fused'
)

# 代码生成过程
kernel = TritonKernel(fused_ir)

# [1] 生成参数
# in_ptr0,
# in_ptr1,
# out_ptr0,
# xnumel,
# XBLOCK: tl.constexpr,

# [2] 生成索引
# pid = tl.program_id(0)
# xoffset = pid * XBLOCK
# xindex = xoffset + tl.arange(0, XBLOCK)
# xmask = xindex < xnumel

# [3] 生成计算
# 符号执行 inner_fn(xindex)
expr = fused_ir.inner_fn('xindex')
# expr = ops.maximum(
#     ops.add(
#         ops.load('in_ptr0', 'xindex'),
#         ops.load('in_ptr1', 'xindex')
#     ),
#     ops.constant(0.0, torch.float32)
# )

# 递归生成代码
# tmp0 = tl.load(in_ptr0 + xindex, xmask)  # load x
# tmp1 = tl.load(in_ptr1 + xindex, xmask)  # load y
# tmp2 = tmp0 + tmp1                       # add
# tmp3 = tl.maximum(tmp2, 0.0)             # relu

# [4] 生成 store
# tl.store(out_ptr0 + xindex, tmp3, xmask)

# 最终生成的 Triton 代码
generated_code = """
@triton.jit
def triton_poi_fused_add_relu_0(
    in_ptr0,
    in_ptr1,
    out_ptr0,
    xnumel,
    XBLOCK: tl.constexpr,
):
    # 计算索引
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    
    # Load
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    tmp1 = tl.load(in_ptr1 + xindex, xmask)
    
    # Compute
    tmp2 = tmp0 + tmp1
    tmp3 = tl.maximum(tmp2, 0.0)
    
    # Store
    tl.store(out_ptr0 + xindex, tmp3, xmask)
"""
```

### 章节 9：Kernel 参数选择（启发式）

#### 9.1 BLOCK_SIZE 选择

```python
class TritonKernel:
    def select_block_size(self, numel, dtype):
        """
        选择最优的 BLOCK_SIZE
        
        考虑因素：
        1. 元素数量
        2. 数据类型大小
        3. 内存对齐
        4. 并行度
        """
        elem_size = dtype.itemsize
        
        # 候选值（2 的幂次）
        candidates = [128, 256, 512, 1024]
        
        for block_size in reversed(candidates):
            # 计算 grid size
            grid_size = triton.cdiv(numel, block_size)
            
            # 条件 1: grid 不能太小（至少要有足够的并行度）
            min_grid = self.get_device_sm_count() * 4
            if grid_size < min_grid:
                continue
            
            # 条件 2: 内存对齐（128 字节 = 缓存行大小）
            if block_size * elem_size % 128 == 0:
                return block_size
        
        return 256  # 默认值
    
    def select_num_warps(self, block_size):
        """
        选择 num_warps
        
        每个 warp = 32 threads
        num_warps = ceil(block_size / 32)，向上取整到 2 的幂次
        """
        min_warps = (block_size + 31) // 32
        
        import math
        num_warps = 2 ** math.ceil(math.log2(min_warps))
        
        # 限制范围
        return min(max(num_warps, 1), 32)
```

### 章节 10：Wrapper 代码生成

```python
# torch/_inductor/codegen/wrapper.py

class WrapperCodegen:
    """
    生成 Python Wrapper 代码
    
    职责：
    1. 解包输入参数
    2. 分配输出缓冲区
    3. 调用 Triton Kernel
    4. 返回结果
    """
    
    def generate(self):
        """生成完整的 Python 模块"""
        code = IndentedBuffer()
        
        # [1] 导入
        code.writeline("import torch")
        code.writeline("import triton")
        code.writeline("import triton.language as tl")
        code.writeline("")
        
        # [2] Kernel 定义
        for kernel in self.kernels:
            code.splice(kernel.code)
            code.writeline("")
        
        # [3] 调用函数
        code.writeline("def call(args):")
        code.indent()
        
        # [3.1] 解包参数
        for i, inp in enumerate(self.graph_inputs):
            code.writeline(f"primals_{i+1} = args[{i}]")
        code.writeline("")
        
        # [3.2] 分配输出缓冲区
        for i, buf in enumerate(self.buffers):
            code.writeline(f"buf{i} = torch.empty_strided(")
            code.indent()
            code.writeline(f"{buf.size},")
            code.writeline(f"{buf.stride},")
            code.writeline(f"device='{buf.device}',")
            code.writeline(f"dtype={buf.dtype}")
            code.dedent()
            code.writeline(")")
        code.writeline("")
        
        # [3.3] 调用 Kernel
        for kernel_call in self.kernel_calls:
            code.splice(kernel_call)
        code.writeline("")
        
        # [3.4] 返回结果
        output_names = [f"buf{i}" for i in self.output_indices]
        code.writeline(f"return ({', '.join(output_names)},)")
        code.dedent()
        
        return code.getvalue()
```

---

## 第五部分：优化技术

### 章节 11：内存优化

#### 11.1 缓冲区复用

```python
class Scheduler:
    def optimize_memory(self):
        """
        内存优化
        
        策略：
        1. 分析缓冲区生命周期
        2. 复用生命周期不重叠的缓冲区
        3. 原地操作（inplace）
        """
        # [1] 计算生命周期
        lifetimes = self.compute_buffer_lifetimes()
        
        # [2] 构建干扰图
        interference_graph = self.build_interference_graph(lifetimes)
        
        # [3] 图着色（寄存器分配算法）
        allocation = self.graph_coloring(interference_graph)
        
        return allocation
    
    def build_interference_graph(self, lifetimes):
        """
        构建干扰图
        
        如果两个缓冲区的生命周期重叠，则它们"干扰"
        """
        graph = {}
        buffers = list(lifetimes.keys())
        
        for i, buf1 in enumerate(buffers):
            graph[buf1] = set()
            for buf2 in buffers[i+1:]:
                # 检查生命周期是否重叠
                if self.lifetimes_overlap(lifetimes[buf1], lifetimes[buf2]):
                    graph[buf1].add(buf2)
                    if buf2 not in graph:
                        graph[buf2] = set()
                    graph[buf2].add(buf1)
        
        return graph
    
    def graph_coloring(self, graph):
        """
        图着色算法
        
        为每个节点分配一个"颜色"（内存池 ID）
        相邻节点不能有相同颜色
        """
        allocation = {}
        colors_used = {}
        
        # 按度数排序（度数高的优先）
        nodes = sorted(graph.keys(), key=lambda n: len(graph[n]), reverse=True)
        
        for node in nodes:
            # 找到邻居使用的颜色
            neighbor_colors = {
                allocation[neighbor]
                for neighbor in graph[node]
                if neighbor in allocation
            }
            
            # 选择最小的未使用颜色
            color = 0
            while color in neighbor_colors:
                color += 1
            
            allocation[node] = color
            colors_used[color] = colors_used.get(color, 0) + 1
        
        return allocation
```

#### 11.2 内存布局优化

```python
class LayoutOptimizer:
    """
    内存布局优化
    
    目标：
    1. 减少 transpose 操作
    2. 提高内存访问连续性
    3. 利用 Tensor Core（如果可用）
    """
    
    def optimize_layout(self, graph):
        """
        优化整个图的内存布局
        """
        # [1] 分析每个节点的首选布局
        preferred_layouts = self.analyze_preferred_layouts(graph)
        
        # [2] 传播布局约束
        final_layouts = self.propagate_layouts(graph, preferred_layouts)
        
        # [3] 插入必要的 transpose
        self.insert_transposes(graph, final_layouts)
        
        return final_layouts
    
    def analyze_preferred_layouts(self, graph):
        """
        分析每个节点的首选布局
        
        例如：
        - Matmul 倾向于 (M, K) @ (K, N) → (M, N)
        - Conv2d 倾向于 NHWC（在某些硬件上）
        """
        layouts = {}
        
        for node in graph.nodes:
            if node.op == 'call_function':
                if node.target == torch.ops.aten.mm:
                    # Matmul: 倾向于行主序
                    layouts[node] = 'row_major'
                elif node.target == torch.ops.aten.conv2d:
                    # Conv: 根据硬件选择
                    if self.has_tensor_cores():
                        layouts[node] = 'nhwc'
                    else:
                        layouts[node] = 'nchw'
                else:
                    layouts[node] = 'any'
        
        return layouts
```

### 章节 12：循环优化

#### 12.1 Loop Tiling

```python
class LoopOptimizer:
    """
    循环优化
    
    技术：
    1. Tiling（分块）
    2. Unrolling（展开）
    3. Vectorization（向量化）
    """
    
    def apply_tiling(self, loop, tile_size):
        """
        应用循环分块
        
        原始循环：
        for i in range(N):
            for j in range(M):
                C[i][j] = A[i][j] + B[i][j]
        
        分块后：
        for ii in range(0, N, TILE_I):
            for jj in range(0, M, TILE_J):
                for i in range(ii, min(ii+TILE_I, N)):
                    for j in range(jj, min(jj+TILE_J, M)):
                        C[i][j] = A[i][j] + B[i][j]
        """
        # 在 Triton 中，这通过 BLOCK_SIZE 自动实现
        pass
```

### 章节 13：算子特化

#### 13.1 Matmul 优化

```python
# torch/_inductor/kernel/mm.py

class MatmulKernel:
    """
    矩阵乘法的特化实现
    
    策略：
    1. 调用 cuBLAS/cuDNN（最优）
    2. 如果需要融合，生成 Triton Kernel
    """
    
    @staticmethod
    def should_use_extern_kernel(A, B):
        """
        决定是否使用外部库（cuBLAS）
        
        条件：
        1. 矩阵足够大（> 128x128）
        2. 没有需要融合的后续操作
        """
        M, K = A.get_size()
        K2, N = B.get_size()
        
        # 小矩阵：使用 Triton
        if M * N * K < 128 * 128 * 128:
            return False
        
        # 大矩阵：使用 cuBLAS
        return True
    
    @staticmethod
    def generate_triton_matmul(A, B):
        """
        生成 Triton Matmul Kernel
        
        使用 Triton 的 tl.dot 指令
        """
        code = """
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算该 block 处理的范围
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 循环遍历 K 维度
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        # 加载 A 和 B 的块
        A_block = tl.load(A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak)
        B_block = tl.load(B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        
        # 矩阵乘法
        acc += tl.dot(A_block, B_block)
    
    # 存储结果
    C = acc.to(tl.float16)
    tl.store(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, C)
"""
        return code
```

---

## 第六部分：编译与执行

### 章节 14：Triton 编译流程

#### 14.1 编译管道

```python
# Triton 编译流程
Triton Python Code
    ↓ [1] 解析
Triton AST
    ↓ [2] 类型推断
Typed Triton AST
    ↓ [3] 转换为 Triton IR (TTIR)
TTIR
    ↓ [4] 优化 TTIR
Optimized TTIR
    ↓ [5] 转换为 LLVM IR
LLVM IR
    ↓ [6] LLVM 优化
Optimized LLVM IR
    ↓ [7] 转换为 PTX
PTX (GPU Assembly)
    ↓ [8] PTX 编译为 CUBIN
CUBIN (GPU Binary)
```

#### 14.2 缓存机制

```python
# torch/_inductor/codecache.py

class CodeCache:
    """
    代码缓存
    
    避免重复编译相同的 Kernel
    """
    
    def __init__(self):
        self.cache_dir = Path("/tmp/torchinductor_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, code, config):
        """
        计算缓存 Key
        
        基于：
        1. Kernel 代码
        2. 编译配置（BLOCK_SIZE, num_warps 等）
        3. 设备信息
        """
        import hashlib
        
        key_data = {
            'code': code,
            'config': config,
            'device': torch.cuda.get_device_name(),
            'triton_version': triton.__version__,
        }
        
        key_str = str(key_data)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def load(self, cache_key):
        """从缓存加载编译结果"""
        cache_file = self.cache_dir / f"{cache_key}.cubin"
        
        if cache_file.exists():
            return cache_file.read_bytes()
        
        return None
    
    def save(self, cache_key, cubin):
        """保存编译结果到缓存"""
        cache_file = self.cache_dir / f"{cache_key}.cubin"
        cache_file.write_bytes(cubin)
```

### 章节 15：动态加载与执行

```python
# torch/_inductor/runtime/runtime_utils.py

class CompiledModule:
    """
    编译后的模块
    
    包含：
    1. 编译后的 Kernel
    2. Wrapper 函数
    3. 元数据
    """
    
    def __init__(self, module_path):
        """
        Args:
            module_path: 生成的 Python 模块路径
        """
        # 动态导入模块
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "compiled_module",
            module_path
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        
        # 获取调用函数
        self.call = self.module.call
    
    def __call__(self, *args, **kwargs):
        """执行编译后的代码"""
        return self.call(args)
```


---

## 第七部分：高级主题

### 章节 17：AutoTuning 机制

```python
# Triton AutoTuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def autotuned_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Kernel 代码
    pass

# 第一次调用会测试所有配置，选择最快的
# 后续调用直接使用缓存的最优配置
```

### 章节 18：自定义后端

```python
# 为新硬件添加自定义后端
class MyDeviceBackend:
    """自定义设备后端"""
    
    @staticmethod
    def compile_fx(gm, example_inputs):
        """编译 FX Graph"""
        # 自定义编译逻辑
        pass
    
    @staticmethod
    def generate_code(ir_nodes):
        """生成设备特定的代码"""
        # 例如：生成 NPU 指令
        pass

# 注册后端
torch._dynamo.list_backends()['mydevice'] = MyDeviceBackend.compile_fx
```

### 章节 19：调试技巧

```python
# [1] 查看生成的代码
import os
os.environ['TORCH_LOGS'] = 'output_code'

@torch.compile
def model(x):
    return x.relu()

x = torch.randn(100, device='cuda')
y = model(x)  # 会打印生成的 Triton 代码

# [2] 保存生成的代码到文件
torch._inductor.config.debug = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.log_dir = "./inductor_logs"

# [3] 禁用特定优化
torch._inductor.config.triton.autotune = False
torch._inductor.config.fx_graph_cache = False

# [4] 强制重新编译
torch._dynamo.reset()
```

---

## 第八部分：实战案例

### 章节 20：案例 1 - GELU 激活函数优化

```python
import torch
import math

@torch.compile
def gelu_approximate(x):
    """
    GELU 激活函数的近似实现
    
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    tanh_inner = torch.tanh(inner)
    result = 0.5 * x * (1.0 + tanh_inner)
    
    return result

# TorchInductor 会将所有操作融合成一个 Kernel
# 性能提升：~9x（相比 Eager 模式）

x = torch.randn(1000000, device='cuda')
y = gelu_approximate(x)
```

### 章节 21：案例 2 - LayerNorm 优化

```python
@torch.compile
def layer_norm_manual(x, weight, bias, eps=1e-5):
    """
    手动实现的 LayerNorm
    
    TorchInductor 会优化为 1-2 个融合 Kernel
    """
    # 计算 mean 和 variance
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    
    # 归一化
    x_normalized = (x - mean) / torch.sqrt(var + eps)
    
    # Scale and shift
    return x_normalized * weight + bias

# 性能对比
x = torch.randn(128, 512, device='cuda')
weight = torch.randn(512, device='cuda')
bias = torch.randn(512, device='cuda')

# Eager: ~0.5ms
# Compiled: ~0.1ms (5x 加速)
```

### 章节 22：案例 3 - 自定义融合算子

```python
@torch.compile
def fused_linear_gelu(x, weight, bias):
    """
    融合 Linear + GELU
    
    TorchInductor 会：
    1. Linear 使用 cuBLAS
    2. GELU 生成融合 Triton Kernel
    """
    # Linear
    out = torch.nn.functional.linear(x, weight, bias)
    
    # GELU
    return torch.nn.functional.gelu(out, approximate='tanh')

# 性能提升：
# - Linear: 使用 cuBLAS（最优）
# - GELU: 融合 Kernel（避免中间结果写回）
# 总体加速：~2-3x
```

---

## 总结

### 核心要点

1. **TorchInductor 的职责**
   - 将 FX Graph 编译为高效的机器码
   - 通过 Lowering、Fusion、CodeGen 实现端到端优化

2. **关键技术**
   - **Lowering**: FX Graph → IR（统一表示）
   - **Fusion**: 算子融合（减少内存访问）
   - **CodeGen**: IR → Triton/C++（高效代码生成）
   - **Compilation**: Triton → PTX → CUBIN（机器码）

3. **性能优化**
   - 算子融合：减少 Kernel 启动和内存访问
   - 内存规划：缓冲区复用、布局优化
   - 循环优化：Tiling、向量化
   - AutoTuning：自动选择最优参数

4. **与 TorchDynamo 的关系**
   ```
   TorchDynamo: 捕获图（字节码 → FX Graph）
   TorchInductor: 优化执行（FX Graph → 机器码）
   ```

### 技术栈总览

```
用户 Python 代码
    ↓ [TorchDynamo]
FX Graph
    ↓ [TorchInductor Lowering]
IR (Pointwise, Reduction, etc.)
    ↓ [Scheduler]
Fused IR Groups
    ↓ [CodeGen]
Triton/C++ Code
    ↓ [Compilation]
PTX/CUBIN
    ↓ [Execution]
高效计算结果 
```

### 推荐阅读顺序

如果你想深入阅读 TorchInductor 源码，建议按以下顺序：

1. **`torch/_inductor/compile_fx.py`** - 理解编译入口 (200 行)
2. **`torch/_inductor/graph.py`** - 理解 GraphLowering (1000+ 行，核心)
3. **`torch/_inductor/lowering.py`** - 理解 Lowering 规则 (3000+ 行)
4. **`torch/_inductor/ir.py`** - 理解 IR 节点定义 (2000+ 行)
5. **`torch/_inductor/scheduler.py`** - 理解调度与融合 (2000+ 行，核心)
6. **`torch/_inductor/codegen/triton.py`** - 理解 Triton 代码生成 (3000+ 行，核心)

**总计**：~15,000+ 行核心代码

---

*本文档基于 PyTorch 2.0+ 源码编写，部分实现细节可能因版本而异*
