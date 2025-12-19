# PyTorch Dynamo 和 Inductor 编译流程

> 本教程采用递进式讲解，从用户代码开始，一步步追踪整个编译流程。
> 本教程使用的torch==2.7.0

---

## 第一步：用户调用 torch.compile()

### 用户代码

```python
import torch

model = MyModel()
compiled_model = torch.compile(model)  # 这一步发生了什么？
output = compiled_model(input_tensor)  # 第一次调用触发编译
```

### torch.compile 做了什么

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/__init__.py`

```python
def compile(model, **kwargs):
    """
    将模型包装为 OptimizedModule
    """
    # 创建编译上下文
    backend = kwargs.get('backend', 'inductor')
    
    # 调用 _dynamo.optimize
    return torch._dynamo.optimize(backend)(model)
```

**调用路径**:
```
torch.compile(model)
  -> torch._dynamo.optimize(backend)
  -> 返回一个装饰器函数
  -> 装饰器包装 model
```

---

## 第二步：创建 OptimizedModule

### optimize 函数的实现

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/eval_frame.py:159-200`

```python
def optimize(backend, **kwargs):
    """
    返回一个装饰器，用于包装模型
    """
    def decorator(model):
        # 创建 OptimizedModule
        return OptimizedModule(
            model,
            dynamo_ctx=OptimizeContext(
                compiler_fn=lookup_backend(backend),  # 获取编译器函数
                **kwargs
            )
        )
    
    return decorator
```

### OptimizedModule 的初始化

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/eval_frame.py:215-267`

```python
class OptimizedModule(torch.nn.Module):
    def __init__(self, mod, dynamo_ctx):
        super().__init__()
        self._orig_mod = mod           # 保存原始模型
        self.dynamo_ctx = dynamo_ctx   # 编译上下文
        self._initialize()             # 初始化
    
    def _initialize(self):
        """
        包装原始模型的 forward 方法
        """
        if isinstance(self.dynamo_ctx, DisableContext):
            # 情况1: 禁用编译
            self._orig_mod.forward = self.dynamo_ctx(self._orig_mod.forward)
        
        elif self._orig_mod.__module__.startswith('torch.nn.'):
            # 情况2: torch.nn 内置模块
            # 使用 wrap_inline 创建新栈帧，避免被 skipfiles 跳过
            self._orig_mod.forward = wrap_inline(
                self.dynamo_ctx(self._orig_mod.forward)
            )
        
        else:
            # 情况3: 用户自定义模块
            # 包装 __call__ 方法
            self._orig_mod.__call__ = self.dynamo_ctx(self._orig_mod.__call__)
```

**关键点**:
- `dynamo_ctx(func)` 会返回一个包装后的函数
- 这个包装函数会拦截执行，触发编译

**调用路径总结**:
```
torch.compile(model)
  -> optimize(backend)(model)
    -> OptimizedModule(model, OptimizeContext(...))
      -> _initialize()
        -> 包装 model.forward 或 model.__call__
```

---

## 第三步：第一次调用触发编译

### 调用 compiled_model

```python
output = compiled_model(input_tensor)  # 触发 __call__
```

### Module.__call__ 的执行

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py:1735-1750`

```python
class Module:
    # __call__ 实际上指向 _wrapped_call_impl
    __call__ = _wrapped_call_impl
    
    def _wrapped_call_impl(self, *args, **kwargs):
        # 如果已编译，调用编译版本
        if self._compiled_call_impl is not None:
            return self._compiled_call_impl(*args, **kwargs)
        else:
            return self._call_impl(*args, **kwargs)
    
    def _call_impl(self, *args, **kwargs):
        # 执行 forward_pre_hooks
        # ...
        
        # 调用 forward 方法（被 dynamo_ctx 包装过）
        result = self.forward(*args, **kwargs)
        
        # 执行 forward_hooks
        # ...
        return result
```

**调用路径**:
```
compiled_model(input_tensor)
  -> OptimizedModule.__call__()
    -> _orig_mod.__call__()  (已被包装)
      -> _call_impl()
        -> forward() (已被 dynamo_ctx 包装)
          -> dynamo_ctx.__call__(forward, *args)
```

---

## 第四步：进入 OptimizeContext

### OptimizeContext 的拦截

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/eval_frame.py:668-704`

```python
class OptimizeContext:
    def __init__(self, compiler_fn, **kwargs):
        self.compiler_fn = compiler_fn  # 后端编译器（如 inductor）
        # ...
    
    def __call__(self, fn):
        """
        包装函数，设置 frame evaluation callback
        """
        def wrapper(*args, **kwargs):
            # 设置 Python 字节码拦截回调
            with enable_python_dispatcher():
                with set_eval_frame(self._custom_eval_frame):
                    # 执行原始函数
                    return fn(*args, **kwargs)
        
        return wrapper
    
    def _custom_eval_frame(self, frame, cache_entry):
        """
        拦截 Python 字节码执行
        """
        # 调用 _compile 进行编译
        return _compile(
            frame.f_code,
            frame.f_globals,
            frame.f_locals,
            compiler_fn=self.compiler_fn,
            # ...
        )
```

**关键点**:
- `set_eval_frame` 设置 Python 解释器的帧评估回调
- 当执行 `forward` 函数时，会调用 `_custom_eval_frame`
- `_custom_eval_frame` 调用 `_compile` 开始编译

**调用路径**:
```
forward(*args)  (被 dynamo_ctx 包装)
  -> OptimizeContext.__call__(forward)(*args)
    -> wrapper(*args)
      -> set_eval_frame(_custom_eval_frame)
        -> 执行 forward 时触发 _custom_eval_frame
          -> _compile(frame.f_code, ...)
```

---

## 第五步：_compile - 字节码转换的入口

### _compile 函数

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py:650-740`

```python
def _compile(
    code,       # forward 函数的字节码
    globals,    # 全局变量
    locals,     # 局部变量（包含 self 和 input_tensor）
    compiler_fn,  # 后端编译器
    # ...
):
    """
    编译 Python 函数的字节码
    """
    
    # 定义 transform 闭包
    def transform(instructions, code_options):
        """
        转换字节码指令
        """
        print(f"原始指令: {instructions}")
        
        # 步骤1: 创建符号执行器
        tracer = InstructionTranslator(
            instructions=instructions,
            code=code,
            f_locals=locals,  # 包含 input_tensor
            f_globals=globals,
            compiler_fn=compiler_fn,
            # ...
        )
        
        # 步骤2: 运行符号执行
        tracer.run()
        
        # 步骤3: 获取输出
        output = tracer.output
        
        # 步骤4: 用新指令替换原始指令
        instructions[:] = output.output_instructions
        
        print(f"新指令: {instructions}")
        
        return output
    
    # 调用 transform_code_object 进行字节码转换
    new_code = transform_code_object(code, transform)
    
    return new_code
```

**调用路径**:
```
_compile(code, globals, locals, compiler_fn)
  -> transform_code_object(code, transform)
    -> transform(instructions, code_options)
      -> InstructionTranslator(...)
      -> tracer.run()
```

---

## 第六步：transform_code_object - 字节码转换

### 字节码的反汇编和重组

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/bytecode_transformation.py:1414-1430`

```python
def transform_code_object(code, transformations):
    """
    转换 Python code 对象
    """
    
    # 步骤1: 反汇编字节码为指令列表
    instructions = list(dis.get_instructions(code))
    
    # 原始指令示例:
    # LOAD_FAST    'self'
    # LOAD_ATTR    'linear'
    # LOAD_FAST    'x'
    # CALL_FUNCTION 1
    # RETURN_VALUE
    
    # 步骤2: 调用 transformations 修改指令
    # transformations 就是上面的 transform 函数
    transformations(instructions, code_options)
    
    # 步骤3: 重新组装为新的 code 对象
    new_code = assemble(instructions, code.co_firstlineno)
    
    # 新指令示例:
    # LOAD_GLOBAL  '__compiled_fn_1'
    # LOAD_FAST    'x'
    # CALL_FUNCTION 1
    # RETURN_VALUE
    
    return new_code
```

**关键点**:
- `transformations` 函数会修改 `instructions` 列表
- 修改后的指令会调用编译后的函数，而不是原始代码

**调用路径**:
```
transform_code_object(code, transform)
  -> dis.get_instructions(code)  # 反汇编
  -> transform(instructions, ...)  # 修改指令
  -> assemble(instructions)  # 重新组装
```

---

## 第七步：InstructionTranslator - 符号执行器的创建

### 初始化符号执行器

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py:3308-3461`

```python
class InstructionTranslator:
    def __init__(
        self,
        instructions,  # 字节码指令列表
        code,          # code 对象
        f_locals,      # 局部变量：{'self': model, 'x': input_tensor}
        f_globals,     # 全局变量
        compiler_fn,   # 后端编译器
        # ...
    ):
        # 保存基本信息
        self.instructions = instructions
        self.f_code = code
        self.f_locals = f_locals
        self.f_globals = f_globals
        
        # 创建 OutputGraph（管理 FX Graph）
        self.output = OutputGraph(
            compiler_fn=compiler_fn,
            # ...
        )
        
        # 创建符号局部变量表
        self.symbolic_locals = {}
        
        # 为每个局部变量创建 LazyVariableTracker
        for name, value in f_locals.items():
            source = LocalSource(name)
            self.symbolic_locals[name] = LazyVariableTracker.create(
                value=value,   # 真实值（如 input_tensor）
                source=source  # 变量来源
            )
        
        # 符号栈（用于模拟 Python 栈操作）
        self.stack = []
        
        # 指令指针
        self.instruction_pointer = 0
```

**关键点**:
- `f_locals` 包含真实的输入 tensor
- 每个变量都被包装为 `LazyVariableTracker`
- `LazyVariableTracker` 延迟创建实际的符号表示

**此时的状态**:
```python
f_locals = {
    'self': <MyModel object>,
    'x': <Tensor [batch, features]>  # 真实的 tensor 对象
}

symbolic_locals = {
    'self': LazyVariableTracker(
        _cache=LazyCache(
            value=<MyModel>,           # 原始模型对象
            source=LocalSource('self'),
            vt=None                     # 尚未实例化
        )
    ),
    'x': LazyVariableTracker(
        _cache=LazyCache(
            value=<Tensor [batch, features]>,  # 原始 tensor（真实数据！）
            source=LocalSource('x'),
            vt=None                            # 尚未实例化，没有 proxy！
        )
    )
}
```

**注意**：
- `LazyVariableTracker` 此时**只包含原始 tensor**，还**没有 proxy**
- proxy 要等到 `realize()` 后才会创建

**调用路径**:
```
InstructionTranslator.__init__(...)
  -> 为每个 f_locals 创建 LazyVariableTracker
    -> LazyVariableTracker.create(value, source)
```

---

## 第八步：tracer.run() - 开始符号执行

### 符号执行的主循环

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py:3497-3550`

```python
class InstructionTranslator:
    def run(self):
        """
        符号执行的主函数
        """
        try:
            # 循环处理每条指令
            while self.instruction_pointer < len(self.instructions):
                self.step()  # 处理一条指令
        except StopIteration:
            # 正常结束
            pass
        
        return self
    
    def step(self):
        """
        处理一条指令
        """
        # 获取当前指令
        inst = self.instructions[self.instruction_pointer]
        
        # 根据操作码分发到对应的处理函数
        handler = self.dispatch_table[inst.opcode]
        
        # 调用处理函数
        handler(self, inst)
        
        # 移动到下一条指令
        self.instruction_pointer += 1
```

**dispatch_table** (指令分发表):
```python
dispatch_table = {
    opcode.LOAD_FAST: InstructionTranslator.LOAD_FAST,
    opcode.LOAD_ATTR: InstructionTranslator.LOAD_ATTR,
    opcode.CALL_FUNCTION: InstructionTranslator.CALL_FUNCTION,
    opcode.BINARY_ADD: InstructionTranslator.BINARY_ADD,
    opcode.RETURN_VALUE: InstructionTranslator.RETURN_VALUE,
    # ... 其他指令
}
```

**调用路径**:
```
tracer.run()
  -> while 循环:
    -> step()
      -> dispatch_table[opcode](self, inst)
```

---

## 第九步：LOAD_FAST - 加载输入参数

### 处理 LOAD_FAST 指令

假设我们要执行：`x = input_tensor`

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py:1390-1424`

```python
def LOAD_FAST(self, inst):
    """
    加载局部变量
    指令: LOAD_FAST 'x'
    """
    name = inst.argval  # 'x'
    
    # 从 symbolic_locals 获取 LazyVariableTracker
    value = self.symbolic_locals[name]
    
    # unwrap() 会触发实例化
    # 如果是 Tensor，会创建 TensorVariable
    self.push(value.unwrap())
```

### unwrap 触发实例化

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/lazy.py:71-76`

```python
class LazyVariableTracker:
    def unwrap(self):
        """
        获取实际的 VariableTracker
        """
        if self._cache._obj is not None:
            # 已经实例化过了
            return self._cache._obj
        else:
            # 第一次访问，触发实例化
            return self
```

如果还未实例化，会在后续使用时触发 `realize()`：

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/lazy.py:64-69`

```python
def realize(self):
    """
    强制实例化
    """
    return self._cache.realize()
```

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/lazy.py:22-35`

```python
class LazyCache:
    def realize(self):
        """
        创建实际的 VariableTracker
        """
        if self._obj is None:
            # 调用 VariableTracker.build
            self._obj = self._create_fn(
                self._value,   # 真实的 Tensor
                self._source   # LocalSource('x')
            )
        return self._obj
```

**状态变化**:

**LOAD_FAST 执行前**:
```python
symbolic_locals['x'] = LazyVariableTracker(
    _cache=LazyCache(
        value=<真实 tensor，shape=[2,128]>,  # 有数据
        source=LocalSource('x'),
        vt=None  # 还没创建
    )
)
# 此时: 没有 proxy，没有 FX Graph 节点
```

**LOAD_FAST 执行中（触发 realize）**:
```python
# unwrap() 发现 vt=None，触发 realize()
# realize() 调用 VariableTracker.build()
# build() 调用 VariableBuilder.wrap_tensor()
# wrap_tensor() 做三件事:
#   1. 创建 FakeTensor (无数据，只有元数据)
#   2. 在 FX Graph 中添加 placeholder 节点
#   3. 创建 proxy 指向该节点
#   4. 创建 TensorVariable
```

**LOAD_FAST 执行后**:
```python
# 栈上现在有:
stack = [
    TensorVariable(
        proxy=Proxy(node=%l_x_),           # 新创建的！
        example_value=FakeTensor(          # 新创建的！
            shape=[2, 128],
            dtype=float32,
            device='cuda'
            # 无数据
        ),
        source=LocalSource('x')
    )
]

# FX Graph 中现在有:
graph.nodes = [
    Node(op='placeholder', target='l_x_', name='l_x_')  # 新添加的！
]

# LazyVariableTracker 现在已经 realized:
symbolic_locals['x']._cache.vt = <上面的 TensorVariable>
symbolic_locals['x']._cache.value = None  # 原始 tensor 已删除
```

**调用路径**:
```
LOAD_FAST(inst)
  -> self.symbolic_locals['x'].unwrap()
    -> 检查 is_realized() -> False
    -> LazyVariableTracker.realize()
      -> LazyCache.realize()
        -> VariableTracker.build(原始tensor, source)
          -> VariableBuilder.wrap_tensor(原始tensor)
            -> wrap_to_fake_tensor_and_record()  # 创建 FakeTensor
            -> create_graph_input()               # 添加节点，创建 proxy
            -> wrap_fx_proxy()                    # 创建 TensorVariable
          -> 返回 TensorVariable
      -> 保存到 _cache.vt
      -> 删除 _cache.value (原始tensor)
    -> 返回 TensorVariable
  -> push 到栈
```

---

## 第十步：VariableTracker.build - 创建符号变量

### build 方法

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/base.py:531-540`

```python
class VariableTracker:
    @staticmethod
    def build(value, source):
        """
        根据 value 的类型创建对应的 VariableTracker
        """
        if source is not None:
            # 使用 VariableBuilder
            return VariableBuilder(tx, source)(value)
        else:
            # 使用 SourcelessBuilder
            return SourcelessBuilder(tx)(value)
```

### VariableBuilder.__call__

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/builder.py:405-427`

```python
class VariableBuilder:
    def __call__(self, value):
        """
        创建 VariableTracker
        """
        # 检查缓存
        if value in self.cache:
            return self.cache[value]
        
        # 调用 _wrap 创建
        result = self._wrap(value)
        
        # 加入缓存
        self.cache[value] = result
        
        return result
    
    def _wrap(self, value):
        """
        根据类型分发
        """
        # 使用类型分发找到对应的 wrap 函数
        wrap_fn = self._type_dispatch.get(type(value))
        
        if isinstance(value, torch.Tensor):
            return self.wrap_tensor(value)
        elif isinstance(value, torch.nn.Module):
            return self.wrap_module(value)
        # ... 其他类型
```

**调用路径**:
```
VariableTracker.build(tensor, source)
  -> VariableBuilder(tx, source)(tensor)
    -> _wrap(tensor)
      -> wrap_tensor(tensor)
```

---

## 第十一步：wrap_tensor - 创建 TensorVariable

### wrap_tensor 的核心逻辑

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/builder.py:1646-1813`

```python
def wrap_tensor(self, value: torch.Tensor):
    """
    将真实 Tensor 转换为 TensorVariable
    这个函数非常关键！这里才真正创建 proxy！
    
    输入: value = <真实的 Tensor，包含实际数据>
    输出: TensorVariable(包含 proxy 和 FakeTensor)
    """
    
    # 步骤1: 创建 FakeTensor（只有元数据）
    fake_tensor = wrap_to_fake_tensor_and_record(
        value,
        tx=self.tx,
        source=self.source
    )
    # fake_tensor 只包含 shape, dtype, device 等元数据
    # 不包含实际数据，用于推断形状和类型
    
    # 步骤2: 在 FX Graph 中创建 placeholder 节点
    # 这是 proxy 第一次被创建的地方！
    proxy = self.tx.output.create_graph_input(
        name=self.name,      # 'l_x_'
        type_=type(value),   # torch.Tensor
        source=self.source   # LocalSource('x')
    )
    # 这会在 FX Graph 中添加一个 placeholder 节点
    # proxy 是指向这个节点的句柄
    
    # 步骤3: 包装为 TensorVariable
    tensor_variable = wrap_fx_proxy(
        tx=self.tx,
        proxy=proxy,              # FX Proxy（指向 graph node）
        example_value=fake_tensor, # FakeTensor（元数据）
        source=self.source
    )
    
    return tensor_variable
```

**关键理解**：
- **LazyVariableTracker** 创建时：只包含原始 tensor，没有 proxy
- **realize() 触发时**：调用 `wrap_tensor`
- **wrap_tensor 执行时**：
  1. 从原始 tensor 创建 FakeTensor
  2. **创建 FX Graph 的 placeholder 节点**
  3. **创建 proxy 指向这个节点**
  4. 创建 TensorVariable，包含 proxy 和 FakeTensor

### 三个关键步骤详解

#### 步骤1: wrap_to_fake_tensor_and_record

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/builder.py:2996-3093`

```python
def wrap_to_fake_tensor_and_record(value, tx, source):
    """
    将真实 Tensor 转换为 FakeTensor
    """
    # 创建 FakeTensor
    fake_tensor = tx.fake_mode.from_tensor(value)
    
    # fake_tensor 示例:
    # FakeTensor(shape=[2, 128], dtype=torch.float32, device='cuda:0')
    # 注意: 没有实际数据！
    
    # 记录元数据（用于 guard）
    tx.output.tracked_fakes.append(fake_tensor)
    
    return fake_tensor
```

#### 步骤2: create_graph_input

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/output_graph.py:600-650`

```python
class OutputGraph:
    def create_graph_input(self, name, type_, source):
        """
        在 FX Graph 中创建 placeholder 节点
        """
        # 在 FX Graph 中添加 placeholder
        proxy = self.graph.placeholder(name)
        
        # 记录为图输入
        grapharg = GraphArg(source, proxy, ...)
        self.graphargs.append(grapharg)
        
        return proxy
```

**此时 FX Graph 的状态**:
```python
# 图中现在有一个节点:
graph = torch.fx.Graph()
# %l_x_ : [num_users=0] = placeholder[target=l_x_]
```

#### 步骤3: wrap_fx_proxy

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/builder.py:2763-2850`

```python
def wrap_fx_proxy(tx, proxy, example_value, source):
    """
    根据 example_value 的类型创建对应的 Variable
    """
    if isinstance(example_value, torch.Tensor):
        return TensorVariable.create(
            tx=tx,
            proxy=proxy,
            example_value=example_value,
            source=source
        )
    # ... 其他类型
```

### TensorVariable 的结构

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/tensor.py:126-205`

```python
class TensorVariable(VariableTracker):
    """
    Tensor 的符号表示
    """
    def __init__(self, proxy, example_value, ...):
        super().__init__()
        self.proxy = proxy              # FX Proxy（指向 graph 节点）
        self.example_value = example_value  # FakeTensor（元数据）
        
        # 从 FakeTensor 提取属性
        self.dtype = example_value.dtype
        self.device = example_value.device
        self.shape = example_value.shape
        self.ndim = example_value.ndim
        # ...
```

**现在的状态**:
```python
# Python 栈
stack = [
    TensorVariable(
        proxy=Proxy(node=%l_x_),
        example_value=FakeTensor(shape=[2, 128], dtype=float32),
        dtype=torch.float32,
        device=device('cuda:0'),
        shape=[2, 128]
    )
]

# FX Graph
graph = torch.fx.Graph()
# %l_x_ : [num_users=0] = placeholder[target=l_x_]
```

**调用路径总结**:
```
LOAD_FAST('x')
  -> LazyVariableTracker.unwrap()
    -> realize()
      -> VariableTracker.build(real_tensor, source)
        -> VariableBuilder.wrap_tensor(real_tensor)
          -> wrap_to_fake_tensor_and_record()  # 创建 FakeTensor
          -> create_graph_input()               # 添加 placeholder 节点到 FX Graph
          -> wrap_fx_proxy()                    # 创建 TensorVariable
          -> TensorVariable.create()
```

---

## 第十二步：BINARY_ADD - 构建计算图

### 执行算术运算

假设代码是：`y = x + 1`

**指令序列**:
```
LOAD_FAST    'x'      # 加载 x（已完成，栈上有 TensorVariable）
LOAD_CONST   1        # 加载常量 1
BINARY_ADD            # 执行加法
STORE_FAST   'y'      # 存储到 y
```

### BINARY_ADD 的处理

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py:2843`

```python
# BINARY_ADD 被映射到 stack_op(operator.add)
BINARY_ADD = stack_op(operator.add)
```

### stack_op 的实现

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py:415-423`

```python
def stack_op(fn):
    """
    将 Python 运算符包装为字节码处理器
    """
    def impl(self, inst):
        # 从栈顶弹出两个操作数
        right = self.pop()  # ConstantVariable(1)
        left = self.pop()   # TensorVariable(proxy=%l_x_)
        
        # 调用 BuiltinVariable 处理
        result = BuiltinVariable.call_function(
            self, fn, [left, right], {}
        )
        
        # 将结果压入栈
        self.push(result)
    
    return impl
```

### BuiltinVariable.call_function

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/builtin.py:1093-1111`

```python
class BuiltinVariable:
    @staticmethod
    def call_function(tx, fn, args, kwargs):
        """
        处理内置函数调用
        fn = operator.add
        args = [TensorVariable, ConstantVariable(1)]
        """
        # 创建 BuiltinVariable
        builtin_var = BuiltinVariable(fn)
        
        # 调用其 call_function 方法
        return builtin_var.call_function(tx, args, kwargs)
    
    def call_function(self, tx, args, kwargs):
        """
        实例方法：处理函数调用
        """
        # 获取处理器
        handler = self._make_handler(tx, args, kwargs)
        
        # 调用处理器
        return handler(tx, args, kwargs)
```

### _make_handler 选择处理器

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/builtin.py:782-832`

```python
def _make_handler(self, tx, args, kwargs):
    """
    根据参数类型选择处理器
    """
    # 如果参数中有 TensorVariable
    if any(isinstance(arg, TensorVariable) for arg in args):
        # 使用 tensor 操作处理器
        return self._handle_insert_op_in_graph
    
    # ... 其他情况
```

### _handle_insert_op_in_graph - 添加到 FX Graph

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/builtin.py:952-1064`

```python
def _handle_insert_op_in_graph(self, tx, args, kwargs):
    """
    在 FX Graph 中插入操作节点
    """
    # 步骤1: 提取 FX Proxy
    proxy_args = [arg.as_proxy() for arg in args]
    # proxy_args = [Proxy(node=%l_x_), 1]
    
    # 步骤2: 在 FX Graph 中创建节点
    result_proxy = tx.output.create_proxy(
        "call_function",
        self.fn,           # operator.add
        args=tuple(proxy_args),
        kwargs={}
    )
    # 这会在 FX Graph 中添加:
    # %add : [num_users=0] = call_function[target=operator.add](args = (%l_x_, 1))
    
    # 步骤3: 计算输出的 FakeTensor
    fake_args = [arg.example_value for arg in args if hasattr(arg, 'example_value')]
    # 用 FakeTensor 执行操作，得到输出形状
    example_value = self.fn(*fake_args)
    # example_value = FakeTensor(shape=[2, 128], dtype=float32)
    
    # 步骤4: 包装为新的 TensorVariable
    result = wrap_fx_proxy(
        tx=tx,
        proxy=result_proxy,
        example_value=example_value
    )
    
    return result
```

### OutputGraph.create_proxy

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/output_graph.py:800-850`

```python
class OutputGraph:
    def create_proxy(self, kind, target, args, kwargs):
        """
        在 FX Graph 中创建节点
        kind: "call_function"
        target: operator.add
        args: (Proxy(%l_x_), 1)
        """
        # 在图中创建节点
        node = self.graph.create_node(
            kind,
            target,
            args,
            kwargs
        )
        
        # 包装为 Proxy
        proxy = Proxy(node, self.tracer)
        
        return proxy
```

**此时 FX Graph 的状态**:
```python
graph = torch.fx.Graph()
# %l_x_ : [num_users=1] = placeholder[target=l_x_]
# %add : [num_users=0] = call_function[target=operator.add](args = (%l_x_, 1), kwargs = {})
```

**调用路径总结**:
```
BINARY_ADD
  -> stack_op(operator.add)
    -> BuiltinVariable.call_function(tx, operator.add, [left, right], {})
      -> _make_handler() -> _handle_insert_op_in_graph
        -> tx.output.create_proxy("call_function", operator.add, ...)
          -> graph.create_node(...)  # 添加节点到 FX Graph
        -> wrap_fx_proxy()  # 创建新的 TensorVariable
```

---

## 第十三步：RETURN_VALUE - 触发编译

### 完成图构建

**指令**: `RETURN_VALUE`

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py:3706-3707`

```python
def RETURN_VALUE(self, inst):
    """
    处理返回语句
    """
    self._return(inst)
```

### _return 触发编译

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py:3668-3697`

```python
def _return(self, inst):
    """
    完成符号执行，触发编译
    """
    # 检查是否有可编译的内容
    if self.output.count_calls() == 0:
        # 没有可编译的操作，跳过
        raise exc.SkipFrame("no content in function call")
    
    # 关键: 调用 compile_subgraph
    self.output.compile_subgraph(
        self,
        reason=GraphCompileReason(
            "return_value",
            [self.frame_summary()],
            graph_break=False
        )
    )
```

**此时的状态**:
```python
# FX Graph 已经构建完成
graph:
  %l_x_ : placeholder[target=l_x_]
  %add : call_function[target=operator.add](args = (%l_x_, 1))
  # 还缺少 output 节点，将在 compile_subgraph 中添加

# 符号栈
stack = [TensorVariable(proxy=Proxy(%add))]
```

**调用路径**:
```
RETURN_VALUE
  -> _return(inst)
    -> output.compile_subgraph(tx, reason=...)
```

---

## 第十四步：compile_subgraph - 编译子图

### compile_subgraph 的核心流程

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/output_graph.py:993-1192`

```python
class OutputGraph:
    def compile_subgraph(self, tx, partial_convert=False, reason=None):
        """
        编译当前构建的子图
        """
        print(f"编译原因: {reason}")  # "return_value"
        
        # 步骤1: 获取栈上的值（返回值）
        stack_values = list(tx.stack)
        # stack_values = [TensorVariable(proxy=Proxy(%add))]
        
        # 步骤2: 清理图
        self.cleanup_graph()
        
        # 步骤3: 创建 FakeRootModule
        root = FakeRootModule(self.nn_modules)
        
        # 步骤4: 调用 compile_and_call_fx_graph
        self.add_output_instructions(
            self.compile_and_call_fx_graph(
                tx, 
                stack_values,  # 返回值
                root, 
                {}
            )
        )
```

### compile_and_call_fx_graph - 核心编译函数

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/output_graph.py:1345-1464`

```python
class OutputGraph:
    def compile_and_call_fx_graph(self, tx, rv, root, replaced_outputs):
        """
        编译 FX Graph 并生成调用指令
        """
        # 步骤1: 添加 output 节点
        output_node = self.create_node(
            "output",
            "output",
            (self.current_tracer.create_arg(
                tuple(x.as_proxy() for x in rv)
            ),),
            {}
        )
        # 现在图完整了:
        # %l_x_ : placeholder
        # %add : call_function[operator.add]
        # output : output[args=(%add,)]
        
        # 步骤2: 移除未使用的节点
        self.remove_unused_graphargs()
        
        # 步骤3: 创建 GraphModule
        gm = fx.GraphModule(root, self.graph)
        
        # 步骤4: 打印图（如果启用调试）
        print(f"FX Graph:\n{gm.graph}")
        
        # 步骤5: 调用后端编译器
        compiled_fn = self.call_user_compiler(gm)
        
        # 步骤6: 安装编译后的函数为全局变量
        name = unique_id("__compiled_fn")
        self.install_global(name, compiled_fn)
        
        # 步骤7: 生成新的字节码指令
        instructions = [
            create_instruction("LOAD_GLOBAL", argval=name),
            create_instruction("LOAD_FAST", argval="x"),
            create_instruction("CALL_FUNCTION", arg=1),
            create_instruction("RETURN_VALUE"),
        ]
        
        return instructions
```

**此时的 FX Graph**:
```python
graph = torch.fx.Graph()
# %l_x_ : [num_users=1] = placeholder[target=l_x_]
# %add : [num_users=1] = call_function[target=operator.add](args = (%l_x_, 1), kwargs = {})
# return add
```

**调用路径**:
```
compile_subgraph(tx, reason="return_value")
  -> compile_and_call_fx_graph(tx, stack_values, root, {})
    -> create_node("output", ...)  # 添加 output 节点
    -> fx.GraphModule(root, self.graph)  # 创建 GraphModule
    -> call_user_compiler(gm)  # 调用后端编译器
```

---

## 第十五步：call_user_compiler - 调用 Inductor

### 调用后端编译器

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_dynamo/output_graph.py:1480-1559`

```python
class OutputGraph:
    def call_user_compiler(self, gm: fx.GraphModule):
        """
        调用用户指定的编译器
        """
        return self._call_user_compiler(gm)
    
    def _call_user_compiler(self, gm: fx.GraphModule):
        """
        实际的编译器调用
        """
        print(f"调用编译器: {self.compiler_fn.__name__}")
        
        # 获取示例输入（用于编译）
        example_inputs = self.example_inputs()
        
        # 调用编译器函数
        # 如果是 Inductor: compiler_fn = torch._inductor.compile_fx
        compiled_fn = self.compiler_fn(gm, example_inputs)
        
        return compiled_fn
```

**调用路径**:
```
call_user_compiler(gm)
  -> _call_user_compiler(gm)
    -> self.compiler_fn(gm, example_inputs)
      -> torch._inductor.compile_fx(gm, example_inputs)
```

---

## 第十六步：Inductor 编译流程

### compile_fx - Inductor 入口

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py:1707-1760`

```python
def compile_fx(
    model_: GraphModule,
    example_inputs_: Sequence[InputType],
    inner_compile=compile_fx_inner,
    # ...
):
    """
    Inductor 的主入口函数
    """
    print(f"Inductor 编译开始")
    print(f"输入图节点数: {len(model_.graph.nodes)}")
    
    # 调用内部编译函数
    return inner_compile(model_, example_inputs_, ...)
```

### _compile_fx_inner - 核心编译逻辑

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py:636-800`

```python
def _compile_fx_inner(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    **graph_kwargs
):
    """
    Inductor 内部编译函数
    """
    print(f"输入图:\n{gm.graph}")
    
    # 调用 fx_codegen_and_compile
    compiled_graph = fx_codegen_and_compile(
        gm, 
        example_inputs, 
        inputs_to_check,
        **graph_kwargs
    )
    
    return compiled_graph
```

### fx_codegen_and_compile - 选择编译模式

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py:1274-1295`

```python
def fx_codegen_and_compile(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    inputs_to_check: Sequence[int],
    **graph_kwargs
):
    """
    选择编译模式
    """
    # 通常使用进程内编译
    scheme = _InProcessFxCompile()
    
    # 调用 codegen_and_compile
    return scheme.codegen_and_compile(
        gm, 
        example_inputs, 
        inputs_to_check, 
        graph_kwargs
    )
```

**调用路径**:
```
torch._inductor.compile_fx(gm, example_inputs)
  -> _compile_fx_inner(gm, example_inputs, ...)
    -> fx_codegen_and_compile(gm, ...)
      -> _InProcessFxCompile().codegen_and_compile(gm, ...)
```

---

## 第十七步：codegen_and_compile - 第一阶段融合

### 进入编译和代码生成

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py:893-1006`

```python
class _InProcessFxCompile(FxCompile):
    def codegen_and_compile(self, gm, example_inputs, inputs_to_check, graph_kwargs):
        """
        代码生成和编译
        """
        print(f"开始代码生成和编译")
        
        # 步骤1: view 转 reshape
        view_to_reshape(gm)
        
        # 步骤2: FakeTensor 传播
        with torch.no_grad():
            fake_mode = fake_tensor_prop(gm, example_inputs)
        
        # 步骤3: 记录原始输出步长
        record_original_output_strides(gm)
        
        # 步骤4: [第一阶段融合] FX Graph 优化
        with V.set_fake_mode(fake_mode):
            print("===== 第一阶段融合: FX Graph 层优化 =====")
            _recursive_post_grad_passes(gm, is_inference=is_inference)
            print(f"优化后的图:\n{gm.graph}")
```

### _recursive_post_grad_passes

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py:370-379`

```python
def _recursive_post_grad_passes(gm: GraphModule, is_inference: bool = False):
    """
    递归应用 post-grad 优化
    """
    # 对子图递归应用
    for subgraph_name in _get_subgraph_names(gm):
        subgraph = getattr(gm, subgraph_name)
        _recursive_post_grad_passes(subgraph, is_inference)
    
    # 应用优化 passes
    post_grad_passes(gm, is_inference)
```

### post_grad_passes - FX Graph 层融合

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/fx_passes/post_grad.py:75-204`

```python
def post_grad_passes(gm: torch.fx.GraphModule, is_inference: bool):
    """
    FX Graph 层面的优化和融合
    操作的是 FX Graph 节点，不是 IR
    """
    print("=== Post-Grad Passes 开始 ===")
    
    # Pass 1: 死代码消除
    if config.dce:
        print("  - 死代码消除")
        gm.graph.eliminate_dead_code()
    
    # Pass 2: 重排序以提高局部性
    if is_inference and config.reorder_for_locality:
        print("  - 重排序优化")
        reorder_for_locality(gm)
    
    # Pass 3: 模式匹配融合（核心）
    if config.pattern_matcher:
        print("  - 模式匹配融合")
        
        # 3.1 批次融合
        print("    * 批次融合")
        group_batch_fusion_passes(gm, pre_grad=False)
        
        # 3.2 移除无操作
        print("    * 移除 noop 操作")
        remove_noop_ops(gm)
        
        # 3.3 应用融合模式（3轮）
        for i, patterns in enumerate(pass_patterns):
            print(f"    * 应用融合模式 pass {i}")
            patterns.apply(gm)
        
        # 3.4 特定融合优化
        for pass_name in config.post_grad_fusion_options:
            print(f"    * {pass_name}")
            if pass_name in POST_GRAD_FUSIONS:
                continue
            pattern_matcher_pass = POST_GRAD_PATTERNS[pass_name]
            pattern_matcher_pass.apply(gm)
            
        # 融合模式包括:
        # - normalization_aten_pass: BN/LN 融合
        # - split_cat_aten_pass: split+cat 融合
        # - pad_aten_mm_pass: pad+matmul 融合
        # - decompose_mm_pass: 矩阵乘分解
    
    # Pass 4: Inplace 优化
    print("  - Inplace 优化")
    reinplace_inplaceable_ops(gm)
    
    # Pass 5: 重新编译图
    print("  - 重新编译图")
    gm.recompile()
    
    print("=== Post-Grad Passes 完成 ===")
```

**融合示例**:

**优化前**:
```python
# FX Graph
%split : call_function[target=torch.split](args = (%input, 2))
%getitem_0 : call_function[target=operator.getitem](args = (%split, 0))
%getitem_1 : call_function[target=operator.getitem](args = (%split, 1))
%cat : call_function[target=torch.cat](args = ([%getitem_0, %getitem_1],))
```

**优化后**:
```python
# FX Graph
# split+getitem+cat 被融合，直接返回原始输入
return input
```

**调用路径**:
```
codegen_and_compile(gm, ...)
  -> _recursive_post_grad_passes(gm, is_inference)
    -> post_grad_passes(gm, is_inference)
      -> group_batch_fusion_passes(gm)
      -> remove_noop_ops(gm)
      -> pass_patterns[i].apply(gm)
      -> POST_GRAD_PATTERNS[pass_name].apply(gm)
```

---

## 第十八步：GraphLowering - 第二阶段融合准备

### 创建 GraphLowering

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py:1094-1119`

```python
# 在 codegen_and_compile 中继续

# 步骤5: 创建 GraphLowering 对象
print("===== 第二阶段准备: FX Graph -> Inductor IR =====")
graph = GraphLowering(
    gm,
    example_inputs=example_inputs,
    shape_env=shape_env,
    graph_id=graph_id,
    # ...
)

# 步骤6: 运行 lowering
with V.set_graph_handler(graph):
    print("运行 graph lowering...")
    graph.run(*example_inputs)
    print("Lowering 完成")
```

### GraphLowering.run - FX 到 IR 转换

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/graph.py:875-877`

```python
class GraphLowering(torch.fx.Interpreter):
    def run(self, *args):
        """
        遍历 FX Graph，将每个节点转为 Inductor IR
        """
        return super().run(*args)
```

`GraphLowering` 继承自 `torch.fx.Interpreter`，会遍历 FX Graph 的每个节点：

```python
# 伪代码
for node in gm.graph.nodes:
    if node.op == "placeholder":
        # 创建 IR Buffer 表示输入
        buffer = ir.InputBuffer(...)
    
    elif node.op == "call_function":
        if node.target == operator.add:
            # 转为 IR 操作
            ir_node = ir.Pointwise(
                device=device,
                dtype=dtype,
                inner_fn=lambda idx: ops.add(input_loader(idx), 1)
            )
    
    elif node.op == "output":
        # 记录输出
        self.graph_outputs = [...]
```

**转换示例**:

**FX Graph**:
```python
%l_x_ : placeholder
%add : call_function[operator.add](%l_x_, 1)
output: (%add,)
```

**Inductor IR**:
```python
buf0 = InputBuffer(name='l_x_', shape=[2, 128])
buf1 = ComputedBuffer(
    name='buf1',
    layout=FixedLayout(device='cuda', dtype=torch.float32, size=[2, 128]),
    data=Pointwise(
        device='cuda',
        dtype=torch.float32,
        inner_fn=lambda idx: ops.add(buf0.load(idx), 1),
        ranges=[2, 128]
    )
)
```

**调用路径**:
```
graph.run(*example_inputs)
  -> torch.fx.Interpreter.run()
    -> 遍历每个 FX 节点
      -> 调用对应的处理方法
        -> 创建 Inductor IR 节点
          -> 记录到 self.buffers
```

---

## 第十九步：Scheduler.codegen - 第二阶段融合

### 代码生成

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/graph.py:1994-2002`

```python
class GraphLowering:
    def codegen(self):
        """
        从 Inductor IR 生成最终代码
        """
        print("===== 第二阶段融合: Inductor IR 层优化 =====")
        
        # 初始化包装代码
        self.init_wrapper_code()
        
        # 更新调度器
        self._update_scheduler()
        
        # 生成代码（这里进行调度器融合）
        print("调度器代码生成...")
        self.scheduler.codegen()
        
        print("代码生成完成")
```

### Scheduler.codegen

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/scheduler.py:4130-4136`

```python
class Scheduler:
    def codegen(self):
        """
        遍历 IR 节点，生成融合后的内核
        """
        return self._codegen(self.nodes)
    
    def _codegen(self, nodes):
        """
        代码生成主循环
        """
        print(f"调度器处理 {len(nodes)} 个节点")
        
        for node in nodes:
            print(f"  处理节点: {node.get_name()}")
            
            # 进入节点上下文
            self.enter_context(node)
            
            # 获取设备
            device = node.get_device()
            
            # 当切换设备或遇到特殊节点时，flush 缓存
            if (device != self.current_device or 
                node.is_extern() or 
                node.is_template()):
                
                print(f"    触发 flush（融合缓存的节点）")
                self.flush()  # 这里进行融合！
            
            # 生成节点代码
            node.codegen()
```

### flush - 真正的融合发生点

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/scheduler.py:3920-3923`

```python
def flush(self):
    """
    将缓存的节点融合成单个内核
    """
    for backend in self.backends.values():
        backend.flush()  # 调用具体后端的 flush
    
    self.free_buffers()
```

**后端 flush 的实现** (以 Triton 为例):

```python
class TritonScheduling(BaseScheduling):
    def flush(self):
        """
        Triton 后端的融合实现
        """
        if not self.ready_to_flush:
            return
        
        # 获取可以融合的节点
        nodes_to_fuse = self.get_fusable_nodes()
        
        if len(nodes_to_fuse) > 1:
            print(f"    融合 {len(nodes_to_fuse)} 个 pointwise 操作")
            
            # 创建融合内核
            fused_kernel = self.create_fused_kernel(nodes_to_fuse)
            
            # 生成 Triton 代码
            triton_code = fused_kernel.codegen()
            
            # 编译 Triton 内核
            compiled_kernel = triton.compile(triton_code)
```

**融合示例**:

**优化前（3个独立内核）**:
```python
# Kernel 1
buf1 = buf0 + 1

# Kernel 2
buf2 = buf1 * 2

# Kernel 3
buf3 = buf2.relu()
```

**优化后（1个融合内核）**:
```python
# Fused Kernel
buf3 = ((buf0 + 1) * 2).relu()
```

**融合类型**:

1. **Pointwise 融合**: 
   - 逐元素操作融合
   - 例: `(x + 1) * 2` 融合为单个内核

2. **Reduction 融合**:
   - 约简操作融合
   - 例: `x.sum().sqrt()` 融合

3. **Vertical 融合**:
   - 生产者-消费者融合
   - 例: `y = relu(matmul(x, w))` 中的 matmul+relu

4. **Horizontal 融合**:
   - 并行操作融合
   - 例: 多个独立的 `relu` 操作

**调用路径**:
```
graph.codegen()
  -> scheduler.codegen()
    -> _codegen(nodes)
      -> 遍历节点:
        -> flush()  # 触发融合
          -> backend.flush()
            -> get_fusable_nodes()
            -> create_fused_kernel()
            -> triton.compile()
```

---

## 第十九步（进阶）：如何自定义融合策略

### 理解融合策略的关键入口点

在前面的步骤中，我们看到融合发生在两个阶段：
1. **FX Graph 层融合**（post_grad_passes）
2. **Inductor IR 层融合**（scheduler.codegen）

现在我们来详细了解如何在这些关键点插入自定义融合逻辑。

### 关键源码位置总结

**源码文件**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py`

```python
# 完整调用链
_compile_fx_inner(gm, example_inputs, **kwargs)
  ↓
fx_codegen_and_compile(gm, example_inputs, ...)
  ↓
_InProcessFxCompile().codegen_and_compile(gm, ...)
  ↓
├─ _recursive_post_grad_passes(gm)  # ← FX Graph 层融合
│   ↓
│   post_grad_passes(gm)            # ← 模式匹配融合
│
├─ GraphLowering(gm, ...)           # ← 创建 IR lowering 对象
│   ↓
│   graph.run(*example_inputs)       # ← FX Graph → Inductor IR (关键转换点！)
│
└─ graph.compile_to_fn()            # ← 代码生成
    ↓
    scheduler.codegen()              # ← IR 层融合
```

### 融合策略修改的三种方法

#### 方法一：通过配置参数（最简单）

**位置**: 用户代码中

```python
import torch
import torch._inductor.config as config

# 控制融合行为
config.max_fusion_size = 32           # 增大融合组大小
config.aggressive_fusion = True       # 激进融合模式
config.triton.cudagraphs = False      # 简化调试

# 控制特定模式的融合
config.pattern_matcher = True         # 启用模式匹配
config.fx_graph_cache = False         # 关闭缓存（便于测试）

# 自定义调度器行为
config.scheduler_fusion_config = {
    "max_pointwise_chain": 10,
    "allow_cross_reduction": True,
    "prefer_vertical_fusion": True,
}

# 编译模型
model = torch.compile(model, backend="inductor")
```

#### 方法二：在 FX Graph 层插入分析逻辑（推荐）

**位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py:1118`

在 `graph.run()` 之后插入自定义分析代码：

```python
# torch/_inductor/compile_fx.py
# 在 _InProcessFxCompile.codegen_and_compile 方法中

with V.set_graph_handler(graph):
    graph.run(*example_inputs)  # 第1119行：IR Lowering 的关键入口
    
    # ========== 自定义融合策略分析（在这里插入）==========
    def get_node_type(target) -> str:
        """识别节点类型"""
        target_str = str(target).lower()
        if 'conv' in target_str:
            return 'conv'
        elif 'batch_norm' in target_str or 'native_batch_norm' in target_str:
            return 'bn'
        elif 'relu' in target_str:
            return 'relu'
        elif 'add' in target_str or 'mul' in target_str:
            return 'add'
        elif 'pool' in target_str:
            return 'pool'
        else:
            return 'other'
    
    def detect_conv_bn_relu_pattern(node) -> Optional[dict]:
        """检测 Conv + BN + ReLU 融合模式"""
        if get_node_type(node.target) != 'relu' or len(node.args) == 0:
            return None
        
        bn_node = node.args[0]
        if not isinstance(bn_node, torch.fx.Node) or get_node_type(bn_node.target) != 'bn':
            return None
        
        if len(bn_node.args) == 0:
            return None
        
        conv_node = bn_node.args[0]
        if isinstance(conv_node, torch.fx.Node) and get_node_type(conv_node.target) == 'conv':
            return {
                'pattern': 'conv_bn_relu',
                'nodes': [conv_node, bn_node, node],
                'node_names': [conv_node.name, bn_node.name, node.name]
            }
        return None
    
    def detect_pointwise_chain(node) -> Optional[dict]:
        """检测 Pointwise 操作链（Add/Mul 连续操作）"""
        if get_node_type(node.target) != 'add':
            return None
        
        chain = [node]
        current = node
        
        # 向前追溯 pointwise 操作链
        while len(current.args) > 0:
            pred = current.args[0]
            if isinstance(pred, torch.fx.Node) and get_node_type(pred.target) == 'add':
                chain.insert(0, pred)
                current = pred
            else:
                break
        
        if len(chain) >= 2:
            return {
                'pattern': 'pointwise_chain',
                'nodes': chain,
                'node_names': [n.name for n in chain],
                'length': len(chain)
            }
        return None
    
    def mark_fusion_group(fusion_info: dict) -> None:
        """标记融合组到节点 meta 中"""
        fusion_id = f"fusion_{fusion_info['pattern']}_{id(fusion_info)}"
        for node in fusion_info['nodes']:
            node.meta['fusion_group'] = fusion_id
            node.meta['fusion_pattern'] = fusion_info['pattern']
    
    def analyze_fusion_opportunities(fx_graph) -> tuple[dict, list]:
        """分析 FX Graph 并识别融合机会"""
        node_stats = {"total": 0, "conv": 0, "bn": 0, "relu": 0, "add": 0, "pool": 0}
        fusion_opportunities = []
        
        for node in fx_graph.nodes:
            if node.op != 'call_function':
                continue
            
            node_stats["total"] += 1
            node_type = get_node_type(node.target)
            if node_type in node_stats:
                node_stats[node_type] += 1
            
            # 尝试匹配各种融合模式
            patterns = [
                detect_conv_bn_relu_pattern(node),
                detect_pointwise_chain(node),
            ]
            
            for pattern in patterns:
                if pattern:
                    fusion_opportunities.append(pattern)
                    mark_fusion_group(pattern)
        
        return node_stats, fusion_opportunities
    
    def print_fusion_analysis(node_stats: dict, fusion_ops: list) -> None:
        """打印融合分析报告"""
        print("\n" + "="*80)
        print("自定义融合策略分析 (FX Graph Level)")
        print("="*80)
        
        print(f"\n节点统计:")
        print(f"  总计: {node_stats['total']:>3d} | Conv: {node_stats['conv']:>2d} | "
              f"BN: {node_stats['bn']:>2d} | ReLU: {node_stats['relu']:>2d} | "
              f"Add/Mul: {node_stats['add']:>2d} | Pool: {node_stats['pool']:>2d}")
        
        print(f"\n融合机会: {len(fusion_ops)} 处")
        for i, fop in enumerate(fusion_ops[:5], 1):
            pattern_display = fop['pattern'].replace('_', ' ').title()
            nodes_display = ' → '.join(fop['node_names'][:3])
            if len(fop['node_names']) > 3:
                nodes_display += f" ... (+{len(fop['node_names']) - 3})"
            print(f"  [{i}] {pattern_display:20s}: {nodes_display}")
        
        if len(fusion_ops) > 5:
            print(f"  ... 还有 {len(fusion_ops) - 5} 处未显示")
        
        print("="*80 + "\n")
    
    # 执行融合分析
    stats, opportunities = analyze_fusion_opportunities(graph.graph)
    print_fusion_analysis(stats, opportunities)
    # ========== 融合策略分析结束 ==========
    
    output_strides: list[Optional[tuple[_StrideExprStr, ...]]] = []
    # ... 后续代码继续 ...
```

**输出示例**:
```
================================================================================
自定义融合策略分析 (FX Graph Level)
================================================================================

节点统计:
  总计: 156 | Conv: 53 | BN: 53 | ReLU: 49 | Add/Mul: 16 | Pool: 1

融合机会: 49 处
  [1] Conv Bn Relu        : conv2d_1 → batch_norm_1 → relu_1
  [2] Conv Bn Relu        : conv2d_2 → batch_norm_2 → relu_2
  [3] Pointwise Chain     : add_1 → mul_1 → add_2
  [4] Conv Bn Relu        : conv2d_3 → batch_norm_3 → relu_3
  [5] Pointwise Chain     : add_3 → mul_2
  ... 还有 44 处未显示
================================================================================
```

#### 方法三：修改 Scheduler 的融合规则（高级）

**位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/scheduler.py`

创建自定义调度器：

```python
# my_custom_scheduler.py
from torch._inductor.scheduler import Scheduler

class CustomScheduler(Scheduler):
    """自定义调度器，修改融合规则"""
    
    def can_fuse(self, node1, node2):
        """
        重写融合判断逻辑
        
        返回 True 表示可以融合，False 表示不能融合
        """
        # 调用原始逻辑
        original_result = super().can_fuse(node1, node2)
        
        # 添加自定义规则：强制融合所有 pointwise 操作
        if self.is_pointwise(node1) and self.is_pointwise(node2):
            print(f"✓ 强制融合: {node1.get_name()} + {node2.get_name()}")
            return True
        
        # 阻止大型 reduction 操作融合
        if self.is_large_reduction(node1) or self.is_large_reduction(node2):
            print(f"✗ 阻止融合: {node1.get_name()} 或 {node2.get_name()} 太大")
            return False
        
        return original_result
    
    def is_large_reduction(self, node):
        """判断是否是大型 reduction"""
        return hasattr(node, 'get_numel') and node.get_numel() > 1000000
    
    def is_pointwise(self, node):
        """判断是否是 pointwise 操作"""
        from torch._inductor.ir import Pointwise
        return isinstance(node.node, Pointwise)

# 应用自定义调度器（通过 monkey patch）
import torch._inductor.scheduler as scheduler_module
scheduler_module.Scheduler = CustomScheduler
```

### 关键转换点详解

#### 1. FX Graph → Inductor IR 的精确位置

**源码**: `compile_fx.py:1119`

```python
with V.set_graph_handler(graph):
    graph.run(*example_inputs)  # ← 这里！FX Graph 被转换为 IR
```

**转换过程**:
```python
# 转换前：FX Graph
%l_x_ : placeholder[target=l_x_]
%add : call_function[target=operator.add](args=(%l_x_, 1))
%mul : call_function[target=operator.mul](args=(%add, 2))
return mul

# graph.run() 执行后：Inductor IR
buf0 = InputBuffer(name='l_x_', shape=[2, 128])
buf1 = Pointwise(  # add
    inner_fn=lambda idx: ops.add(buf0.load(idx), 1)
)
buf2 = Pointwise(  # mul
    inner_fn=lambda idx: ops.mul(buf1.load(idx), 2)
)
```

#### 2. IR 融合的触发时机

**源码**: `scheduler.py:3920`

```python
def flush(self):
    """
    将缓存的 pointwise/reduction 节点融合成单个内核
    """
    # 这里进行实际的融合操作
    for backend in self.backends.values():
        backend.flush()
```

**融合示例**:
```python
# 融合前（3 个独立内核）
@triton.jit
def kernel_1(in_ptr, out_ptr):
    out = in_ptr + 1

@triton.jit
def kernel_2(in_ptr, out_ptr):
    out = in_ptr * 2

@triton.jit
def kernel_3(in_ptr, out_ptr):
    out = relu(in_ptr)

# 融合后（1 个融合内核）
@triton.jit
def fused_kernel(in_ptr, out_ptr):
    tmp = in_ptr + 1
    tmp = tmp * 2
    out = relu(tmp)
```

### 扩展示例：添加自定义融合模式

#### 场景：融合 MatMul + Add (常见于 GEMM+Bias)

```python
def detect_matmul_add_pattern(node) -> Optional[dict]:
    """检测 MatMul + Add 融合模式（GEMM + Bias）"""
    if 'add' not in str(node.target).lower():
        return None
    
    if len(node.args) > 0:
        mm_node = node.args[0]
        if isinstance(mm_node, torch.fx.Node) and 'matmul' in str(mm_node.target).lower():
            return {
                'pattern': 'matmul_add',
                'nodes': [mm_node, node],
                'node_names': [mm_node.name, node.name],
                'optimization': 'Can use cuBLAS Epilogue Fusion'
            }
    return None

# 在 analyze_fusion_opportunities 中添加
patterns = [
    detect_conv_bn_relu_pattern(node),
    detect_pointwise_chain(node),
    detect_matmul_add_pattern(node),  # ← 新增
]
```

### 调试技巧

#### 1. 查看 FX Graph 在各个阶段的变化

```python
# 在 compile_fx.py 中添加打印
with V.set_fake_mode(fake_mode):
    print("===== Post-Grad Passes 前 =====")
    print(gm.code)
    
    _recursive_post_grad_passes(gm, is_inference=is_inference)
    
    print("===== Post-Grad Passes 后 =====")
    print(gm.code)
```

#### 2. 查看生成的 Triton 代码

```bash
# 设置环境变量
export TORCH_COMPILE_DEBUG=1
export TORCH_LOGS="+inductor"

# 运行代码
python your_script.py

# 查看生成的代码
cat torch_compile_debug/run_*/torchinductor/*/output_code.py
```

#### 3. 禁用特定优化进行对比

```python
import torch._inductor.config as config

# 禁用模式匹配融合
config.pattern_matcher = False

# 禁用 IR 层融合
config.triton.pointwise_fusion = False
config.triton.reduction_fusion = False

# 对比性能
model_no_fusion = torch.compile(model, backend="inductor")
# 然后重新启用优化测试对比
```

### 完整的修改流程示例

```python
# 步骤 1: 在用户代码中设置配置
import torch
import torch._inductor.config as config

config.debug = True
config.trace.enabled = True

# 步骤 2: 修改 compile_fx.py（如前面所示）
# 在 graph.run() 后添加融合分析逻辑

# 步骤 3: 运行并查看结果
model = torch.compile(model, backend="inductor")
output = model(input)

# 步骤 4: 查看生成的融合报告
# 会在终端看到：
# ================================================================================
# 自定义融合策略分析 (FX Graph Level)
# ================================================================================
# 节点统计: ...
# 融合机会: ...
```

### 性能验证

```python
import torch.utils.benchmark as benchmark

# 测试原始模型
def test_original():
    return model(input)

# 测试编译后模型（默认融合）
compiled_model = torch.compile(model)
def test_compiled():
    return compiled_model(input)

# 测试自定义融合策略
# (修改 compile_fx.py 后)
def test_custom_fusion():
    return compiled_model(input)

# 对比
t0 = benchmark.Timer(stmt='test_original()', globals=globals())
t1 = benchmark.Timer(stmt='test_compiled()', globals=globals())

print(f"原始模型: {t0.blocked_autorange().mean * 1000:.2f} ms")
print(f"编译模型: {t1.blocked_autorange().mean * 1000:.2f} ms")
print(f"加速比: {t0.blocked_autorange().mean / t1.blocked_autorange().mean:.2f}x")
```

### 总结

**融合策略修改的三个层次**:

1. **配置层**（最简单）
   - 修改 `torch._inductor.config`
   - 适合快速实验

2. **FX Graph 层**（推荐）
   - 在 `graph.run()` 后插入分析逻辑
   - 可以标记融合组
   - 不破坏原有流程

3. **IR/Scheduler 层**（高级）
   - 修改 `Scheduler.can_fuse()`
   - 完全控制融合决策
   - 需要深入了解 IR 结构

**关键入口点**:
- FX Graph → IR: `compile_fx.py:1119` (`graph.run()`)
- IR 融合: `scheduler.py:3920` (`flush()`)
- 模式匹配: `fx_passes/post_grad.py` (`post_grad_passes()`)

---

## 第二十步：生成最终代码

### 编译和包装

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/graph.py:2085-2154`

```python
class GraphLowering:
    def compile_to_module(self):
        """
        编译为 Python 模块
        """
        # 生成包装代码
        wrapper_code, kernel_code = self.codegen()
        
        # 写入文件
        code_file = self.write_code(wrapper_code, kernel_code)
        
        # 编译模块
        mod = self.load_module(code_file)
        
        # 输出调试信息
        print(f"输出代码写入: {mod.__file__}")
        V.debug.output_code(mod.__file__)
        
        return mod
```

### 生成的 Python 代码示例

```python
# output_code.py

import torch
from torch import empty_strided, as_strided
from torch._inductor.triton_heuristics import grid

@triton.jit
def triton_poi_fused_add_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    """
    融合后的 Triton 内核
    """
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 1.0
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)

def call(args):
    """
    包装函数
    """
    primals_1 = args[0]  # 输入 tensor
    
    # 分配输出 buffer
    buf0 = empty_strided((2, 128), (128, 1), device='cuda', dtype=torch.float32)
    
    # 调用 Triton 内核
    grid = lambda meta: (triton.cdiv(256, meta['XBLOCK']),)
    triton_poi_fused_add_0[grid](primals_1, buf0, 256, XBLOCK=128)
    
    return (buf0,)
```

### 调试文件输出

**目录结构**:
```
./torch_compile_debug/
└── run_2025_12_19_10_28_53_819926-pid_70840/
    └── torchinductor/
        └── model__0_inference_0.0/
            ├── fx_graph_runnable.py       # 可运行的 FX Graph
            ├── fx_graph_transformed.py    # 优化后的 FX Graph
            ├── output_code.py             # 生成的包装代码
            ├── __compiled_fn_0.py         # 编译后的内核
            └── debug.log                  # 调试日志
```

**写入时机**:

**源码位置**: `/usr/local/lib/python3.10/dist-packages/torch/_inductor/debug.py:562-563`

```python
def output_code(self, filename: str):
    """
    复制生成的代码到调试目录
    """
    shutil.copy(filename, self.filename("output_code.py"))
```

---

## 完整流程总结

### 端到端调用链

```
第1步: torch.compile(model)
  -> torch._dynamo.optimize(backend)
  -> OptimizedModule(model, OptimizeContext(...))

第2步: compiled_model(input)
  -> OptimizedModule.__call__()
  -> forward() [被 dynamo_ctx 包装]
  -> OptimizeContext.__call__()

第3步: set_eval_frame(_custom_eval_frame)
  -> _custom_eval_frame()
  -> _compile(code, globals, locals, compiler_fn)

第4步: transform_code_object(code, transform)
  -> dis.get_instructions(code)  # 反汇编
  -> transform(instructions)     # 转换

第5步: InstructionTranslator.__init__()
  -> 创建 symbolic_locals
  -> 为每个变量创建 LazyVariableTracker

第6步: tracer.run()
  -> while 循环处理指令

第7-11步: 处理各种指令
  LOAD_FAST('x')
    -> LazyVariableTracker.unwrap()
    -> VariableBuilder.wrap_tensor()
    -> wrap_to_fake_tensor_and_record()  # 创建 FakeTensor
    -> create_graph_input()               # 添加 placeholder 到 FX Graph
    -> TensorVariable.create()            # 创建符号变量

  BINARY_ADD
    -> stack_op(operator.add)
    -> BuiltinVariable.call_function()
    -> _handle_insert_op_in_graph()
    -> tx.output.create_proxy("call_function", operator.add, ...)
    -> graph.create_node()                # 添加操作节点到 FX Graph

第12步: RETURN_VALUE
  -> _return()
  -> output.compile_subgraph()

第13步: compile_and_call_fx_graph()
  -> create_node("output")              # 添加 output 节点
  -> fx.GraphModule(root, graph)        # 创建 GraphModule
  -> call_user_compiler(gm)             # 调用后端编译器

第14步: torch._inductor.compile_fx(gm, example_inputs)
  -> _compile_fx_inner()
  -> fx_codegen_and_compile()
  -> _InProcessFxCompile.codegen_and_compile()

第15步: [第一阶段融合] post_grad_passes(gm)
  -> group_batch_fusion_passes()        # 批次融合
  -> remove_noop_ops()                  # 移除无操作
  -> pass_patterns.apply()              # 应用融合模式
  -> POST_GRAD_PATTERNS.apply()         # 特定融合优化

第16步: [Lowering] GraphLowering.run()
  -> 遍历 FX Graph 节点
  -> 转换为 Inductor IR

第17步: [第二阶段融合] scheduler.codegen()
  -> _codegen(nodes)
  -> 遍历 IR 节点
  -> flush()                            # 触发融合
    -> backend.flush()                  # 后端融合实现
    -> create_fused_kernel()            # 创建融合内核

第18步: 代码生成和编译
  -> 生成 Triton/C++ 代码
  -> 编译为 .so
  -> 加载到 Python

第19步: 生成新字节码
  -> 创建调用编译函数的指令
  -> 替换原始指令

第20步: 返回编译后的函数
  -> 后续调用直接执行编译后的代码
```

### 关键数据结构演变

**阶段1: Python 对象（f_locals 中）**
```python
input_tensor: torch.Tensor(
    data=[actual data...],  # 真实数据
    shape=[2, 128],
    device='cuda'
)
```

**阶段2: LazyVariableTracker（初始化时）**
```python
LazyVariableTracker(
    _cache=LazyCache(
        value=<真实 tensor，包含数据>,  # 原始对象
        source=LocalSource('x'),          # 变量来源
        vt=None                           # 还没创建 VariableTracker
    )
)
# 注意: 此时没有 proxy！
```

**阶段3: realize() 后 -> TensorVariable（包含 proxy）**
```python
TensorVariable(
    proxy=Proxy(node=%l_x_),                      # FX Graph 节点的句柄
    example_value=FakeTensor(                      # 只有元数据
        shape=[2, 128], 
        dtype=float32,
        device='cuda'
        # 没有实际数据！
    ),
    source=LocalSource('x')
)
# 注意: 
# 1. proxy 是在 realize() 时才创建的
# 2. FakeTensor 不包含数据，只有 shape/dtype/device
# 3. 原始 tensor 的数据已经不需要了
```

**关键转换点**:
```
原始 tensor (有数据) 
  -> LazyVariableTracker (持有原始 tensor)
    -> realize() 触发
      -> wrap_tensor() 执行
        -> 创建 FakeTensor (无数据，只有元数据)
        -> 创建 FX Graph placeholder 节点
        -> 创建 proxy 指向节点
        -> 创建 TensorVariable (包含 proxy + FakeTensor)
```

**阶段4: FX Graph**
```python
%l_x_ : placeholder[target=l_x_]
%add : call_function[target=operator.add](args=(%l_x_, 1))
return add
```

**阶段5: Inductor IR**
```python
buf0 = InputBuffer(name='l_x_', shape=[2, 128])
buf1 = Pointwise(
    inner_fn=lambda idx: ops.add(buf0.load(idx), 1)
)
```

**阶段6: Triton 代码**
```python
@triton.jit
def kernel(in_ptr, out_ptr, n):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + idx)
    y = x + 1.0
    tl.store(out_ptr + idx, y)
```

**阶段7: 编译后的函数**
```python
def __compiled_fn_1(x):
    return compiled_kernel(x)
```

### IR 层次结构

1. **Python 字节码** - 最原始
2. **FX Graph** - 图结构表示
3. **Inductor IR** - 底层 Buffer/Loop IR
4. **Triton/C++ 代码** - 生成的源代码
5. **LLVM IR** - 编译器中间表示
6. **机器码** - 最终执行

### 优化发生的位置

**Dynamo 层**:
- 减少 Python 解释器开销
- 捕获完整的计算图

**FX Graph 层** (第一阶段融合):
- 算子融合: split+cat, conv+bn
- 算子替换: 高效实现
- 死代码消除
- 图重排序

**Inductor IR 层** (第二阶段融合):
- 内核融合: pointwise, reduction
- 循环优化: 循环融合、循环展开
- 内存优化: 布局优化
- 自动调优: block size, warp 配置

**Triton/CUDA 层**:
- 向量化
- 共享内存优化
- 寄存器分配

---

## 调试和验证

### 查看生成的文件

```bash
# 设置环境变量
export TORCH_COMPILE_DEBUG=1
export TORCH_LOGS="+dynamo,+inductor"

# 运行代码
python your_script.py

# 查看生成的文件
ls -R ./torch_compile_debug/
```

### 查看 FX Graph

```python
import torch

model = torch.compile(model)
output = model(input)  # 第一次运行触发编译

# 查看生成的图
print(model._orig_mod.forward.graph)
```

### 禁用某些优化

```python
# 禁用模式匹配
torch._inductor.config.pattern_matcher = False

# 禁用特定融合
torch._inductor.config.post_grad_fusion_options = []

# 禁用 CUDA Graphs
torch._inductor.config.triton.cudagraphs = False
```

### 性能分析

```python
import torch.utils.benchmark as benchmark

# 测试编译前
t0 = benchmark.Timer(
    stmt='model(input)',
    globals={'model': model, 'input': input}
)

# 测试编译后
compiled_model = torch.compile(model)
t1 = benchmark.Timer(
    stmt='model(input)',
    globals={'model': compiled_model, 'input': input}
)

print(f"原始: {t0.timeit(100).mean * 1000:.2f} ms")
print(f"编译: {t1.timeit(100).mean * 1000:.2f} ms")
```

---

这个递进式教程从用户调用 `torch.compile()` 开始，一步步追踪整个编译流程，清楚地展示了：

1. 如何拦截 Python 执行
2. 如何进行符号执行
3. 如何构建 FX Graph
4. 如何触发编译
5. 如何进行两阶段融合
6. 如何生成最终代码

每一步都标注了源码位置和调用路径，便于理解和调试。
