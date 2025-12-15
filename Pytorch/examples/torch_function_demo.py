"""
演示 __torch_function__ 协议的工作机制
展示 torch.relu(x) 如何触发自定义类的 __torch_function__
"""

import torch
import torch.fx as fx

print("=" * 70)
print("演示 __torch_function__ 协议")
print("=" * 70)

# ============================================================
# 示例 1: 最简单的 __torch_function__ 实现
# ============================================================

class InterceptTensor:
    """实现 __torch_function__ 协议的最简单示例"""
    
    def __init__(self, data):
        self.data = data
        print(f"创建 InterceptTensor，数据: {self.data}")
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """拦截所有 torch.* 函数"""
        print(f"\n[拦截] 捕获到函数调用!")
        print(f"  函数: {func.__name__}")
        print(f"  参数类型: {types}")
        print(f"  位置参数数量: {len(args)}")
        
        # 提取真实数据并调用原函数
        real_args = tuple(
            a.data if isinstance(a, InterceptTensor) else a 
            for a in args
        )
        result_data = func(*real_args, **(kwargs or {}))
        
        # 包装返回值
        return InterceptTensor(result_data)
    
    def __repr__(self):
        return f"InterceptTensor({self.data})"

print("\n" + "=" * 70)
print("[示例 1] 简单的 __torch_function__ 拦截")
print("-" * 70)

x = InterceptTensor(torch.tensor([1.0, -2.0, 3.0]))
print(f"\n执行: torch.relu(x)")
y = torch.relu(x)
print(f"\n结果: {y}")

# ============================================================
# 示例 2: 追踪调用栈
# ============================================================

import inspect

class StackTraceTensor:
    """追踪 __torch_function__ 的调用栈"""
    
    def __init__(self, data):
        self.data = data
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        print(f"\n[__torch_function__ 被调用]")
        print(f"  函数: {func}")
        
        # 打印调用栈
        print(f"\n  调用栈:")
        frame = inspect.currentframe()
        for i, frame_info in enumerate(inspect.getouterframes(frame)[:6]):
            filename = frame_info.filename.split('/')[-1]
            print(f"    [{i}] {filename}:{frame_info.lineno} in {frame_info.function}")
        
        # 执行原函数
        real_args = tuple(
            a.data if isinstance(a, StackTraceTensor) else a 
            for a in args
        )
        result_data = func(*real_args, **(kwargs or {}))
        return StackTraceTensor(result_data)

print("\n" + "=" * 70)
print("[示例 2] 查看 __torch_function__ 的调用路径")
print("-" * 70)

x = StackTraceTensor(torch.randn(3))
print("\n执行: torch.relu(x)")
y = torch.relu(x)

# ============================================================
# 示例 3: Proxy 的 __torch_function__ 行为
# ============================================================

print("\n" + "=" * 70)
print("[示例 3] Proxy 的 __torch_function__ 实际行为")
print("-" * 70)

class Model(torch.nn.Module):
    def forward(self, x):
        # 这里会触发 Proxy.__torch_function__
        y = torch.relu(x)
        z = torch.add(y, 1.0)
        return z

print("\n开始符号追踪...")
model = Model()
traced = fx.symbolic_trace(model)

print("\n生成的图:")
print(traced.graph)

print("\n详细节点信息:")
for node in traced.graph.nodes:
    if node.op == 'call_function':
        print(f"  {node.name}:")
        print(f"    操作: {node.op}")
        print(f"    目标: {node.target.__name__ if hasattr(node.target, '__name__') else node.target}")
        print(f"    参数: {node.args}")

# ============================================================
# 示例 4: 多个 Proxy 参数的情况
# ============================================================

print("\n" + "=" * 70)
print("[示例 4] 多个参数的 __torch_function__")
print("-" * 70)

class DetailedInterceptTensor:
    """展示多参数情况的详细信息"""
    
    def __init__(self, data, name="unnamed"):
        self.data = data
        self.name = name
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        print(f"\n[拦截] {func.__name__}")
        
        # 显示每个参数的信息
        print(f"  参数详情:")
        for i, arg in enumerate(args):
            if isinstance(arg, DetailedInterceptTensor):
                print(f"    args[{i}]: DetailedInterceptTensor(name='{arg.name}')")
            else:
                print(f"    args[{i}]: {type(arg).__name__} = {arg}")
        
        # 执行
        real_args = tuple(
            a.data if isinstance(a, DetailedInterceptTensor) else a 
            for a in args
        )
        result_data = func(*real_args, **(kwargs or {}))
        return DetailedInterceptTensor(result_data, f"result_of_{func.__name__}")
    
    def __repr__(self):
        return f"DetailedInterceptTensor(name='{self.name}')"

x = DetailedInterceptTensor(torch.tensor([1.0, 2.0]), name="x")
y = DetailedInterceptTensor(torch.tensor([3.0, 4.0]), name="y")

print("\n执行: torch.add(x, y)")
z = torch.add(x, y)

print("\n执行: torch.mul(z, 2.0)")
w = torch.mul(z, 2.0)

# ============================================================
# 示例 5: 对比有无 __torch_function__ 的行为
# ============================================================

print("\n" + "=" * 70)
print("[示例 5] 对比：普通 Tensor vs 实现了 __torch_function__ 的类")
print("-" * 70)

# 普通 Tensor
print("\n[普通 Tensor]")
normal_tensor = torch.tensor([1.0, -2.0, 3.0])
print(f"  执行前: type = {type(normal_tensor)}")
result = torch.relu(normal_tensor)
print(f"  执行后: type = {type(result)}")
print(f"  __torch_function__ 被调用: 否")

# 自定义类
print("\n[自定义类 (有 __torch_function__)]")
custom_tensor = InterceptTensor(torch.tensor([1.0, -2.0, 3.0]))
print(f"  执行前: type = {type(custom_tensor)}")
result = torch.relu(custom_tensor)
print(f"  执行后: type = {type(result)}")

# ============================================================
# 示例 6: 检查 PyTorch 内部的实现
# ============================================================

print("\n" + "=" * 70)
print("[示例 6] 检查 PyTorch 函数的属性")
print("-" * 70)

print(f"\ntorch.relu 的类型: {type(torch.relu)}")
print(f"torch.relu 的模块: {torch.relu.__module__ if hasattr(torch.relu, '__module__') else 'N/A'}")
print(f"torch.relu 是否是内建函数: {isinstance(torch.relu, type(abs))}")

# 检查是否有特殊标记
if hasattr(torch.relu, '__wrapped__'):
    print(f"torch.relu.__wrapped__: {torch.relu.__wrapped__}")

# ============================================================
# 示例 7: 演示 C++ -> Python 的回调
# ============================================================

print("\n" + "=" * 70)
print("[示例 7] 演示完整的调用流程")
print("-" * 70)

class VerboseInterceptTensor:
    """详细输出每一步"""
    
    call_count = 0
    
    def __init__(self, data):
        self.data = data
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        cls.call_count += 1
        call_id = cls.call_count
        
        print(f"\n[步骤 {call_id}.1] Python: __torch_function__ 被调用")
        print(f"[步骤 {call_id}.2] Python: 接收到函数 {func.__name__}")
        print(f"[步骤 {call_id}.3] Python: 提取参数...")
        
        real_args = tuple(
            a.data if isinstance(a, VerboseInterceptTensor) else a 
            for a in args
        )
        
        print(f"[步骤 {call_id}.4] Python: 调用真实函数 {func.__name__}(...)")
        result_data = func(*real_args, **(kwargs or {}))
        
        print(f"[步骤 {call_id}.5] Python: 包装返回值")
        result = VerboseInterceptTensor(result_data)
        
        print(f"[步骤 {call_id}.6] Python: 返回给 C++ 层")
        return result

print("\n执行: torch.relu(VerboseInterceptTensor(...))")
print("=" * 70)
print("[用户代码] x = VerboseInterceptTensor(...)")
x = VerboseInterceptTensor(torch.randn(3))

print("\n[用户代码] y = torch.relu(x)")
print("-" * 70)
print("[C++] torch.relu 被调用")
print("[C++] 检查参数类型...")
print("[C++] 发现 VerboseInterceptTensor 有 __torch_function__")
print("[C++] 准备回调 Python...")

y = torch.relu(x)

print("\n[C++] 接收到 Python 返回值")
print("[C++] 返回给用户代码")
print("=" * 70)
print(f"[用户代码] 得到结果: {type(y)}")

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 70)
print("总结")
print("=" * 70)

print("""
__torch_function__ 的工作流程:

1. 用户调用: torch.relu(custom_obj)
2. C++ 检查: 遍历参数，查找 __torch_function__
3. 发现协议: custom_obj 的类实现了 __torch_function__
4. 回调 Python: C++ 调用 custom_obj.__torch_function__(...)
5. Python 处理: 在 __torch_function__ 中自定义行为
6. 返回结果: Python 返回给 C++，C++ 返回给用户

关键点:
- __torch_function__ 必须是 @classmethod
- PyTorch 在 C++ 层主动检查并调用
- 不是 Python 的标准协议，是 PyTorch 特有的
- 支持任意自定义类型（Proxy, FakeTensor, 自定义 Tensor 等）
""")

