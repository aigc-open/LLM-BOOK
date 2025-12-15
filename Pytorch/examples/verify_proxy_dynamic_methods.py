"""
验证 Proxy 魔术方法的动态生成机制
展示这些方法确实存在，但不是直接定义的
"""

import torch
import torch.fx as fx
import inspect

print("=" * 70)
print("验证 Proxy 魔术方法的动态生成")
print("=" * 70)

# 1. 查看 Proxy 类的所有魔术方法
print("\n[1] Proxy 类拥有的魔术方法:")
print("-" * 70)

proxy_magic_methods = [
    m for m in dir(fx.Proxy) 
    if m.startswith('__') and m.endswith('__') and not m.startswith('_Proxy')
]

# 分类显示
arithmetic = ['__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__', '__pow__']
bitwise = ['__and__', '__or__', '__xor__', '__lshift__', '__rshift__', '__invert__']
comparison = ['__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__']
reflected = ['__radd__', '__rsub__', '__rmul__', '__rtruediv__']
special = ['__call__', '__getattr__', '__getitem__', '__iter__', '__bool__']

print(f"\n算术运算 ({len([m for m in arithmetic if m in proxy_magic_methods])} 个):")
for m in arithmetic:
    if m in proxy_magic_methods:
        print(f"  {m}")

print(f"\n位运算 ({len([m for m in bitwise if m in proxy_magic_methods])} 个):")
for m in bitwise:
    if m in proxy_magic_methods:
        print(f"  {m}")

print(f"\n比较运算 ({len([m for m in comparison if m in proxy_magic_methods])} 个):")
for m in comparison:
    if m in proxy_magic_methods:
        print(f"  {m}")

print(f"\n反射方法 ({len([m for m in reflected if m in proxy_magic_methods])} 个):")
for m in reflected:
    if m in proxy_magic_methods:
        print(f"  {m}")

print(f"\n总计: {len(proxy_magic_methods)} 个魔术方法")

# 2. 检查方法的定义位置
print("\n" + "=" * 70)
print("[2] 检查方法的来源")
print("-" * 70)

print("\n动态生成的方法:")
for method_name in ['__add__', '__sub__', '__mul__']:
    method = getattr(fx.Proxy, method_name)
    print(f"\n  {method_name}:")
    print(f"    函数名: {method.__name__}")
    print(f"    模块: {method.__module__}")
    # 注意: 动态生成的方法无法获取源码
    try:
        source = inspect.getsource(method)
        print(f"    源码: {source[:50]}...")
    except (OSError, TypeError) as e:
        print(f"    源码: 无法获取（动态生成）")

print("\n直接定义的方法:")
for method_name in ['__call__', '__getattr__', '__torch_function__']:
    if hasattr(fx.Proxy, method_name):
        method = getattr(fx.Proxy, method_name)
        print(f"\n  {method_name}:")
        print(f"    函数名: {method.__name__}")
        print(f"    模块: {method.__module__}")
        try:
            source_lines = inspect.getsourcelines(method)[0]
            print(f"    源码（前3行）:")
            for line in source_lines[:3]:
                print(f"      {line.rstrip()}")
        except (OSError, TypeError) as e:
            print(f"    源码: 无法获取")

# 3. 验证方法的行为
print("\n" + "=" * 70)
print("[3] 验证动态生成方法的行为")
print("-" * 70)

class TestModel(torch.nn.Module):
    def forward(self, x, y):
        # 测试各种运算符
        a = x + y      # __add__
        b = x - y      # __sub__
        c = x * y      # __mul__
        d = x / 2      # __truediv__
        e = x // 2     # __floordiv__
        f = x % 2      # __mod__
        g = x ** 2     # __pow__
        h = x == y     # __eq__
        return a

traced = fx.symbolic_trace(TestModel())

print("\n生成的图:")
for node in traced.graph.nodes:
    if node.op == 'call_function':
        print(f"  {node.name}: {node.target} (来自魔术方法)")

# 4. 显示 magic_methods 字典的内容
print("\n" + "=" * 70)
print("[4] torch.fx.graph 中的 magic_methods 定义")
print("-" * 70)

from torch.fx.graph import magic_methods, reflectable_magic_methods

print(f"\nmagic_methods 包含 {len(magic_methods)} 个方法:")
for i, (method, template) in enumerate(list(magic_methods.items())[:10], 1):
    print(f"  {i}. '{method}': {template}")
print(f"  ... 还有 {len(magic_methods) - 10} 个")

print(f"\nreflectable_magic_methods 包含 {len(reflectable_magic_methods)} 个方法:")
for i, (method, template) in enumerate(list(reflectable_magic_methods.items())[:5], 1):
    print(f"  {i}. '{method}': {template}")
print(f"  ... 还有 {len(reflectable_magic_methods) - 5} 个")

# 5. 对比：如果手写需要多少代码
print("\n" + "=" * 70)
print("[5] 动态生成的优势")
print("-" * 70)

total_methods = len(magic_methods) + len(reflectable_magic_methods)
lines_per_method = 7  # 估计每个方法需要约 7 行代码

print(f"\n如果手写所有魔术方法:")
print(f"  需要实现的方法数: {total_methods}")
print(f"  估计代码行数: {total_methods * lines_per_method} 行")
print(f"\n使用动态生成:")
print(f"  核心代码: ~30 行（循环 + setattr）")
print(f"  节省代码: {total_methods * lines_per_method - 30} 行")
print(f"  代码减少: {((total_methods * lines_per_method - 30) / (total_methods * lines_per_method) * 100):.1f}%")

print("\n" + "=" * 70)
print("验证完成!")
print("=" * 70)
print("\n结论:")
print("1. Proxy 类确实拥有 20+ 个魔术方法")
print("2. 这些方法不是直接定义的，而是通过循环 + setattr 动态生成")
print("3. 动态生成大大减少了代码冗余，提高了可维护性")
print("4. 直接定义的方法（如 __call__, __getattr__）有特殊逻辑需求")

