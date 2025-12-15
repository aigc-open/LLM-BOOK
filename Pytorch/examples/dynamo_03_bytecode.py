"""
TorchDynamo 实战示例 3：查看字节码
运行：python dynamo_03_bytecode.py
"""
import torch
import dis

def simple_add(x, y):
    """最简单的函数：两个 Tensor 相加"""
    return x + y

def complex_function(x, y):
    """复杂一点的函数"""
    z = x + y
    if z.sum() > 0:
        return z * 2
    else:
        return z + 1

print("=" * 60)
print("simple_add 的字节码:")
print("=" * 60)
dis.dis(simple_add)

print("\n" + "=" * 60)
print("complex_function 的字节码:")
print("=" * 60)
dis.dis(complex_function)

print("\n" + "=" * 60)
print("运行 TorchDynamo 编译:")
print("=" * 60)

# 开启字节码打印
import torch._dynamo as dynamo
import logging
torch._dynamo.config.log_level = logging.INFO

compiled_fn = torch.compile(complex_function)

x = torch.randn(3)
y = torch.randn(3)
result = compiled_fn(x, y)
print(f"\n结果: {result}")

