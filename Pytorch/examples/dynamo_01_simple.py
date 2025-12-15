"""
TorchDynamo 实战示例 1：最简单的例子
运行：python dynamo_01_simple.py
"""
import torch

def simple_add(x, y):
    """最简单的函数：两个 Tensor 相加"""
    return x + y

# 测试数据
x = torch.randn(3)
y = torch.randn(3)

print("=" * 50)
print("原始执行:")
print("=" * 50)
result1 = simple_add(x, y)
print(f"结果: {result1}")

print("\n" + "=" * 50)
print("torch.compile 执行:")
print("=" * 50)

# 使用 torch.compile
compiled_fn = torch.compile(simple_add)
result2 = compiled_fn(x, y)
print(f"结果: {result2}")

print("\n结果是否一致:", torch.allclose(result1, result2))

