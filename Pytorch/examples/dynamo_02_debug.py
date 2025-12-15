"""
TorchDynamo 实战示例 2：开启调试日志
运行：python dynamo_02_debug.py
"""
import torch
import torch._dynamo as dynamo
import logging

# ============ 关键：开启详细日志 ============
# 设置日志级别
torch._dynamo.config.log_level = logging.INFO
torch._dynamo.config.verbose = True

# 打印生成的代码
torch._dynamo.config.output_code = True

# 打印 Guard
torch._dynamo.config.print_guards = True

# ============================================

def simple_add(x, y):
    """最简单的函数：两个 Tensor 相加"""
    return x + y

# 测试数据
x = torch.randn(3)
y = torch.randn(3)

print("=" * 60)
print("开始编译...")
print("=" * 60)
compiled_fn = torch.compile(simple_add)

print("\n" + "=" * 60)
print("第一次执行（会触发编译）:")
print("=" * 60)
result = compiled_fn(x, y)
print(f"结果: {result}")

print("\n" + "=" * 60)
print("第二次执行（使用缓存）:")
print("=" * 60)
result = compiled_fn(x, y)
print(f"结果: {result}")

print("\n" + "=" * 60)
print("第三次执行（形状改变，重新编译）:")
print("=" * 60)
x3 = torch.randn(5)  # 不同的形状
y3 = torch.randn(5)
result = compiled_fn(x3, y3)
print(f"结果: {result}")

