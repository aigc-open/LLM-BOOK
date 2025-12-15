"""
TorchDynamo 实战示例 5：完整调试流程
运行：python dynamo_05_full_debug.py
"""
import torch
import torch._dynamo as dynamo
import logging

# ============ 配置 ============
# 日志
torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True

# 输出生成的代码
torch._dynamo.config.output_code = True

# 打印 Guard
torch._dynamo.config.print_guards = True

# ============ 测试函数 ============
def my_function(x, y):
    """简单的测试函数"""
    # 步骤 1: 加法
    z = x + y
    
    # 步骤 2: ReLU
    z = torch.relu(z)
    
    # 步骤 3: 乘法
    result = z * 2
    
    return result

# ============ 执行 ============
print("=" * 60)
print("编译中...")
print("=" * 60)

compiled_fn = torch.compile(my_function)

print("\n" + "=" * 60)
print("第一次运行（触发编译）")
print("=" * 60)

x = torch.randn(4)
y = torch.randn(4)
result = compiled_fn(x, y)

print(f"\n输入 x: {x}")
print(f"输入 y: {y}")
print(f"输出: {result}")

print("\n" + "=" * 60)
print("第二次运行（使用缓存）")
print("=" * 60)

x2 = torch.randn(4)
y2 = torch.randn(4)
result2 = compiled_fn(x2, y2)

print(f"\n输入 x: {x2}")
print(f"输入 y: {y2}")
print(f"输出: {result2}")

print("\n" + "=" * 60)
print("第三次运行（形状改变，重新编译）")
print("=" * 60)

x3 = torch.randn(8)  # 不同的形状！
y3 = torch.randn(8)
result3 = compiled_fn(x3, y3)

print(f"\n输入 x: {x3}")
print(f"输入 y: {y3}")
print(f"输出: {result3}")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)

