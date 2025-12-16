"""
测试 TorchInductor lowerings 字典的构建和使用机制

演示：
1. lowerings 字典的结构
2. 如何查找已注册的算子
3. lowering 函数的调用过程
"""

import torch
from torch._inductor import lowering
from torch._inductor.ir import TensorBox, Pointwise
import torch._dynamo as dynamo

def explore_lowerings_dict():
    """探索 lowerings 字典的内容"""
    print("=" * 80)
    print("1. lowerings 字典概览")
    print("=" * 80)
    
    print(f"\n已注册的 lowering 函数数量: {len(lowering.lowerings)}")
    
    # 按命名空间分组统计
    from collections import defaultdict
    stats = defaultdict(int)
    
    for op in lowering.lowerings.keys():
        if hasattr(op, '__module__'):
            namespace = str(op).split('.')[0] if '.' in str(op) else 'unknown'
            stats[namespace] += 1
    
    print("\n各命名空间的算子数量:")
    for namespace, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {namespace}: {count} 个算子")


def check_specific_ops():
    """检查特定算子的注册情况"""
    print("\n" + "=" * 80)
    print("2. 检查特定算子的注册情况")
    print("=" * 80)
    
    test_ops = [
        ("加法", torch.ops.aten.add.Tensor),
        ("乘法", torch.ops.aten.mul.Tensor),
        ("ReLU", torch.ops.aten.relu.default),
        ("矩阵乘法", torch.ops.aten.mm.default),
        ("卷积", torch.ops.aten.convolution.default),
        ("均值", torch.ops.aten.mean.default),
        ("Softmax", torch.ops.aten._softmax.default),
    ]
    
    for name, op in test_ops:
        is_registered = op in lowering.lowerings
        status = "✓ 已注册" if is_registered else "✗ 未注册"
        print(f"  {name:12s} ({op.name():30s}): {status}")
        
        if is_registered:
            fn = lowering.lowerings[op]
            print(f"               lowering 函数: {fn.__name__} (位于 {fn.__module__})")


def inspect_lowering_function():
    """深入查看一个 lowering 函数"""
    print("\n" + "=" * 80)
    print("3. 深入查看 aten.relu 的 lowering 函数")
    print("=" * 80)
    
    relu_op = torch.ops.aten.relu.default
    relu_lowering = lowering.lowerings.get(relu_op)
    
    if relu_lowering:
        print(f"\n函数对象: {relu_lowering}")
        print(f"函数名称: {relu_lowering.__name__}")
        print(f"函数模块: {relu_lowering.__module__}")
        
        if hasattr(relu_lowering, '__wrapped__'):
            print(f"被包装的原始函数: {relu_lowering.__wrapped__}")
        
        # 查看函数签名
        import inspect
        try:
            sig = inspect.signature(relu_lowering)
            print(f"函数签名: {sig}")
        except:
            print("无法获取函数签名")


def trace_lowering_process():
    """追踪一个简单模型的 lowering 过程"""
    print("\n" + "=" * 80)
    print("4. 追踪简单模型的 lowering 过程")
    print("=" * 80)
    
    def simple_model(x, y):
        """简单的测试模型：加法 + ReLU"""
        z = x + y  # aten.add
        return z.relu()  # aten.relu
    
    # 创建测试输入
    x = torch.randn(10, 10, device='cuda' if torch.cuda.is_available() else 'cpu')
    y = torch.randn(10, 10, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n测试模型定义:")
    print(f"  def simple_model(x, y):")
    print(f"      z = x + y    # aten.add.Tensor")
    print(f"      return z.relu()  # aten.relu.default")
    
    print(f"\n输入形状: x={x.shape}, y={y.shape}")
    print(f"设备: {x.device}")
    
    # 使用 torch.compile 编译
    try:
        compiled_model = torch.compile(simple_model, backend='inductor')
        
        print(f"\n使用 torch.compile(backend='inductor') 编译...")
        result = compiled_model(x, y)
        print(f"编译成功！输出形状: {result.shape}")
        
        # 验证正确性
        eager_result = simple_model(x, y)
        is_correct = torch.allclose(result, eager_result, rtol=1e-5, atol=1e-5)
        print(f"结果正确性: {'✓ 通过' if is_correct else '✗ 失败'}")
        
    except Exception as e:
        print(f"编译失败: {e}")


def show_registration_mechanism():
    """展示注册机制的工作原理"""
    print("\n" + "=" * 80)
    print("5. 注册机制示例（模拟）")
    print("=" * 80)
    
    print("""
注册过程（在 lowering.py 模块加载时自动执行）：

# 步骤 1: 定义全局字典
lowerings: dict = {}

# 步骤 2: 使用装饰器注册算子
@register_lowering(aten.relu)
def relu(x):
    # 将 aten.relu 转换为 Inductor IR
    def inner_fn(idx):
        return ops.maximum(ops.load(x, idx), ops.constant(0.0))
    return Pointwise.create(...)

# 步骤 3: 装饰器自动执行注册
#   ↓
# lowerings[aten.relu.default] = wrapped_relu
#   ↓
# 后续编译时查找：lowering_fn = lowerings.get(target)

关键点：
✓ 全局字典，模块加载时自动填充
✓ 装饰器模式，声明式注册
✓ 延迟查找，编译时才使用
✓ 支持算子重载（同一算子的多个版本）
    """)


def list_common_lowerings():
    """列出常用的 lowering 函数"""
    print("\n" + "=" * 80)
    print("6. 常用算子的 lowering 函数（部分）")
    print("=" * 80)
    
    # 筛选常见的 ATen 算子
    common_patterns = [
        'aten.add', 'aten.sub', 'aten.mul', 'aten.div',
        'aten.relu', 'aten.sigmoid', 'aten.tanh',
        'aten.sum', 'aten.mean', 'aten.max', 'aten.min',
        'aten.matmul', 'aten.mm', 'aten.bmm',
        'aten.conv', 'aten.batch_norm',
    ]
    
    found_ops = []
    for op, fn in lowering.lowerings.items():
        op_str = str(op)
        for pattern in common_patterns:
            if pattern in op_str:
                found_ops.append((op_str, fn.__name__))
                break
    
    print(f"\n找到 {len(found_ops)} 个常用算子:")
    for i, (op_str, fn_name) in enumerate(sorted(found_ops)[:20], 1):
        print(f"  {i:2d}. {op_str:50s} → {fn_name}")
    
    if len(found_ops) > 20:
        print(f"  ... 还有 {len(found_ops) - 20} 个")


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "TorchInductor lowerings 字典探索工具" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        explore_lowerings_dict()
        check_specific_ops()
        inspect_lowering_function()
        show_registration_mechanism()
        list_common_lowerings()
        trace_lowering_process()
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✓ 探索完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

