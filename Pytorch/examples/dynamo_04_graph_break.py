"""
TorchDynamo 实战示例 4：Graph Break 分析
运行：python dynamo_04_graph_break.py
"""
import torch
import torch._dynamo as dynamo

def with_print(x):
    """包含 print（会导致 Graph Break）"""
    x = x + 1
    print(f"中间值: {x}")  # Graph Break!
    x = x * 2
    return x

def with_item(x):
    """使用 tensor.item()（会导致 Graph Break）"""
    x = x + 1
    if x.item() > 0:  # Graph Break!
        return x * 2
    return x + 1

def no_break(x):
    """没有 Graph Break"""
    x = x + 1
    x = x * 2
    return x

print("=" * 60)
print("分析 Graph Break:")
print("=" * 60)

for fn in [with_print, with_item, no_break]:
    print(f"\n函数: {fn.__name__}")
    print("-" * 40)
    
    explanation = dynamo.explain(fn)
    
    print(f"  Graph 数量: {explanation.graph_count}")
    print(f"  Graph Break 数量: {explanation.graph_break_count}")
    
    if explanation.graph_break_count > 0:
        print(f"  Break 原因:")
        for i, reason in enumerate(explanation.break_reasons):
            print(f"    [{i+1}] {reason}")
    else:
        print(f"  [√] 没有 Graph Break！")

print("\n" + "=" * 60)
print("实际运行:")
print("=" * 60)

compiled_with_print = torch.compile(with_print)
x = torch.randn(3)
print(f"\nwith_print 结果: {compiled_with_print(x)}")

compiled_no_break = torch.compile(no_break)
print(f"no_break 结果: {compiled_no_break(x)}")

