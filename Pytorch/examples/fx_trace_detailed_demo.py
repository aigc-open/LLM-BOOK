"""
FX 符号追踪详细演示
展示 fx.symbolic_trace() 每一步的执行过程
"""

import torch
import torch.fx as fx

# ============================================================
# 示例 1: 基础模型追踪
# ============================================================

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        y = self.linear(x)
        z = torch.relu(y)
        return z

print("=" * 60)
print("示例 1: 基础模型追踪")
print("=" * 60)

model = SimpleModel()
traced = fx.symbolic_trace(model)

print("\n生成的计算图:")
print(traced.graph)

print("\n生成的 Python 代码:")
print(traced.code)

print("\n遍历图节点:")
print(f"{'Node Name':<15} {'Op Type':<15} {'Target':<25} {'Args'}")
print("-" * 80)
for node in traced.graph.nodes:
    args_str = ', '.join(str(arg) for arg in node.args)
    print(f"{node.name:<15} {node.op:<15} {str(node.target):<25} {args_str}")

print("\n执行测试:")
x = torch.randn(3, 10)
original_output = model(x)
traced_output = traced(x)
print(f"原始输出 shape: {original_output.shape}")
print(f"追踪输出 shape: {traced_output.shape}")
print(f"结果一致: {torch.allclose(original_output, traced_output)}")

# ============================================================
# 示例 2: 复杂模型 - 多个操作
# ============================================================

class ComplexModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3)
        self.bn = torch.nn.BatchNorm2d(16)
        self.pool = torch.nn.MaxPool2d(2)
    
    def forward(self, x):
        # Conv -> BN -> ReLU -> Pool
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # 算术运算
        x = x * 2.0
        x = x + 1.0
        
        return x

print("\n" + "=" * 60)
print("示例 2: 复杂模型追踪")
print("=" * 60)

complex_model = ComplexModel()
traced_complex = fx.symbolic_trace(complex_model)

print("\n生成的计算图:")
print(traced_complex.graph)

print("\n生成的代码 (前 20 行):")
code_lines = traced_complex.code.split('\n')
for i, line in enumerate(code_lines[:20], 1):
    print(f"{i:2d}: {line}")

# ============================================================
# 示例 3: 查看节点详细信息
# ============================================================

print("\n" + "=" * 60)
print("示例 3: 节点详细分析")
print("=" * 60)

for i, node in enumerate(traced.graph.nodes, 1):
    print(f"\n节点 {i}: {node.name}")
    print(f"  类型 (op):      {node.op}")
    print(f"  目标 (target):  {node.target}")
    print(f"  参数 (args):    {node.args}")
    print(f"  关键字参数:      {node.kwargs}")
    print(f"  使用者数量:      {len(node.users)}")
    if node.users:
        user_names = [u.name for u in node.users.keys()]
        print(f"  被使用于:        {', '.join(user_names)}")

# ============================================================
# 示例 4: 图变换 - 算子替换
# ============================================================

print("\n" + "=" * 60)
print("示例 4: 图变换 - 将 ReLU 替换为 GELU")
print("=" * 60)

def replace_relu_with_gelu(graph_module: fx.GraphModule) -> fx.GraphModule:
    """将图中的 ReLU 替换为 GELU"""
    graph = graph_module.graph
    
    for node in graph.nodes:
        if node.op == 'call_function' and node.target == torch.relu:
            print(f"找到 ReLU 节点: {node.name}")
            # 替换目标函数
            node.target = torch.nn.functional.gelu
            print(f"   替换为 GELU")
    
    # 重新编译
    graph_module.recompile()
    return graph_module

# 复制一个模型进行替换
traced_modified = fx.symbolic_trace(SimpleModel())
traced_modified = replace_relu_with_gelu(traced_modified)

print("\n修改后的代码:")
print(traced_modified.code)

print("\n验证替换:")
x = torch.randn(3, 10)
output_gelu = traced_modified(x)
print(f"输出 shape: {output_gelu.shape}")

# ============================================================
# 示例 5: 统计图信息
# ============================================================

print("\n" + "=" * 60)
print("示例 5: 图统计信息")
print("=" * 60)

def analyze_graph(graph: fx.Graph):
    """分析图的统计信息"""
    stats = {
        'placeholder': 0,
        'call_module': 0,
        'call_function': 0,
        'call_method': 0,
        'get_attr': 0,
        'output': 0
    }
    
    for node in graph.nodes:
        stats[node.op] += 1
    
    print("\n节点类型统计:")
    for op_type, count in stats.items():
        if count > 0:
            print(f"  {op_type:<20} : {count}")
    
    total = sum(stats.values())
    print(f"  {'总计':<20} : {total}")

analyze_graph(traced.graph)
analyze_graph(traced_complex.graph)

# ============================================================
# 示例 6: 导出为 DOT 格式 (可视化)
# ============================================================

print("\n" + "=" * 60)
print("示例 6: 图的邻接关系")
print("=" * 60)

def print_graph_structure(graph: fx.Graph):
    """打印图的邻接结构"""
    print("\n节点依赖关系:")
    for node in graph.nodes:
        if node.op == 'placeholder':
            print(f"  [输入] {node.name}")
        elif node.op == 'output':
            print(f"  [输出] {node.name} <- {node.args[0]}")
        else:
            deps = []
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    deps.append(arg.name)
            deps_str = ', '.join(deps) if deps else '(无)'
            print(f"  [计算] {node.name} <- [{deps_str}]")

print_graph_structure(traced.graph)

print("\n" + "=" * 60)
print("演示完成!")
print("=" * 60)

