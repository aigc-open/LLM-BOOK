"""
演示：字节码到 FX Graph 的转换过程
模拟 TorchDynamo 的符号化执行
"""

import torch
import torch.fx as fx
import dis
from typing import Any


class SimpleOutputGraph:
    """简化的 FX Graph 构建器"""
    
    def __init__(self):
        self.graph = fx.Graph()
        self.node_counter = 0
    
    def create_placeholder(self, name):
        """创建 placeholder 节点"""
        node = self.graph.placeholder(name)
        print(f"    [FX Graph] 创建 placeholder: {name}")
        return node
    
    def create_call_function(self, target, args):
        """创建 call_function 节点"""
        node = self.graph.call_function(target, args)
        args_str = ", ".join(str(arg) for arg in args)
        print(f"    [FX Graph] 创建 call_function: {target.__name__}({args_str})")
        return node
    
    def set_output(self, node):
        """设置输出节点"""
        self.graph.output(node)
        print(f"    [FX Graph] 设置 output: {node}")
    
    def print_graph(self):
        """打印生成的图"""
        print("\n" + "="*60)
        print("生成的 FX Graph:")
        print("="*60)
        print(self.graph)
        
        print("\n" + "="*60)
        print("生成的 Python 代码:")
        print("="*60)
        
        # 生成可读的 Python 代码
        module = fx.GraphModule(torch.nn.Module(), self.graph)
        print(module.code)


class SimpleTensorVariable:
    """简化的 Tensor 符号化变量"""
    
    def __init__(self, node, graph):
        self.node = node
        self.graph = graph
    
    def __add__(self, other):
        """符号化加法"""
        print("  [符号化执行] 执行 __add__")
        
        if isinstance(other, SimpleTensorVariable):
            result_node = self.graph.create_call_function(
                torch.add,
                (self.node, other.node)
            )
        else:
            # Tensor + 常量
            result_node = self.graph.create_call_function(
                torch.add,
                (self.node, other)
            )
        
        return SimpleTensorVariable(result_node, self.graph)
    
    def __mul__(self, other):
        """符号化乘法"""
        print("  [符号化执行] 执行 __mul__")
        
        if isinstance(other, SimpleTensorVariable):
            result_node = self.graph.create_call_function(
                torch.mul,
                (self.node, other.node)
            )
        else:
            # Tensor * 常量
            result_node = self.graph.create_call_function(
                torch.mul,
                (self.node, other)
            )
        
        return SimpleTensorVariable(result_node, self.graph)
    
    def __sub__(self, other):
        """符号化减法"""
        print("  [符号化执行] 执行 __sub__")
        
        if isinstance(other, SimpleTensorVariable):
            result_node = self.graph.create_call_function(
                torch.sub,
                (self.node, other.node)
            )
        else:
            result_node = self.graph.create_call_function(
                torch.sub,
                (self.node, other)
            )
        
        return SimpleTensorVariable(result_node, self.graph)
    
    def __repr__(self):
        return f"TensorVariable({self.node})"


def symbolic_trace_manual(fn, arg_names, title=""):
    """
    手动模拟符号化追踪
    
    Args:
        fn: 要追踪的函数
        arg_names: 参数名列表
        title: 标题
    """
    print("\n" + "="*60)
    print(f"演示: {title}")
    print("="*60)
    
    # 1. 显示字节码
    print("\n[步骤 1] 获取字节码:")
    print("-"*60)
    dis.dis(fn)
    
    # 2. 创建 FX Graph 构建器
    print("\n[步骤 2] 创建符号化执行环境:")
    print("-"*60)
    graph = SimpleOutputGraph()
    
    # 3. 创建符号化参数
    print("  创建符号化变量 (placeholder):")
    symbolic_vars = {}
    for name in arg_names:
        print(f"  - 创建 {name}")
        node = graph.create_placeholder(name)
        symbolic_vars[name] = SimpleTensorVariable(node, graph)
    
    # 4. 符号化执行
    print("\n[步骤 3] 符号化执行函数体:")
    print("-"*60)
    
    # 调用函数（符号化执行）
    result = fn(*symbolic_vars.values())
    
    # 5. 设置输出
    print("\n[步骤 4] 设置输出:")
    print("-"*60)
    graph.set_output(result.node)
    
    # 6. 显示结果
    graph.print_graph()
    
    return graph.graph


def demo1_simple():
    """演示 1: 简单的加法和乘法"""
    
    def my_function(x, y):
        z = x + y
        return z * 2
    
    symbolic_trace_manual(
        my_function,
        ["x", "y"],
        "简单的加法和乘法 (z = x + y; return z * 2)"
    )


def demo2_complex():
    """演示 2: 更复杂的操作"""
    
    def complex_function(a, b, c):
        temp1 = a + b
        temp2 = temp1 * c
        result = temp2 - a
        return result
    
    symbolic_trace_manual(
        complex_function,
        ["a", "b", "c"],
        "复杂操作 (temp1=a+b; temp2=temp1*c; result=temp2-a)"
    )


def demo3_real_torch_fx():
    """演示 3: 使用真实的 torch.fx.symbolic_trace"""
    
    print("\n" + "="*60)
    print("演示: 使用真实的 torch.fx.symbolic_trace")
    print("="*60)
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
        
        def forward(self, x):
            x = self.linear(x)
            x = torch.relu(x)
            return x * 2
    
    model = SimpleModel()
    
    print("\n[原始模型代码]")
    print("-"*60)
    print("""
class SimpleModel(torch.nn.Module):
    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x * 2
    """)
    
    print("\n[FX 符号化追踪]")
    print("-"*60)
    traced = torch.fx.symbolic_trace(model)
    
    print("\n[生成的 FX Graph]")
    print("-"*60)
    print(traced.graph)
    
    print("\n[生成的 Python 代码]")
    print("-"*60)
    print(traced.code)


def demo4_detailed_walkthrough():
    """演示 4: 详细的逐步执行过程"""
    
    print("\n" + "="*60)
    print("演示: 详细的逐步执行过程")
    print("="*60)
    
    def target_function(x, y):
        z = x + y
        return z * 2
    
    print("\n[目标函数]")
    print("-"*60)
    print("""
def target_function(x, y):
    z = x + y
    return z * 2
    """)
    
    print("\n[字节码]")
    print("-"*60)
    dis.dis(target_function)
    
    print("\n[逐指令符号化执行]")
    print("="*60)
    
    graph = SimpleOutputGraph()
    stack = []
    locals_dict = {}
    
    print("\n初始状态:")
    print(f"  stack: {stack}")
    print(f"  locals: {locals_dict}")
    
    print("\n--- 指令 1: LOAD_FAST 0 (x) ---")
    print("  操作: 加载局部变量 x")
    node_x = graph.create_placeholder("x")
    var_x = SimpleTensorVariable(node_x, graph)
    stack.append(var_x)
    print(f"  stack: [{var_x}]")
    
    print("\n--- 指令 2: LOAD_FAST 1 (y) ---")
    print("  操作: 加载局部变量 y")
    node_y = graph.create_placeholder("y")
    var_y = SimpleTensorVariable(node_y, graph)
    stack.append(var_y)
    print(f"  stack: [{var_x}, {var_y}]")
    
    print("\n--- 指令 3: BINARY_ADD ---")
    print("  操作: 二元加法")
    right = stack.pop()
    left = stack.pop()
    print(f"  弹出: {right}, {left}")
    result_add = left + right
    stack.append(result_add)
    print(f"  stack: [{result_add}]")
    
    print("\n--- 指令 4: STORE_FAST 2 (z) ---")
    print("  操作: 存储到局部变量 z")
    value = stack.pop()
    locals_dict["z"] = value
    print(f"  stack: {stack}")
    print(f"  locals: {list(locals_dict.keys())}")
    
    print("\n--- 指令 5: LOAD_FAST 2 (z) ---")
    print("  操作: 加载局部变量 z")
    var_z = locals_dict["z"]
    stack.append(var_z)
    print(f"  stack: [{var_z}]")
    
    print("\n--- 指令 6: LOAD_CONST 1 (2) ---")
    print("  操作: 加载常量 2")
    const_2 = 2
    stack.append(const_2)
    print(f"  stack: [{var_z}, {const_2}]")
    
    print("\n--- 指令 7: BINARY_MULTIPLY ---")
    print("  操作: 二元乘法")
    right = stack.pop()
    left = stack.pop()
    print(f"  弹出: {right}, {left}")
    result_mul = left * right
    stack.append(result_mul)
    print(f"  stack: [{result_mul}]")
    
    print("\n--- 指令 8: RETURN_VALUE ---")
    print("  操作: 返回值")
    ret_val = stack.pop()
    print(f"  返回: {ret_val}")
    graph.set_output(ret_val.node)
    
    graph.print_graph()


def demo5_compare_with_real():
    """演示 5: 对比真实的 TorchDynamo"""
    
    print("\n" + "="*60)
    print("演示: 对比真实的 TorchDynamo 行为")
    print("="*60)
    
    def my_function(x, y):
        z = x + y
        return z * 2
    
    print("\n[使用 torch.compile]")
    print("-"*60)
    
    compiled_fn = torch.compile(my_function, backend="eager")
    
    print("第一次调用 (会触发编译):")
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    result = compiled_fn(x, y)
    print(f"  输入: x.shape={x.shape}, y.shape={y.shape}")
    print(f"  输出: result.shape={result.shape}")
    print("  (TorchDynamo 在后台分析字节码，构建 FX Graph)")
    
    print("\n第二次调用 (使用缓存):")
    x2 = torch.randn(3, 4)
    y2 = torch.randn(3, 4)
    result2 = compiled_fn(x2, y2)
    print(f"  输入: x.shape={x2.shape}, y.shape={y2.shape}")
    print(f"  输出: result.shape={result2.shape}")
    print("  (直接使用缓存的编译结果，极快！)")
    
    # 使用 torch._dynamo.explain 查看详细信息
    print("\n[使用 torch._dynamo.explain 查看详情]")
    print("-"*60)
    
    explanation = torch._dynamo.explain(my_function)(x, y)
    print(f"编译的图数量: {explanation.graph_count}")
    print(f"Graph Break 数量: {explanation.graph_break_count}")
    
    if explanation.graphs:
        print("\n生成的 FX Graph 代码:")
        print(explanation.out_guards)  # Guard 信息


def main():
    """主函数"""
    print("\n" + "="*60)
    print("  字节码到 FX Graph 转换演示")
    print("  理解 TorchDynamo 的符号化执行")
    print("="*60)
    
    # 运行所有演示
    demo1_simple()
    demo2_complex()
    demo4_detailed_walkthrough()
    demo3_real_torch_fx()
    demo5_compare_with_real()
    
    print("\n" + "="*60)
    print("  演示完成！")
    print("="*60)
    
    print("\n核心理解:")
    print("  1. 字节码是 Python 函数的底层指令表示")
    print("  2. 符号化执行 = 用符号变量代替真实值，记录操作")
    print("  3. 每条字节码指令被转换为 FX Graph 中的节点")
    print("  4. LOAD_FAST -> placeholder, BINARY_ADD -> call_function")
    print("  5. 最终生成的 FX Graph 可以被进一步优化和编译")
    print()


if __name__ == "__main__":
    main()

