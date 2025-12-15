"""
演示 Python 执行帧的结构
帮助理解 TorchDynamo 如何拦截函数调用
"""

import sys
import dis
import inspect
from types import FrameType


def print_separator(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def show_frame_info(frame: FrameType):
    """显示执行帧的详细信息"""
    code = frame.f_code
    
    print(f"\n[代码对象信息]")
    print(f"  函数名: {code.co_name}")
    print(f"  文件名: {code.co_filename}")
    print(f"  第一行: {code.co_firstlineno}")
    print(f"  参数个数: {code.co_argcount}")
    print(f"  参数名称: {code.co_varnames[:code.co_argcount]}")
    print(f"  所有变量: {code.co_varnames}")
    print(f"  常量: {code.co_consts}")
    
    print(f"\n[执行状态]")
    print(f"  当前行号: {frame.f_lineno}")
    print(f"  最后指令: {frame.f_lasti}")
    print(f"  局部变量: {frame.f_locals}")
    
    print(f"\n[字节码 (前 20 条)]")
    instructions = list(dis.get_instructions(code))[:20]
    for inst in instructions:
        print(f"  {inst.offset:4d} {inst.opname:20s} {inst.argval}")


def demo1_simple_function(x, y):
    """演示 1：简单的函数"""
    print_separator("演示 1：简单函数的执行帧")
    
    # 获取当前帧
    frame = sys._getframe()
    show_frame_info(frame)
    
    # 执行计算
    result = x + y
    return result


def demo2_with_locals(a, b):
    """演示 2：有局部变量的函数"""
    print_separator("演示 2：带局部变量的执行帧")
    
    # 创建一些局部变量
    temp1 = a * 2
    temp2 = b * 3
    result = temp1 + temp2
    
    # 获取帧
    frame = sys._getframe()
    show_frame_info(frame)
    
    return result


def demo3_nested_calls():
    """演示 3：嵌套调用和调用栈"""
    print_separator("演示 3：嵌套调用的调用栈")
    
    def inner1(x):
        def inner2(y):
            # 最内层
            frame = sys._getframe()
            print("\n[调用栈追踪]")
            
            depth = 0
            current = frame
            while current is not None:
                code = current.f_code
                print(f"  [{depth}] {code.co_name:20s} "
                      f"行号:{current.f_lineno:3d} "
                      f"局部变量: {list(current.f_locals.keys())}")
                current = current.f_back
                depth += 1
            
            return y * 2
        
        return inner2(x + 1)
    
    return inner1(10)


def demo4_bytecode_execution():
    """演示 4：字节码执行过程"""
    print_separator("演示 4：字节码指令详解")
    
    def target_function(a, b):
        """目标函数：将被分析字节码"""
        x = a + b      # LOAD_FAST, LOAD_FAST, BINARY_ADD, STORE_FAST
        y = x * 2      # LOAD_FAST, LOAD_CONST, BINARY_MULTIPLY, STORE_FAST
        return y       # LOAD_FAST, RETURN_VALUE
    
    print("\n[字节码详细分析]")
    dis.dis(target_function)
    
    print("\n[字节码执行模拟]")
    print("假设调用 target_function(3, 5):")
    print()
    print("  初始状态:")
    print("    局部变量: {}")
    print("    栈: []")
    print()
    print("  执行 LOAD_FAST 0 (a):")
    print("    局部变量: {'a': 3, 'b': 5}")
    print("    栈: [3]")
    print()
    print("  执行 LOAD_FAST 1 (b):")
    print("    栈: [3, 5]")
    print()
    print("  执行 BINARY_ADD:")
    print("    弹出 5 和 3，相加得 8")
    print("    栈: [8]")
    print()
    print("  执行 STORE_FAST 2 (x):")
    print("    局部变量: {'a': 3, 'b': 5, 'x': 8}")
    print("    栈: []")
    print()
    print("  ... 继续执行后续指令 ...")
    print()
    print("  最终 RETURN_VALUE:")
    print("    返回栈顶的 16")


def demo5_frame_hook_concept():
    """演示 5：拦截的概念"""
    print_separator("演示 5：TorchDynamo 拦截概念")
    
    print("\n[正常执行流程]")
    print("  1. Python 调用函数")
    print("  2. CPython 创建执行帧 (Frame)")
    print("  3. 调用 _PyEval_EvalFrameDefault(frame)")
    print("  4. 逐条执行字节码")
    print("  5. 返回结果")
    
    print("\n[TorchDynamo 拦截后的流程]")
    print("  1. Python 调用函数")
    print("  2. CPython 创建执行帧 (Frame)")
    print("  3. 调用 custom_eval_frame(frame)  <-- 被替换！")
    print("     |")
    print("     +-> 检查缓存")
    print("     |")
    print("     +-> 检查 Guard (形状、类型等)")
    print("     |")
    print("     +-> 如果需要：")
    print("         |")
    print("         +-> 分析字节码")
    print("         +-> 构建 FX Graph")
    print("         +-> 编译优化")
    print("         +-> 缓存结果")
    print("     |")
    print("     +-> 执行优化后的代码")
    print("  4. 返回结果")
    
    print("\n[关键点]")
    print("  • 在步骤 3，eval_frame 函数被替换")
    print("  • 每次函数调用都会经过 custom_eval_frame")
    print("  • custom_eval_frame 可以选择编译或直接执行")
    print("  • 用户代码完全不需要修改")


def demo6_torch_example():
    """演示 6：实际的 torch.compile 例子"""
    print_separator("演示 6：torch.compile 实际应用")
    
    try:
        import torch
        
        # 定义一个简单的函数
        def my_function(x, y):
            z = x + y
            return z * 2
        
        print("\n[原始函数字节码]")
        dis.dis(my_function)
        
        # 应用 torch.compile
        compiled_fn = torch.compile(my_function, backend="eager")
        
        print("\n[第一次调用 - 会被编译]")
        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        result = compiled_fn(x, y)
        print(f"  输入形状: {x.shape}")
        print(f"  结果形状: {result.shape}")
        print("  (此时 TorchDynamo 分析字节码并构建 FX Graph)")
        
        print("\n[第二次调用 - 使用缓存]")
        x2 = torch.randn(3, 4)
        y2 = torch.randn(3, 4)
        result2 = compiled_fn(x2, y2)
        print(f"  输入形状: {x2.shape}")
        print(f"  结果形状: {result2.shape}")
        print("  (Guard 检查通过，直接使用缓存的代码)")
        
        print("\n[第三次调用 - 不同形状]")
        x3 = torch.randn(5, 6)
        y3 = torch.randn(5, 6)
        result3 = compiled_fn(x3, y3)
        print(f"  输入形状: {x3.shape}")
        print(f"  结果形状: {result3.shape}")
        print("  (Guard 失败，需要重新编译)")
        
    except ImportError:
        print("\n  [跳过] PyTorch 未安装")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  Python 执行帧（Frame）演示")
    print("  理解 TorchDynamo 如何拦截函数调用")
    print("=" * 60)
    
    # 运行所有演示
    demo1_simple_function(10, 20)
    demo2_with_locals(3, 4)
    demo3_nested_calls()
    demo4_bytecode_execution()
    demo5_frame_hook_concept()
    demo6_torch_example()
    
    print("\n" + "=" * 60)
    print("  演示完成！")
    print("=" * 60)
    print("\n关键理解:")
    print("  1. Frame = 函数的运行环境（代码 + 变量 + 状态）")
    print("  2. CPython 用 eval_frame 函数执行帧")
    print("  3. PEP 523 允许替换 eval_frame")
    print("  4. TorchDynamo 用 custom_eval_frame 拦截并编译")
    print("  5. 拦截是透明的，用户无感知\n")


if __name__ == "__main__":
    main()

