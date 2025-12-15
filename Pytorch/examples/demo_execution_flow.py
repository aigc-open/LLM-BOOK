"""
演示：torch.compile 的完整执行流程
包括装饰器阶段、Hook 安装、拦截、编译、执行的每一步
"""

import torch
import sys
from functools import wraps


# 全局变量用于追踪执行流程
execution_log = []
hook_installed = False


def log_step(step_name, location, description):
    """记录执行步骤"""
    execution_log.append({
        'step': step_name,
        'location': location,
        'description': description
    })
    print(f"\n{'='*60}")
    print(f"[{step_name}] {location}")
    print(f"{description}")
    print('='*60)


def print_execution_summary():
    """打印执行摘要"""
    print("\n\n" + "="*60)
    print("执行流程摘要")
    print("="*60)
    for i, log in enumerate(execution_log, 1):
        print(f"{i}. [{log['step']}] {log['location']}")
        print(f"   {log['description']}")


# ============ 模拟 TorchDynamo 的组件 ============

class MockEvalFrameHook:
    """模拟 eval_frame Hook"""
    
    def __init__(self):
        self.is_installed = False
        self.callback = None
        self.cache = {}
    
    def set_eval_frame(self, callback):
        """安装 Hook"""
        log_step(
            "安装 Hook",
            "Python 层: set_eval_frame()",
            "在 CPython 解释器中替换 eval_frame 函数"
        )
        
        self.callback = callback
        self.is_installed = True
        global hook_installed
        hook_installed = True
    
    def custom_eval_frame(self, fn, args, kwargs):
        """模拟 C 层的 custom_eval_frame"""
        log_step(
            "拦截函数调用",
            "C 层: custom_eval_frame()",
            f"CPython 准备执行 {fn.__name__}，被 Hook 拦截"
        )
        
        # 检查缓存
        cache_key = id(fn.__code__)
        
        if cache_key in self.cache:
            log_step(
                "缓存命中",
                "C 层: custom_eval_frame()",
                "在缓存中找到编译结果，检查 Guard..."
            )
            
            # 模拟 Guard 检查
            cached_entry = self.cache[cache_key]
            if self._check_guards(cached_entry['guards'], args):
                log_step(
                    "Guard 通过",
                    "C 层: custom_eval_frame()",
                    "Guard 检查通过，直接执行缓存的优化代码"
                )
                
                # 执行缓存的代码
                result = cached_entry['optimized_fn'](*args, **kwargs)
                
                log_step(
                    "返回结果",
                    "C 层 -> Python 层 -> 用户",
                    f"执行完成，返回结果（使用缓存，极快！）"
                )
                
                return result
            else:
                log_step(
                    "Guard 失败",
                    "C 层: custom_eval_frame()",
                    "Guard 检查失败，需要重新编译"
                )
        else:
            log_step(
                "缓存未命中",
                "C 层: custom_eval_frame()",
                "第一次调用，缓存中没有编译结果"
            )
        
        # 调用 Python 层编译
        log_step(
            "调用 Python 编译",
            "C 层 -> Python 层",
            "调用 py_callback(frame) 进行编译"
        )
        
        guarded_code = self._compile_frame(fn, args, kwargs)
        
        log_step(
            "编译完成",
            "Python 层 -> C 层",
            "返回 GuardedCode（包含优化代码和 Guard）"
        )
        
        # 缓存
        self.cache[cache_key] = guarded_code
        
        log_step(
            "缓存结果",
            "C 层: custom_eval_frame()",
            "将编译结果存入缓存"
        )
        
        # 执行优化代码
        log_step(
            "执行优化代码",
            "C 层: execute_optimized_code()",
            "执行编译后的优化代码"
        )
        
        result = guarded_code['optimized_fn'](*args, **kwargs)
        
        log_step(
            "返回结果",
            "C 层 -> Python 层 -> 用户",
            "执行完成，返回结果"
        )
        
        return result
    
    def _compile_frame(self, fn, args, kwargs):
        """模拟 Python 层的 convert_frame_assert"""
        log_step(
            "开始编译",
            "Python 层: convert_frame_assert()",
            "接收到 C 层的编译请求"
        )
        
        # 步骤 1: 获取字节码
        log_step(
            "获取字节码",
            "Python 层: InstructionTranslator",
            f"分析 {fn.__name__} 的字节码指令"
        )
        
        import dis
        print("\n字节码:")
        dis.dis(fn)
        
        # 步骤 2: 符号化执行
        log_step(
            "符号化执行",
            "Python 层: InstructionTranslator.run()",
            "逐条处理字节码，构建 FX Graph"
        )
        
        print("\n模拟符号化执行:")
        print("  - LOAD_FAST x -> 创建 placeholder x")
        print("  - LOAD_FAST y -> 创建 placeholder y")
        print("  - BINARY_ADD -> 创建 call_function torch.add(x, y)")
        print("  - RETURN_VALUE -> 设置 output")
        
        # 步骤 3: 生成 FX Graph
        log_step(
            "生成 FX Graph",
            "Python 层: OutputGraph",
            "完成 FX Graph 构建"
        )
        
        print("\n生成的 FX Graph:")
        print("  graph():")
        print("    %x = placeholder[target=x]")
        print("    %y = placeholder[target=y]")
        print("    %add = call_function[torch.add](%x, %y)")
        print("    return add")
        
        # 步骤 4: 后端编译
        log_step(
            "后端编译",
            "Python 层: Inductor",
            "优化 FX Graph，生成 CUDA kernel"
        )
        
        print("\n编译优化:")
        print("  - 算子融合")
        print("  - 内存优化")
        print("  - 生成 Triton/C++ 代码")
        
        # 步骤 5: 生成 Guard
        log_step(
            "生成 Guard",
            "Python 层: GuardBuilder",
            "记录执行条件（形状、类型等）"
        )
        
        guards = self._generate_guards(args)
        
        print("\n生成的 Guard:")
        for guard in guards:
            print(f"  - {guard}")
        
        # 返回 GuardedCode
        def optimized_fn(*args, **kwargs):
            """优化后的函数"""
            return fn(*args, **kwargs)
        
        return {
            'optimized_fn': optimized_fn,
            'guards': guards,
        }
    
    def _generate_guards(self, args):
        """生成 Guard"""
        guards = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                guards.append(f"arg{i}.shape == {tuple(arg.shape)}")
                guards.append(f"arg{i}.dtype == {arg.dtype}")
        return guards
    
    def _check_guards(self, guards, args):
        """检查 Guard（简化版）"""
        # 这里简化为总是返回 True
        # 实际实现会检查每个 Guard
        return True


# 全局 Hook 实例
mock_hook = MockEvalFrameHook()


def mock_compile(fn):
    """模拟 torch.compile 装饰器"""
    
    log_step(
        "装饰器执行",
        "装饰器阶段: @torch.compile",
        f"包装函数 {fn.__name__}，返回 wrapper"
    )
    
    print(f"\n注意:")
    print(f"  - 此时还没有编译")
    print(f"  - 还没有生成 FX Graph")
    print(f"  - 还没有安装 Hook")
    print(f"  - 只是返回了一个 wrapper 函数")
    
    @wraps(fn)
    def wrapper(*args, **kwargs):
        """包装函数"""
        log_step(
            "进入 wrapper",
            "Python 层: wrapper()",
            "用户调用函数，进入 wrapper"
        )
        
        # 安装 Hook
        mock_hook.set_eval_frame(lambda frame: None)
        
        try:
            log_step(
                "调用原始函数",
                "Python 层: wrapper() -> fn()",
                f"wrapper 调用原始函数 {fn.__name__}"
            )
            
            # 模拟 CPython 的拦截
            if mock_hook.is_installed:
                result = mock_hook.custom_eval_frame(fn, args, kwargs)
            else:
                result = fn(*args, **kwargs)
            
            return result
        finally:
            # 恢复 Hook
            pass
    
    return wrapper


# ============ 演示程序 ============

def demo1_first_call():
    """演示 1: 第一次调用（触发编译）"""
    global execution_log, hook_installed
    execution_log = []
    hook_installed = False
    
    print("\n" + "="*60)
    print("演示 1: 第一次调用（触发编译）")
    print("="*60)
    
    # 阶段 1: 装饰器
    print("\n【阶段 1: 装饰器阶段】")
    print("-"*60)
    
    @mock_compile
    def my_function(x, y):
        return x + y
    
    print(f"\n装饰器已执行，my_function 现在是: {type(my_function)}")
    
    # 阶段 2: 第一次调用
    print("\n\n【阶段 2: 第一次调用】")
    print("-"*60)
    print("执行: result = my_function(x, y)")
    
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    result = my_function(x, y)
    
    print(f"\n结果: {result.shape}")
    
    # 打印摘要
    print_execution_summary()


def demo2_second_call():
    """演示 2: 第二次调用（使用缓存）"""
    global execution_log
    
    print("\n\n" + "="*60)
    print("演示 2: 第二次调用（使用缓存）")
    print("="*60)
    
    # 先进行第一次调用
    execution_log = []
    
    @mock_compile
    def my_function(x, y):
        return x + y
    
    # 第一次调用（静默）
    x1 = torch.randn(3, 4)
    y1 = torch.randn(3, 4)
    execution_log = []  # 清空日志
    
    print("\n第一次调用完成，现在进行第二次调用...")
    print("-"*60)
    print("执行: result = my_function(x, y)  # 相同形状")
    
    # 第二次调用
    x2 = torch.randn(3, 4)
    y2 = torch.randn(3, 4)
    result = my_function(x2, y2)
    
    print(f"\n结果: {result.shape}")
    
    # 打印摘要
    print_execution_summary()
    
    print("\n对比第一次和第二次:")
    print("  第一次: 经历完整的编译过程（慢）")
    print("  第二次: 直接使用缓存（极快！）")


def demo3_key_points():
    """演示 3: 关键点总结"""
    print("\n\n" + "="*60)
    print("关键点总结")
    print("="*60)
    
    print("\n1. 装饰器 vs 编译")
    print("-"*60)
    print("  装饰器执行时机: 函数定义时")
    print("  编译执行时机: 第一次调用时")
    print("  装饰器做的事: 包装函数，返回 wrapper")
    print("  编译做的事: 字节码 -> FX Graph -> 优化代码")
    
    print("\n2. Hook 安装")
    print("-"*60)
    print("  安装时机: wrapper() 执行时")
    print("  安装位置: set_eval_frame(callback)")
    print("  作用: 替换 CPython 的 eval_frame 函数")
    
    print("\n3. 拦截发生")
    print("-"*60)
    print("  触发时机: 调用原始函数时")
    print("  触发位置: custom_eval_frame() [C 层]")
    print("  作用: 检查缓存，决定编译或执行")
    
    print("\n4. 跨语言调用")
    print("-"*60)
    print("  C -> Python: py_callback(frame)")
    print("  Python 做: 编译（字节码分析、FX Graph 构建）")
    print("  Python -> C: 返回 GuardedCode")
    print("  C 做: 缓存、执行优化代码")
    
    print("\n5. 缓存机制")
    print("-"*60)
    print("  第一次: 缓存未命中 -> 编译 -> 缓存 -> 执行")
    print("  第二次: 缓存命中 -> Guard 检查 -> 直接执行")
    print("  Guard 失败: 重新编译（例如不同形状）")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("torch.compile 完整执行流程演示")
    print("="*60)
    
    # 运行演示
    demo1_first_call()
    demo2_second_call()
    demo3_key_points()
    
    print("\n\n" + "="*60)
    print("演示完成！")
    print("="*60)
    
    print("\n核心理解:")
    print("  1. 装饰器只是包装，不是编译")
    print("  2. Hook 在 wrapper 中安装")
    print("  3. 编译在第一次调用时发生")
    print("  4. C 层拦截，Python 层编译")
    print("  5. 第二次调用使用缓存（极快）")


if __name__ == "__main__":
    main()

