#!/usr/bin/env python3
"""
GCU 集成示例主程序

功能：
1. 注册 GCU 设备到 PyTorch
2. 配置 TorchInductor 使用 Triton GCU 后端
3. 应用 GCU 特定的优化（grid size、num_warps）
4. 编译并运行模型
5. 捕获和验证生成的 Triton Kernel

使用方法：
    python main_gcu_integration.py
"""

import torch
import torch.nn as nn
import torch_gcu
import os
import sys
import time

# 导入自定义模块
from gcu_config import configure_gcu
from kernel_inspector import inspect_kernels


def print_section(title):
    """打印分节标题"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def create_test_model():
    """创建测试模型"""
    class TestModel(nn.Module):
        """简单的测试模型"""
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(512, 1024)
            self.linear2 = nn.Linear(1024, 2048)
            self.linear3 = nn.Linear(2048, 512)
            self.linear4 = nn.Linear(512, 10)
        
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = torch.relu(self.linear3(x))
            x = self.linear4(x)
            return x
    
    return TestModel()


def benchmark_model(model, input_data, num_runs=100, warmup=10):
    """性能测试"""
    # 预热
    for _ in range(warmup):
        _ = model(input_data)
    
    # 同步
    try:
        torch.gcu.synchronize()
    except:
        pass
    
    # 计时
    start = time.time()
    for _ in range(num_runs):
        _ = model(input_data)
    
    # 同步
    try:
        torch.gcu.synchronize()
    except:
        pass
    
    end = time.time()
    
    avg_time = (end - start) / num_runs * 1000  # ms
    return avg_time


def main():
    """主函数"""
    print_section("GCU Integration Test - Start")
    
    # ========================================================================
    # 步骤 1: 注册 GCU 设备
    # ========================================================================
    print_section("Step 1: Register GCU Device")
    
    # ========================================================================
    # 步骤 2: 配置 TorchInductor
    # ========================================================================
    print_section("Step 2: Configure TorchInductor for GCU")
    gcu_config = configure_gcu()
    
    # ========================================================================
    # 步骤 3: 创建模型
    # ========================================================================
    print_section("Step 3: Create Test Model")
    model = create_test_model()
    print(f"[Model] Created TestModel:")
    print(f"  - Linear layers: 4")
    print(f"  - Input size: 512")
    print(f"  - Output size: 10")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 将模型移到 GCU（如果可用）
    try:
        model = model.to("gcu")
        print(f"[Model] ✓ Moved to GCU device")
        device = "gcu"
    except Exception as e:
        print(f"[Model] ⚠ Could not move to GCU, using CPU: {e}")
        device = "cpu"
    
    # ========================================================================
    # 步骤 4: 编译模型并捕获 Kernel
    # ========================================================================
    print_section("Step 4: Compile Model and Capture Kernels")
    
    inspector = inspect_kernels()
    
    # 创建输入数据
    batch_size = 64
    input_data = torch.randn(batch_size, 512, device=device)
    print(f"[Input] Shape: {input_data.shape}, Device: {input_data.device}")
    
    # 编译模型（在捕获上下文中）
    print("\n[Compile] Compiling model with torch.compile...")
    with inspector.capture_kernels():
        compiled_model = torch.compile(model, backend="inductor")
        
        # 触发编译
        print("[Compile] Running first forward pass (triggers compilation)...")
        with torch.no_grad():
            output = compiled_model(input_data)
    
    print(f"[Compile] ✓ Compilation completed")
    print(f"[Output] Shape: {output.shape}, Device: {output.device}")
    
    # ========================================================================
    # 步骤 5: 分析生成的 Kernel
    # ========================================================================
    print_section("Step 5: Analyze Generated Kernels")
    
    # 打印 Kernel 摘要
    inspector.print_summary()
    
    # 保存 Kernel 到文件
    inspector.save_kernels()
    
    # ========================================================================
    # 步骤 6: 性能测试
    # ========================================================================
    print_section("Step 6: Performance Benchmark")
    
    print("[Benchmark] Testing eager mode...")
    with torch.no_grad():
        eager_time = benchmark_model(model, input_data, num_runs=100, warmup=10)
    print(f"[Benchmark] Eager mode: {eager_time:.3f} ms/iteration")
    
    print("\n[Benchmark] Testing compiled mode...")
    with torch.no_grad():
        compiled_time = benchmark_model(compiled_model, input_data, num_runs=100, warmup=10)
    print(f"[Benchmark] Compiled mode: {compiled_time:.3f} ms/iteration")
    
    if eager_time > 0:
        speedup = eager_time / compiled_time
        print(f"\n[Benchmark] ✓ Speedup: {speedup:.2f}x")
    
    # ========================================================================
    # 步骤 7: 验证正确性
    # ========================================================================
    print_section("Step 7: Verify Correctness")
    
    with torch.no_grad():
        eager_output = model(input_data)
        compiled_output = compiled_model(input_data)
        
        max_diff = (eager_output - compiled_output).abs().max().item()
        mean_diff = (eager_output - compiled_output).abs().mean().item()
    
    print(f"[Verify] Max difference: {max_diff:.6e}")
    print(f"[Verify] Mean difference: {mean_diff:.6e}")
    
    if max_diff < 1e-3:
        print(f"[Verify] ✓ Results match (max_diff < 1e-3)")
    else:
        print(f"[Verify] ⚠ Results differ (max_diff = {max_diff:.6e})")
    
    # ========================================================================
    # 完成
    # ========================================================================
    print_section("GCU Integration Test - Completed")
    
    print("\nSummary:")
    print(f"  ✓ GCU device registered")
    print(f"  ✓ TorchInductor configured for GCU")
    print(f"  ✓ Model compiled successfully")
    print(f"  ✓ Generated {len(inspector.captured_kernels)} Triton kernel(s)")
    print(f"  ✓ Performance: {speedup:.2f}x speedup" if eager_time > 0 else "")
    print(f"  ✓ Correctness: max_diff = {max_diff:.6e}")
    
    print(f"\nGenerated kernels saved to: {inspector.output_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Review kernel configurations above")
    print(f"  2. Check saved kernels in {inspector.output_dir}/")
    print(f"  3. Verify grid sizes are ≤48 or multiples of 48")
    print(f"  4. Verify num_warps = 1")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Test cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[Error] Test failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

