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
import os
import sys
import time


# 现在导入 GCU 相关模块
import torch_gcu
import triton_gcu


def print_section(title):
    """打印分节标题"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def create_test_model(model_size="large"):
    """创建测试模型
    
    Args:
        model_size: "small", "medium", "large", "xlarge", "pointwise"
        
    注意：
        - Linear 层会使用 extern_kernels (cuBLAS)，不生成 Triton kernel
        - pointwise 操作（如 add, mul, relu, gelu）会生成 Triton kernel
        - 使用 "pointwise" 模式来强制生成 Triton kernel
    """
    
    if model_size == "pointwise":
        # 专门用于测试 Triton kernel 生成的模型
        # 只使用 pointwise 操作，不使用 Linear
        input_dim = 1024
        
        class PointwiseModel(nn.Module):
            """纯 pointwise 操作模型 - 确保生成 Triton kernel"""
            def __init__(self):
                super().__init__()
                self.scale1 = nn.Parameter(torch.randn(input_dim))
                self.bias1 = nn.Parameter(torch.randn(input_dim))
                self.scale2 = nn.Parameter(torch.randn(input_dim))
                self.bias2 = nn.Parameter(torch.randn(input_dim))
                self.scale3 = nn.Parameter(torch.randn(input_dim))
                self.bias3 = nn.Parameter(torch.randn(input_dim))
                
            def forward(self, x):
                # 多个 pointwise 操作 - 这些会生成 Triton kernel
                x = x * self.scale1 + self.bias1
                x = torch.relu(x)
                x = x * self.scale2 + self.bias2
                x = torch.sigmoid(x)
                x = x * self.scale3 + self.bias3
                x = torch.tanh(x)
                # 更多融合操作
                x = x * 2.0 + 1.0
                x = torch.relu(x)
                x = x.pow(2)
                x = torch.sqrt(x + 1e-6)
                return x
        
        return PointwiseModel(), input_dim
    
    elif model_size == "mixed":
        # 混合模型：Linear + 大量 pointwise 操作
        input_dim = 512
        
        class MixedModel(nn.Module):
            """混合模型 - Linear 用 extern, pointwise 用 Triton"""
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(input_dim, input_dim)
                self.linear2 = nn.Linear(input_dim, input_dim)
                self.scale = nn.Parameter(torch.randn(input_dim))
                self.bias = nn.Parameter(torch.randn(input_dim))
                
            def forward(self, x):
                # Linear (extern kernel)
                x = self.linear1(x)
                # Pointwise ops (Triton kernel) - 这些会被 fuse 成一个 kernel
                x = x * self.scale + self.bias
                x = torch.relu(x)
                x = x * 0.5 + 0.5
                x = torch.gelu(x)
                # Another Linear
                x = self.linear2(x)
                # More pointwise
                x = torch.sigmoid(x)
                x = x * 2.0 - 1.0
                return x
        
        return MixedModel(), input_dim
    
    # 原有的模型选项
    if model_size == "small":
        hidden_dims = [256, 256, 128]
        input_dim = 256
    elif model_size == "medium":
        hidden_dims = [512, 512, 512, 256]
        input_dim = 512
    elif model_size == "large":
        hidden_dims = [1024, 1024, 1024, 512, 256]
        input_dim = 1024
    else:  # xlarge
        hidden_dims = [2048, 2048, 1024, 1024, 512, 256]
        input_dim = 2048
    
    class LargeModel(nn.Module):
        """较大的测试模型，用于展示编译优化效果"""
        def __init__(self):
            super().__init__()
            
            # 构建多层网络
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(hidden_dim))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 10))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    return LargeModel(), input_dim



def test():
    """快速测试函数"""
    model, input_dim = create_test_model("large")
    print(f"[Model] Created LargeModel:")
    print(f"  - Input size: {input_dim}")
    print(f"  - Output size: 10")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 将模型移到 GCU
    try:
        model = model.to("gcu")
        print(f"[Model] ✓ Moved to GCU device")
        device = "gcu"
    except Exception as e:
        print(f"[Model] ⚠ Could not move to GCU, using CPU: {e}")
        device = "cpu"
    
    batch_size = 128  # 更大的 batch size
    input_data = torch.randn(batch_size, input_dim, device=device)
    print(f"[Input] Shape: {input_data.shape}, Device: {input_data.device}")
    
    # 编译模型
    print("\n[Compile] Compiling model with torch.compile...")
    compiled_model = torch.compile(model, backend="inductor")
    
    # 触发编译
    print("[Compile] Running first forward pass (triggers compilation)...")
    with torch.no_grad():
        output = compiled_model(input_data)

    print(f"[Compile] ✓ Compilation completed")
    print(f"[Output] Shape: {output.shape}, Device: {output.device}")


def benchmark_model(model, input_data, num_runs=100, warmup=50):
    """性能测试
    
    Args:
        model: 要测试的模型
        input_data: 输入数据
        num_runs: 测试次数
        warmup: 预热次数（对于编译模型，需要更多预热来确保编译完成）
    """
    # 预热 - 确保编译完成
    print(f"      Warming up ({warmup} iterations)...", end=" ", flush=True)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)
    
    # 同步
    try:
        torch.gcu.synchronize()
    except:
        pass
    print("done")
    
    # 多次测量取平均
    times = []
    with torch.no_grad():
        for run in range(3):  # 3 轮测量
            # 同步开始
            try:
                torch.gcu.synchronize()
            except:
                pass
            
            # 计时
            start = time.time()
            for _ in range(num_runs):
                _ = model(input_data)
            
            # 同步结束
            try:
                torch.gcu.synchronize()
            except:
                pass
            
            end = time.time()
            avg_time = (end - start) / num_runs * 1000  # ms
            times.append(avg_time)
            print(f"      Run {run+1}: {avg_time:.3f} ms/iter")
    
    # 返回最小值（最能代表真实性能）
    return min(times)


def main():
    """主函数"""
    
    # ========================================================================
    # 步骤 3: 创建模型
    # ========================================================================
    print_section("Step 3: Create Test Model")
    
    # 使用 pointwise 模型来确保生成 Triton kernel
    # 可选: "small", "medium", "large", "xlarge", "pointwise", "mixed"
    # - pointwise: 纯 pointwise 操作，100% 生成 Triton kernel
    # - mixed: Linear + pointwise，部分生成 Triton kernel
    # - large: 主要是 Linear，可能不生成 Triton kernel
    model_size = "pointwise"  # 强制使用 Triton kernel
    model, input_dim = create_test_model(model_size)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Created {model_size.upper()} Model:")
    print(f"  - Input size: {input_dim}")
    print(f"  - Output size: 10")
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Model size: {num_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 将模型移到 GCU（如果可用）
    try:
        model = model.to("gcu")
        print(f"[Model] ✓ Moved to GCU device")
        device = "gcu"
    except Exception as e:
        print(f"[Model] ⚠ Could not move to GCU, using CPU: {e}")
        device = "cpu"
    
    
    # 创建输入数据 - 使用更大的 batch size
    batch_size = 128  # 大 batch size 更能体现编译优化
    input_data = torch.randn(batch_size, input_dim, device=device)
    print(f"[Input] Shape: {input_data.shape}, Device: {input_data.device}")
    print(f"[Input] Total elements: {batch_size * input_dim:,}")
    
    # 编译模型（在捕获上下文中）
    print("\n[Compile] Compiling model with torch.compile...")
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
    
    
    # ========================================================================
    # 步骤 6: 性能测试
    # ========================================================================
    print_section("Step 6: Performance Benchmark")
    
    # 先确保编译完全完成（额外预热）
    print("[Benchmark] Pre-warming compiled model to ensure compilation is complete...")
    with torch.no_grad():
        for _ in range(2):
            _ = compiled_model(input_data)
    try:
        torch.gcu.synchronize()
    except:
        pass
    print("[Benchmark] ✓ Pre-warming done\n")
    
    print("[Benchmark] Testing eager mode...")
    eager_time = benchmark_model(model, input_data, num_runs=100, warmup=20)
    print(f"[Benchmark] → Eager mode: {eager_time:.3f} ms/iteration")
    
    print("\n[Benchmark] Testing compiled mode...")
    compiled_time = benchmark_model(compiled_model, input_data, num_runs=100, warmup=20)
    print(f"[Benchmark] → Compiled mode: {compiled_time:.3f} ms/iteration")
    
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
    
    return 0


if __name__ == "__main__":
    # test()
    main()

