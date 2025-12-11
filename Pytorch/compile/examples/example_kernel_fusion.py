#!/usr/bin/env python3
"""
第九章示例代码：Pointwise 操作与 Kernel Fusion
演示 TorchInductor 如何融合 Pointwise 操作以提升性能
"""

import os
import time
import torch
import torch.nn as nn

# 设置环境变量以查看生成的代码（可选，取消注释以启用）
# os.environ['TORCH_LOGS'] = 'output_code'


# =============================================================================
# 模型定义
# =============================================================================

class PointwiseModel(nn.Module):
    """
    纯 Pointwise 操作模型
    
    所有操作都是逐元素的，会被 TorchInductor 融合成一个 Triton Kernel：
    triton_poi_fused_add_mul_relu_sigmoid_tanh_...
    
    预期性能提升：3-5x
    """
    
    def __init__(self, dim=1024):
        super().__init__()
        self.scale1 = nn.Parameter(torch.randn(dim))
        self.bias1 = nn.Parameter(torch.randn(dim))
        self.scale2 = nn.Parameter(torch.randn(dim))
        self.bias2 = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        # 以下所有操作会被融合成 1 个 Triton Kernel
        x = x * self.scale1 + self.bias1   # mul + add
        x = torch.relu(x)                   # relu
        x = x * self.scale2 + self.bias2   # mul + add
        x = torch.sigmoid(x)                # sigmoid
        x = x * 2.0 + 1.0                  # mul + add
        x = torch.tanh(x)                   # tanh
        return x


class MixedModel(nn.Module):
    """
    混合模型：Linear + Pointwise
    
    Linear 层会调用 cuBLAS（extern kernel），不生成 Triton Kernel
    Pointwise 操作会被融合成 Triton Kernel
    
    生成的 Kernel 结构：
    1. triton_poi_fused_... (第一组 Pointwise)
    2. aten.mm / aten.addmm (Linear，调用 cuBLAS)
    3. triton_poi_fused_... (第二组 Pointwise)
    
    预期性能提升：1.5-2x
    """
    
    def __init__(self, dim=512):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.randn(dim))
        self.bias = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        # Linear (extern kernel - cuBLAS)
        x = self.linear1(x)
        
        # Pointwise 组 1 (Triton kernel - 会被融合)
        x = x * self.scale + self.bias
        x = torch.relu(x)
        x = x * 0.5 + 0.5
        x = torch.gelu(x)
        
        # Linear (extern kernel - cuBLAS)
        x = self.linear2(x)
        
        # Pointwise 组 2 (Triton kernel)
        x = torch.sigmoid(x)
        
        return x


class ComplexPointwiseModel(nn.Module):
    """
    复杂的 Pointwise 模型
    
    包含更多种类的 Pointwise 操作，展示 Triton 的融合能力
    """
    
    def __init__(self, dim=1024):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(dim))
        self.bias = nn.Parameter(torch.randn(dim))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        # 第一组操作
        x = x * self.scale + self.bias
        x = torch.relu(x)
        
        # 第二组操作 - 更多数学函数
        x = x.pow(2)              # 平方
        x = torch.sqrt(x + 1e-6)  # 开方（加 epsilon 避免 0）
        
        # 第三组操作 - 归一化风格
        x = x * self.gamma + self.beta
        
        # 第四组操作 - 激活函数
        x = torch.silu(x)  # SiLU = x * sigmoid(x)
        
        # 第五组操作 - clamp
        x = x.clamp(min=-10, max=10)
        
        return x


# =============================================================================
# 工具函数
# =============================================================================

def benchmark(fn, x, warmup=50, runs=100, sync_fn=None):
    """
    性能测试函数
    
    Args:
        fn: 要测试的函数
        x: 输入数据
        warmup: 预热次数
        runs: 测试次数
        sync_fn: 同步函数（CUDA 需要同步）
    
    Returns:
        平均执行时间（毫秒）
    """
    if sync_fn is None:
        if x.is_cuda:
            sync_fn = torch.cuda.synchronize
        else:
            sync_fn = lambda: None
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn(x)
        sync_fn()
    
    # 测量多轮取最小值
    times = []
    with torch.no_grad():
        for _ in range(3):
            sync_fn()
            start = time.time()
            for _ in range(runs):
                _ = fn(x)
            sync_fn()
            elapsed = (time.time() - start) / runs * 1000
            times.append(elapsed)
    
    return min(times)


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


# =============================================================================
# 主测试
# =============================================================================

def test_pointwise_fusion():
    """测试纯 Pointwise 模型的融合效果"""
    print_section("Test 1: Pure Pointwise Model")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = PointwiseModel(1024).to(device).eval()
    compiled_model = torch.compile(model, backend="inductor")
    
    x = torch.randn(256, 1024, device=device)
    
    eager_time = benchmark(model, x)
    compiled_time = benchmark(compiled_model, x)
    speedup = eager_time / compiled_time
    
    print(f"Model: PointwiseModel (dim=1024)")
    print(f"Input: (256, 1024)")
    print(f"Device: {device}")
    print(f"")
    print(f"Eager mode:    {eager_time:.3f} ms")
    print(f"Compiled mode: {compiled_time:.3f} ms")
    print(f"Speedup:       {speedup:.2f}x")
    
    # 验证正确性
    with torch.no_grad():
        y_eager = model(x)
        y_compiled = compiled_model(x)
        max_diff = (y_eager - y_compiled).abs().max().item()
    
    print(f"")
    print(f"Max difference: {max_diff:.2e}")
    print(f"✓ Results match" if max_diff < 1e-5 else "✗ Results differ")
    
    return speedup


def test_mixed_model():
    """测试混合模型（Linear + Pointwise）"""
    print_section("Test 2: Mixed Model (Linear + Pointwise)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MixedModel(512).to(device).eval()
    compiled_model = torch.compile(model, backend="inductor")
    
    x = torch.randn(256, 512, device=device)
    
    eager_time = benchmark(model, x)
    compiled_time = benchmark(compiled_model, x)
    speedup = eager_time / compiled_time
    
    print(f"Model: MixedModel (dim=512)")
    print(f"Input: (256, 512)")
    print(f"Device: {device}")
    print(f"")
    print(f"Eager mode:    {eager_time:.3f} ms")
    print(f"Compiled mode: {compiled_time:.3f} ms")
    print(f"Speedup:       {speedup:.2f}x")
    
    # 验证正确性
    with torch.no_grad():
        y_eager = model(x)
        y_compiled = compiled_model(x)
        max_diff = (y_eager - y_compiled).abs().max().item()
    
    print(f"")
    print(f"Max difference: {max_diff:.2e}")
    print(f"✓ Results match" if max_diff < 1e-5 else "✗ Results differ")
    
    return speedup


def test_complex_pointwise():
    """测试复杂 Pointwise 模型"""
    print_section("Test 3: Complex Pointwise Model")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ComplexPointwiseModel(1024).to(device).eval()
    compiled_model = torch.compile(model, backend="inductor")
    
    x = torch.randn(512, 1024, device=device)
    
    eager_time = benchmark(model, x)
    compiled_time = benchmark(compiled_model, x)
    speedup = eager_time / compiled_time
    
    print(f"Model: ComplexPointwiseModel (dim=1024)")
    print(f"Input: (512, 1024)")
    print(f"Device: {device}")
    print(f"")
    print(f"Eager mode:    {eager_time:.3f} ms")
    print(f"Compiled mode: {compiled_time:.3f} ms")
    print(f"Speedup:       {speedup:.2f}x")
    
    # 验证正确性
    with torch.no_grad():
        y_eager = model(x)
        y_compiled = compiled_model(x)
        max_diff = (y_eager - y_compiled).abs().max().item()
    
    print(f"")
    print(f"Max difference: {max_diff:.2e}")
    print(f"✓ Results match" if max_diff < 1e-5 else "✗ Results differ")
    
    return speedup


def test_batch_size_effect():
    """测试 batch size 对优化效果的影响"""
    print_section("Test 4: Batch Size Effect on Speedup")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = PointwiseModel(1024).to(device).eval()
    compiled_model = torch.compile(model, backend="inductor")
    
    batch_sizes = [1, 8, 32, 128, 512, 2048]
    
    print(f"Model: PointwiseModel (dim=1024)")
    print(f"Device: {device}")
    print(f"")
    print(f"{'Batch Size':<12} {'Eager (ms)':<12} {'Compiled (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for bs in batch_sizes:
        x = torch.randn(bs, 1024, device=device)
        
        try:
            eager_time = benchmark(model, x, warmup=20, runs=50)
            compiled_time = benchmark(compiled_model, x, warmup=20, runs=50)
            speedup = eager_time / compiled_time
            print(f"{bs:<12} {eager_time:<12.3f} {compiled_time:<15.3f} {speedup:<10.2f}x")
        except Exception as e:
            print(f"{bs:<12} Error: {e}")


# =============================================================================
# 入口
# =============================================================================

def main():
    print("=" * 60)
    print(" Pointwise 操作与 Kernel Fusion 示例")
    print(" 第九章：理解 TorchInductor 的优化策略")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # 运行测试
    speedup1 = test_pointwise_fusion()
    speedup2 = test_mixed_model()
    speedup3 = test_complex_pointwise()
    test_batch_size_effect()
    
    # 总结
    print_section("Summary")
    print(f"Pointwise Model Speedup:         {speedup1:.2f}x")
    print(f"Mixed Model Speedup:             {speedup2:.2f}x")
    print(f"Complex Pointwise Model Speedup: {speedup3:.2f}x")
    print(f"")
    print("Key Observations:")
    print("  1. Pure Pointwise models benefit most from Kernel Fusion")
    print("  2. Linear layers use cuBLAS (extern kernels), limiting fusion")
    print("  3. Larger batch sizes generally show better speedup")
    print("  4. Small models may not benefit due to compilation overhead")


if __name__ == "__main__":
    main()

