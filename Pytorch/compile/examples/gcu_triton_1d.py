#!/usr/bin/env python3
"""
GCU Triton 1D 算子测试

使用 pytest 测试不同类型的 1D Pointwise 操作：
- 简单 Pointwise: 基础的逐元素操作
- 复杂 Pointwise: 多层融合操作
- Mixed: Linear + Pointwise 混合

使用方法：
    # 运行所有测试
    pytest gcu_triton_1d.py -v
    
    # 只运行 pointwise 测试
    pytest gcu_triton_1d.py -v -k "pointwise"
    
    # 只运行 mixed 测试
    pytest gcu_triton_1d.py -v -k "mixed"
    
    # 运行并显示性能数据
    pytest gcu_triton_1d.py -v -s
    
    # 运行特定模型
    pytest gcu_triton_1d.py -v -k "SimplePointwise"
"""

# =============================================================================
# 应用 GCU 补丁（必须在 torch.compile 之前）
# =============================================================================
import os
os.environ['TRITON_NUM_WARPS'] = '4'
os.environ['TRITON_MAX_BLOCK_SIZE'] = '256'

import gcu_patches

# =============================================================================
# 标准导入
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import time
from typing import Tuple
from dataclasses import dataclass


# =============================================================================
# 测试配置
# =============================================================================

@dataclass
class TestConfig:
    """测试配置"""
    batch_size: int = 128
    input_dim: int = 1024
    warmup: int = 20
    runs: int = 50
    rtol: float = 1e-4
    atol: float = 1e-4


CONFIG = TestConfig()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def device():
    """检测并返回可用设备"""
    try:
        import torch_gcu
        return "gcu"
    except ImportError:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


@pytest.fixture(scope="module")
def config():
    """返回测试配置"""
    return CONFIG


# =============================================================================
# 1D Pointwise 模型
# =============================================================================

class SimplePointwise(nn.Module):
    """简单的 Pointwise 操作"""
    
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(dim))
        self.bias = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        x = x * self.scale + self.bias
        x = torch.relu(x)
        return x


class ComplexPointwise(nn.Module):
    """复杂的 Pointwise 操作（多层融合）"""
    
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.scale1 = nn.Parameter(torch.randn(dim))
        self.bias1 = nn.Parameter(torch.randn(dim))
        self.scale2 = nn.Parameter(torch.randn(dim))
        self.bias2 = nn.Parameter(torch.randn(dim))
        self.scale3 = nn.Parameter(torch.randn(dim))
        self.bias3 = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        # 多个 pointwise 操作 - 会被融合成一个 Triton kernel
        x = x * self.scale1 + self.bias1
        x = torch.relu(x)
        x = x * self.scale2 + self.bias2
        x = torch.sigmoid(x)
        x = x * self.scale3 + self.bias3
        x = torch.tanh(x)
        # 更多操作
        x = x * 2.0 + 1.0
        x = torch.relu(x)
        x = x.pow(2)
        x = torch.sqrt(x + 1e-6)
        return x


class GatedPointwise(nn.Module):
    """门控 Pointwise 操作（GLU 风格）"""
    
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.gate_scale = nn.Parameter(torch.ones(dim))
        self.gate_bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        # 主路径
        main = x * self.scale + self.bias
        main = F.gelu(main)
        
        # 门控路径
        gate = torch.sigmoid(x * self.gate_scale + self.gate_bias)
        
        # 门控输出
        return main * gate


class SiLUPointwise(nn.Module):
    """SiLU (Swish) Pointwise 操作"""
    
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(dim))
        self.bias = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        x = x * self.scale + self.bias
        x = F.silu(x)  # x * sigmoid(x)
        x = x * 2.0 - 1.0
        return x


# =============================================================================
# Mixed 模型 (Linear + Pointwise)
# =============================================================================

class MixedModel(nn.Module):
    """混合模型 - Linear 用 extern kernel, Pointwise 用 Triton"""
    
    def __init__(self, dim: int = 512):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.randn(dim))
        self.bias = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        # Linear (extern kernel - cuBLAS)
        x = self.linear1(x)
        # Pointwise ops (Triton kernel)
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


class LargeLinearModel(nn.Module):
    """较大的 Linear 模型（用于对比）"""
    
    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# 测试工具函数
# =============================================================================

def sync_device(device: str):
    """同步设备"""
    if device == "gcu":
        torch.gcu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def benchmark_model(model, inputs, device: str, warmup: int = 20, runs: int = 50) -> Tuple[float, float]:
    """
    性能测试
    
    Returns:
        (eager_time_ms, compiled_time_ms)
    """
    compiled_model = torch.compile(model, backend="inductor")
    
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
            _ = compiled_model(*inputs)
    sync_device(device)
    
    # 测试 eager
    times_eager = []
    with torch.no_grad():
        for _ in range(3):
            sync_device(device)
            start = time.time()
            for _ in range(runs):
                _ = model(*inputs)
            sync_device(device)
            times_eager.append((time.time() - start) / runs * 1000)
    
    # 测试 compiled
    times_compiled = []
    with torch.no_grad():
        for _ in range(3):
            sync_device(device)
            start = time.time()
            for _ in range(runs):
                _ = compiled_model(*inputs)
            sync_device(device)
            times_compiled.append((time.time() - start) / runs * 1000)
    
    return min(times_eager), min(times_compiled)


def verify_correctness(model, inputs, device: str, rtol: float = 1e-4, atol: float = 1e-4) -> Tuple[bool, float]:
    """验证正确性"""
    compiled_model = torch.compile(model, backend="inductor")
    
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    
    with torch.no_grad():
        eager_out = model(*inputs)
        compiled_out = compiled_model(*inputs)
        
        max_diff = (eager_out - compiled_out).abs().max().item()
        is_close = torch.allclose(eager_out, compiled_out, rtol=rtol, atol=atol)
    
    return is_close, max_diff


# =============================================================================
# Pytest 测试类
# =============================================================================

class TestSimplePointwise:
    """简单 Pointwise 操作测试"""
    
    @pytest.mark.parametrize("model_class,name", [
        (SimplePointwise, "SimplePointwise"),
        (SiLUPointwise, "SiLUPointwise"),
    ])
    def test_simple_correctness(self, device, config, model_class, name):
        """测试简单 Pointwise 模型正确性"""
        model = model_class(config.input_dim).to(device).eval()
        x = torch.randn(config.batch_size, config.input_dim, device=device)
        
        is_correct, max_diff = verify_correctness(model, x, device, config.rtol, config.atol)
        
        print(f"\n[{name}] max_diff={max_diff:.2e}")
        assert is_correct, f"{name} correctness failed: max_diff={max_diff:.2e}"
    
    @pytest.mark.parametrize("model_class,name", [
        (SimplePointwise, "SimplePointwise"),
        (SiLUPointwise, "SiLUPointwise"),
    ])
    def test_simple_performance(self, device, config, model_class, name):
        """测试简单 Pointwise 模型性能"""
        model = model_class(config.input_dim).to(device).eval()
        x = torch.randn(config.batch_size, config.input_dim, device=device)
        
        eager_time, compiled_time = benchmark_model(model, x, device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        print(f"\n[{name}] eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms, speedup={speedup:.2f}x")


class TestComplexPointwise:
    """复杂 Pointwise 操作测试"""
    
    @pytest.mark.parametrize("model_class,name", [
        (ComplexPointwise, "ComplexPointwise"),
        (GatedPointwise, "GatedPointwise"),
    ])
    def test_complex_correctness(self, device, config, model_class, name):
        """测试复杂 Pointwise 模型正确性"""
        model = model_class(config.input_dim).to(device).eval()
        x = torch.randn(config.batch_size, config.input_dim, device=device)
        
        is_correct, max_diff = verify_correctness(model, x, device, config.rtol, config.atol)
        
        print(f"\n[{name}] max_diff={max_diff:.2e}")
        assert is_correct, f"{name} correctness failed: max_diff={max_diff:.2e}"
    
    @pytest.mark.parametrize("model_class,name", [
        (ComplexPointwise, "ComplexPointwise"),
        (GatedPointwise, "GatedPointwise"),
    ])
    def test_complex_performance(self, device, config, model_class, name):
        """测试复杂 Pointwise 模型性能"""
        model = model_class(config.input_dim).to(device).eval()
        x = torch.randn(config.batch_size, config.input_dim, device=device)
        
        eager_time, compiled_time = benchmark_model(model, x, device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        print(f"\n[{name}] eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms, speedup={speedup:.2f}x")
        
        # 复杂 Pointwise 应该有较好的加速
        assert speedup > 0.5, f"{name} performance regression: speedup={speedup:.2f}x"


class TestMixedModels:
    """混合模型测试"""
    
    def test_mixed_correctness(self, device, config):
        """测试 MixedModel 正确性"""
        model = MixedModel(512).to(device).eval()
        x = torch.randn(config.batch_size, 512, device=device)
        
        is_correct, max_diff = verify_correctness(model, x, device, config.rtol, config.atol)
        
        print(f"\n[MixedModel] max_diff={max_diff:.2e}")
        assert is_correct, f"MixedModel correctness failed: max_diff={max_diff:.2e}"
    
    def test_mixed_performance(self, device, config):
        """测试 MixedModel 性能"""
        model = MixedModel(512).to(device).eval()
        x = torch.randn(config.batch_size, 512, device=device)
        
        eager_time, compiled_time = benchmark_model(model, x, device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        print(f"\n[MixedModel] eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms, speedup={speedup:.2f}x")
    
    def test_large_linear_correctness(self, device, config):
        """测试 LargeLinearModel 正确性"""
        model = LargeLinearModel(config.input_dim).to(device).eval()
        x = torch.randn(config.batch_size, config.input_dim, device=device)
        
        is_correct, max_diff = verify_correctness(model, x, device, rtol=1e-3, atol=1e-3)
        
        print(f"\n[LargeLinearModel] max_diff={max_diff:.2e}")
        assert is_correct, f"LargeLinearModel correctness failed: max_diff={max_diff:.2e}"
    
    def test_large_linear_performance(self, device, config):
        """测试 LargeLinearModel 性能"""
        model = LargeLinearModel(config.input_dim).to(device).eval()
        x = torch.randn(config.batch_size, config.input_dim, device=device)
        
        eager_time, compiled_time = benchmark_model(model, x, device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        print(f"\n[LargeLinearModel] eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms, speedup={speedup:.2f}x")


# =============================================================================
# 参数化测试
# =============================================================================

class TestBatchSizes:
    """测试不同 batch size"""
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64, 128, 256, 512])
    def test_pointwise_batch_sizes(self, device, config, batch_size):
        """测试不同 batch size 下 Pointwise 的性能"""
        model = ComplexPointwise(config.input_dim).to(device).eval()
        x = torch.randn(batch_size, config.input_dim, device=device)
        
        eager_time, compiled_time = benchmark_model(model, x, device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        print(f"\n[batch_size={batch_size}] eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms, speedup={speedup:.2f}x")


class TestInputDims:
    """测试不同 input dim"""
    
    @pytest.mark.parametrize("input_dim", [128, 256, 512, 1024, 2048, 4096])
    def test_pointwise_input_dims(self, device, config, input_dim):
        """测试不同 input_dim 下 Pointwise 的性能"""
        model = ComplexPointwise(input_dim).to(device).eval()
        x = torch.randn(config.batch_size, input_dim, device=device)
        
        eager_time, compiled_time = benchmark_model(model, x, device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        print(f"\n[input_dim={input_dim}] eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms, speedup={speedup:.2f}x")


# =============================================================================
# 全模型对比测试
# =============================================================================

class TestAllModels:
    """所有模型对比测试"""
    
    @pytest.mark.parametrize("model_class,name,dim", [
        (SimplePointwise, "SimplePointwise", 1024),
        (ComplexPointwise, "ComplexPointwise", 1024),
        (GatedPointwise, "GatedPointwise", 1024),
        (SiLUPointwise, "SiLUPointwise", 1024),
        (MixedModel, "MixedModel", 512),
        (LargeLinearModel, "LargeLinearModel", 1024),
    ])
    def test_all_models_comparison(self, device, config, model_class, name, dim):
        """对比所有模型的性能"""
        model = model_class(dim).to(device).eval()
        x = torch.randn(config.batch_size, dim, device=device)
        
        is_correct, max_diff = verify_correctness(model, x, device, rtol=1e-3, atol=1e-3)
        eager_time, compiled_time = benchmark_model(model, x, device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        status = "✓" if is_correct else "✗"
        indicator = "🚀" if speedup > 2.0 else "✓" if speedup > 1.0 else "⚠"
        
        print(f"\n[{name}] {status} correct, speedup={speedup:.2f}x {indicator}")


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    """直接运行时的入口"""
    print("=" * 70)
    print(" GCU Triton 1D 算子测试")
    print(" 使用 pytest 运行: pytest gcu_triton_1d.py -v -s")
    print("=" * 70)
    
    # 检测设备
    try:
        import torch_gcu
        device = "gcu"
        print(f"\n[Device] Using GCU")
    except ImportError:
        if torch.cuda.is_available():
            device = "cuda"
            print(f"\n[Device] Using CUDA")
        else:
            device = "cpu"
            print(f"\n[Device] Using CPU")
    
    print("\n运行测试命令:")
    print("  pytest gcu_triton_1d.py -v                    # 运行所有测试")
    print("  pytest gcu_triton_1d.py -v -k 'pointwise'     # 只测试 pointwise")
    print("  pytest gcu_triton_1d.py -v -k 'mixed'         # 只测试 mixed")
    print("  pytest gcu_triton_1d.py -v -k 'performance'   # 只测试性能")
    print("  pytest gcu_triton_1d.py -v -k 'correctness'   # 只测试正确性")
    print("  pytest gcu_triton_1d.py -v -s                 # 显示详细输出")
    
    # 快速演示
    print("\n" + "=" * 70)
    print(" 快速演示测试")
    print("=" * 70)
    
    config = CONFIG
    
    models = [
        (SimplePointwise, "SimplePointwise", config.input_dim),
        (ComplexPointwise, "ComplexPointwise", config.input_dim),
        (GatedPointwise, "GatedPointwise", config.input_dim),
        (MixedModel, "MixedModel", 512),
    ]
    
    print(f"\n{'Model':<20} {'Correct':<10} {'Eager(ms)':<12} {'Compiled(ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for model_class, name, dim in models:
        model = model_class(dim).to(device).eval()
        x = torch.randn(config.batch_size, dim, device=device)
        
        is_correct, max_diff = verify_correctness(model, x, device)
        eager_time, compiled_time = benchmark_model(model, x, device)
        speedup = eager_time / compiled_time
        
        status = "✓" if is_correct else "✗"
        print(f"{name:<20} {status:<10} {eager_time:<12.3f} {compiled_time:<14.3f} {speedup:<10.2f}x")
    
    print("-" * 70)


if __name__ == "__main__":
    main()
