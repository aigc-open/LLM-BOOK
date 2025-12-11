#!/usr/bin/env python3
"""
GCU Triton 2D 算子测试

使用 pytest 测试不同类型的 2D 操作：
- Pointwise 2D: 逐元素操作，如 ReLU、GELU、残差连接
- Row-wise Reduction: 行方向归约，如 Softmax、LayerNorm
- Mixed: 混合操作，如 MLP、Attention

使用方法：
    # 运行所有测试
    pytest gcu_triton_2d.py -v
    
    # 只运行 pointwise 测试
    pytest gcu_triton_2d.py -v -k "pointwise"
    
    # 只运行 reduction 测试
    pytest gcu_triton_2d.py -v -k "reduction"
    
    # 运行并显示性能数据
    pytest gcu_triton_2d.py -v -s
    
    # 运行特定模型
    pytest gcu_triton_2d.py -v -k "RMSNorm"
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
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass


# =============================================================================
# 测试配置
# =============================================================================

@dataclass
class TestConfig:
    """测试配置"""
    batch_size: int = 64
    seq_len: int = 128
    hidden_dim: int = 512
    warmup: int = 20
    runs: int = 50
    rtol: float = 1e-4  # 相对误差容限
    atol: float = 1e-4  # 绝对误差容限


# 全局配置
CONFIG = TestConfig()


# =============================================================================
# 设备检测 Fixture
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
# 2D Pointwise 模型
# =============================================================================

class Pointwise2DModel(nn.Module):
    """纯 Pointwise 操作模型"""
    
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(1, hidden_dim))
        self.gate_scale = nn.Parameter(torch.ones(1, hidden_dim))
        self.gate_bias = nn.Parameter(torch.zeros(1, hidden_dim))
    
    def forward(self, x):
        x = x * self.scale + self.bias
        x = F.gelu(x)
        gate = torch.sigmoid(x * self.gate_scale + self.gate_bias)
        x = x * gate
        x = x * 2.0 - 1.0
        x = torch.tanh(x)
        return x


class ResidualPointwise2D(nn.Module):
    """带残差连接的 Pointwise 模型"""
    
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.norm_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.scale = nn.Parameter(torch.ones(hidden_dim) * 0.1)
    
    def forward(self, x):
        residual = x
        x = x * self.norm_weight + self.norm_bias
        x = F.silu(x)
        x = x * self.scale
        x = x + residual
        return x


class SimplePointwise(nn.Module):
    """简单的 Pointwise 操作"""
    
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(hidden_dim))
        self.bias = nn.Parameter(torch.randn(hidden_dim))
    
    def forward(self, x):
        x = x * self.scale + self.bias
        x = torch.relu(x)
        x = x * 2.0 + 1.0
        return x


# =============================================================================
# 2D Reduction 模型
# =============================================================================

class RowSoftmax2D(nn.Module):
    """行方向 Softmax"""
    
    def __init__(self, hidden_dim: int = 512, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.pre_scale = nn.Parameter(torch.ones(1, hidden_dim))
    
    def forward(self, x):
        x = x * self.pre_scale
        x = x / self.temperature
        x = F.softmax(x, dim=-1)
        return x


class LayerNorm2D(nn.Module):
    """手动实现的 LayerNorm"""
    
    def __init__(self, hidden_dim: int = 512, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class RMSNorm2D(nn.Module):
    """RMS Normalization (LLaMA 风格)"""
    
    def __init__(self, hidden_dim: int = 512, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# =============================================================================
# 混合操作模型
# =============================================================================

class FusedMLP(nn.Module):
    """SwiGLU 风格的 MLP"""
    
    def __init__(self, hidden_dim: int = 512, intermediate_dim: int = 2048):
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)


class SelfAttentionScores(nn.Module):
    """Self-Attention 分数计算"""
    
    def __init__(self, head_dim: int = 64):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, q, k):
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * self.scale
        return F.softmax(scores, dim=-1)


class TransformerBlock(nn.Module):
    """简化的 Transformer Block"""
    
    def __init__(self, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.norm1 = RMSNorm2D(hidden_dim)
        self.norm2 = RMSNorm2D(hidden_dim)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.mlp = FusedMLP(hidden_dim, hidden_dim * 4)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.o_proj(out)
        
        x = residual + out
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


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
    # 编译模型
    compiled_model = torch.compile(model, backend="inductor")
    
    # 确保 inputs 是 tuple
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
            _ = compiled_model(*inputs)
    sync_device(device)
    
    # 测试 eager 模式
    times_eager = []
    with torch.no_grad():
        for _ in range(3):
            sync_device(device)
            start = time.time()
            for _ in range(runs):
                _ = model(*inputs)
            sync_device(device)
            times_eager.append((time.time() - start) / runs * 1000)
    
    # 测试 compiled 模式
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
    """
    验证正确性
    
    Returns:
        (is_correct, max_diff)
    """
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

class TestPointwise2D:
    """Pointwise 2D 操作测试"""
    
    @pytest.mark.parametrize("model_class,name", [
        (Pointwise2DModel, "Pointwise2DModel"),
        (ResidualPointwise2D, "ResidualPointwise2D"),
        (SimplePointwise, "SimplePointwise"),
    ])
    def test_pointwise_correctness(self, device, config, model_class, name):
        """测试 Pointwise 模型正确性"""
        model = model_class(config.hidden_dim).to(device).eval()
        x = torch.randn(config.batch_size, config.hidden_dim, device=device)
        
        is_correct, max_diff = verify_correctness(model, x, device, config.rtol, config.atol)
        
        print(f"\n[{name}] max_diff={max_diff:.2e}")
        assert is_correct, f"{name} correctness failed: max_diff={max_diff:.2e}"
    
    @pytest.mark.parametrize("model_class,name", [
        (Pointwise2DModel, "Pointwise2DModel"),
        (ResidualPointwise2D, "ResidualPointwise2D"),
        (SimplePointwise, "SimplePointwise"),
    ])
    def test_pointwise_performance(self, device, config, model_class, name):
        """测试 Pointwise 模型性能"""
        model = model_class(config.hidden_dim).to(device).eval()
        x = torch.randn(config.batch_size, config.hidden_dim, device=device)
        
        eager_time, compiled_time = benchmark_model(model, x, device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        print(f"\n[{name}] eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms, speedup={speedup:.2f}x")
        
        # Pointwise 操作应该有显著加速
        assert speedup > 0.5, f"{name} performance regression: speedup={speedup:.2f}x"


class TestReduction2D:
    """Reduction 2D 操作测试"""
    
    @pytest.mark.parametrize("model_class,name", [
        (RowSoftmax2D, "RowSoftmax2D"),
        (LayerNorm2D, "LayerNorm2D"),
        (RMSNorm2D, "RMSNorm2D"),
    ])
    def test_reduction_correctness(self, device, config, model_class, name):
        """测试 Reduction 模型正确性"""
        model = model_class(config.hidden_dim).to(device).eval()
        x = torch.randn(config.batch_size, config.hidden_dim, device=device)
        
        is_correct, max_diff = verify_correctness(model, x, device, config.rtol, config.atol)
        
        print(f"\n[{name}] max_diff={max_diff:.2e}")
        assert is_correct, f"{name} correctness failed: max_diff={max_diff:.2e}"
    
    @pytest.mark.parametrize("model_class,name", [
        (RowSoftmax2D, "RowSoftmax2D"),
        (LayerNorm2D, "LayerNorm2D"),
        (RMSNorm2D, "RMSNorm2D"),
    ])
    def test_reduction_performance(self, device, config, model_class, name):
        """测试 Reduction 模型性能"""
        model = model_class(config.hidden_dim).to(device).eval()
        x = torch.randn(config.batch_size, config.hidden_dim, device=device)
        
        eager_time, compiled_time = benchmark_model(model, x, device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        print(f"\n[{name}] eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms, speedup={speedup:.2f}x")


class TestMixed2D:
    """混合操作测试"""
    
    def test_fused_mlp_correctness(self, device, config):
        """测试 FusedMLP 正确性"""
        model = FusedMLP(config.hidden_dim, config.hidden_dim * 4).to(device).eval()
        x = torch.randn(config.batch_size, config.hidden_dim, device=device)
        
        is_correct, max_diff = verify_correctness(model, x, device, config.rtol, config.atol)
        
        print(f"\n[FusedMLP] max_diff={max_diff:.2e}")
        assert is_correct, f"FusedMLP correctness failed: max_diff={max_diff:.2e}"
    
    def test_fused_mlp_performance(self, device, config):
        """测试 FusedMLP 性能"""
        model = FusedMLP(config.hidden_dim, config.hidden_dim * 4).to(device).eval()
        x = torch.randn(config.batch_size, config.hidden_dim, device=device)
        
        eager_time, compiled_time = benchmark_model(model, x, device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        print(f"\n[FusedMLP] eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms, speedup={speedup:.2f}x")
    
    def test_attention_scores_correctness(self, device, config):
        """测试 Attention Scores 正确性"""
        model = SelfAttentionScores(head_dim=64).to(device).eval()
        q = torch.randn(config.batch_size // 4, config.seq_len, 64, device=device)
        k = torch.randn(config.batch_size // 4, config.seq_len, 64, device=device)
        
        is_correct, max_diff = verify_correctness(model, (q, k), device, config.rtol, config.atol)
        
        print(f"\n[SelfAttentionScores] max_diff={max_diff:.2e}")
        assert is_correct, f"SelfAttentionScores correctness failed: max_diff={max_diff:.2e}"
    
    def test_attention_scores_performance(self, device, config):
        """测试 Attention Scores 性能"""
        model = SelfAttentionScores(head_dim=64).to(device).eval()
        q = torch.randn(config.batch_size // 4, config.seq_len, 64, device=device)
        k = torch.randn(config.batch_size // 4, config.seq_len, 64, device=device)
        
        eager_time, compiled_time = benchmark_model(model, (q, k), device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        print(f"\n[SelfAttentionScores] eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms, speedup={speedup:.2f}x")


class TestTransformer:
    """Transformer Block 测试"""
    
    def test_transformer_block_correctness(self, device, config):
        """测试 TransformerBlock 正确性"""
        model = TransformerBlock(config.hidden_dim, num_heads=8).to(device).eval()
        x = torch.randn(config.batch_size // 4, config.seq_len, config.hidden_dim, device=device)
        
        # Transformer 可能有更大的误差
        is_correct, max_diff = verify_correctness(model, x, device, rtol=1e-3, atol=1e-3)
        
        print(f"\n[TransformerBlock] max_diff={max_diff:.2e}")
        assert is_correct, f"TransformerBlock correctness failed: max_diff={max_diff:.2e}"
    
    def test_transformer_block_performance(self, device, config):
        """测试 TransformerBlock 性能"""
        model = TransformerBlock(config.hidden_dim, num_heads=8).to(device).eval()
        x = torch.randn(config.batch_size // 4, config.seq_len, config.hidden_dim, device=device)
        
        eager_time, compiled_time = benchmark_model(model, x, device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        print(f"\n[TransformerBlock] eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms, speedup={speedup:.2f}x")


# =============================================================================
# 参数化批量测试
# =============================================================================

class TestBatchSizes:
    """测试不同 batch size 的影响"""
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64, 128, 256])
    def test_pointwise_batch_sizes(self, device, config, batch_size):
        """测试不同 batch size 下 Pointwise 的性能"""
        model = Pointwise2DModel(config.hidden_dim).to(device).eval()
        x = torch.randn(batch_size, config.hidden_dim, device=device)
        
        eager_time, compiled_time = benchmark_model(model, x, device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        print(f"\n[batch_size={batch_size}] eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms, speedup={speedup:.2f}x")


class TestHiddenDims:
    """测试不同 hidden_dim 的影响"""
    
    @pytest.mark.parametrize("hidden_dim", [128, 256, 512, 1024, 2048])
    def test_pointwise_hidden_dims(self, device, config, hidden_dim):
        """测试不同 hidden_dim 下 Pointwise 的性能"""
        model = Pointwise2DModel(hidden_dim).to(device).eval()
        x = torch.randn(config.batch_size, hidden_dim, device=device)
        
        eager_time, compiled_time = benchmark_model(model, x, device, config.warmup, config.runs)
        speedup = eager_time / compiled_time
        
        print(f"\n[hidden_dim={hidden_dim}] eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms, speedup={speedup:.2f}x")


# =============================================================================
# 命令行入口（兼容直接运行）
# =============================================================================

def main():
    """直接运行时的入口"""
    print("=" * 70)
    print(" GCU Triton 2D 算子测试")
    print(" 使用 pytest 运行: pytest gcu_triton_2d.py -v -s")
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
    print("  pytest gcu_triton_2d.py -v                    # 运行所有测试")
    print("  pytest gcu_triton_2d.py -v -k 'pointwise'     # 只测试 pointwise")
    print("  pytest gcu_triton_2d.py -v -k 'reduction'     # 只测试 reduction")
    print("  pytest gcu_triton_2d.py -v -k 'performance'   # 只测试性能")
    print("  pytest gcu_triton_2d.py -v -k 'correctness'   # 只测试正确性")
    print("  pytest gcu_triton_2d.py -v -s                 # 显示详细输出")
    
    # 运行简单的演示测试
    print("\n" + "=" * 70)
    print(" 快速演示测试")
    print("=" * 70)
    
    config = CONFIG
    
    # 测试一个简单模型
    model = Pointwise2DModel(config.hidden_dim).to(device).eval()
    x = torch.randn(config.batch_size, config.hidden_dim, device=device)
    
    is_correct, max_diff = verify_correctness(model, x, device)
    eager_time, compiled_time = benchmark_model(model, x, device)
    speedup = eager_time / compiled_time
    
    print(f"\n[Pointwise2DModel]")
    print(f"  Correctness: {'✓' if is_correct else '✗'} (max_diff={max_diff:.2e})")
    print(f"  Performance: eager={eager_time:.3f}ms, compiled={compiled_time:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
