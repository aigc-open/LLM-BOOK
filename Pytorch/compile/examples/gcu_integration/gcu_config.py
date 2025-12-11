"""
GCU 配置模块
针对 GCU 硬件特性进行优化配置
"""
import torch
import torch._inductor.config as inductor_config
import triton
import functools


class GCUConfig:
    """GCU 性能配置"""
    
    # GCU 硬件限制
    MAX_GRID_SIZE = 48
    OPTIMAL_NUM_WARPS = 1
    BLOCK_SIZE_CANDIDATES = [64, 128, 256, 512]
    
    def __init__(self):
        self._original_getitem = None
        self._patches_applied = False
    
    def configure_inductor(self):
        """配置 TorchInductor 使用 GCU"""
        print("\n[GCU Config] Configuring TorchInductor...")
        
        # 设置设备类型
        inductor_config.cpp.device = "gcu"
        print(f"[GCU Config] ✓ Set device to 'gcu'")
        
        # 配置 Triton 后端
        try:
            triton.runtime.driver.set_active("gcu")
            print(f"[GCU Config] ✓ Set Triton backend to 'gcu'")
        except Exception as e:
            print(f"[GCU Config] ⚠ Could not set Triton backend: {e}")
        
        # 禁用一些可能不兼容的优化
        inductor_config.triton.cudagraph_trees = False
        inductor_config.triton.cudagraphs = False
        print(f"[GCU Config] ✓ Disabled CUDA graphs")
        
        # 启用调试信息
        inductor_config.debug = False  # 可以改为 True 查看更多信息
        inductor_config.trace.enabled = False
        
        print("[GCU Config] ✓ Inductor configuration completed\n")
    
    def apply_gcu_optimizations(self):
        """应用 GCU 特定的优化"""
        if self._patches_applied:
            print("[GCU Config] Optimizations already applied")
            return
        
        print("[GCU Config] Applying GCU-specific optimizations...")
        
        # 1. 修改 Triton Heuristics
        self._patch_heuristics()
        
        # 2. 拦截 Kernel 启动，调整 Grid Size
        self._patch_kernel_launch()
        
        self._patches_applied = True
        print("[GCU Config] ✓ All optimizations applied\n")
    
    def _patch_heuristics(self):
        """修改 Inductor 的 Triton 配置生成逻辑"""
        try:
            import torch._inductor.codegen.triton as inductor_triton
            
            # 保存原始函数
            if hasattr(inductor_triton, 'triton_config'):
                original_config = inductor_triton.triton_config
                
                def gcu_triton_config(*args, **kwargs):
                    """强制 num_warps=1"""
                    configs = original_config(*args, **kwargs)
                    # 修改所有配置的 num_warps
                    if isinstance(configs, list):
                        for cfg in configs:
                            if hasattr(cfg, 'num_warps'):
                                cfg.num_warps = self.OPTIMAL_NUM_WARPS
                    elif hasattr(configs, 'num_warps'):
                        configs.num_warps = self.OPTIMAL_NUM_WARPS
                    return configs
                
                inductor_triton.triton_config = gcu_triton_config
                print("[GCU Config] ✓ Patched Triton heuristics (num_warps=1)")
        except Exception as e:
            print(f"[GCU Config] ⚠ Could not patch heuristics: {e}")
    
    def _patch_kernel_launch(self):
        """拦截 Triton Kernel 启动，自动调整 Grid Size"""
        # 保存原始的 JITFunction.__getitem__
        self._original_getitem = triton.JITFunction.__getitem__
        
        def gcu_getitem(self, grid):
            """包装后的 __getitem__，自动调整 grid"""
            # 调整 grid
            if callable(grid):
                # grid 是 lambda 函数
                original_grid_fn = grid
                
                def adjusted_grid_fn(meta):
                    original_grid = original_grid_fn(meta)
                    adjusted = self._adjust_grid(original_grid)
                    if adjusted != original_grid:
                        print(f"[GCU Grid] Adjusted: {original_grid} → {adjusted}")
                    return adjusted
                
                adjusted_grid = adjusted_grid_fn
            else:
                # grid 是固定值
                adjusted_grid = self._adjust_grid(grid)
                if adjusted_grid != grid:
                    print(f"[GCU Grid] Adjusted: {grid} → {adjusted_grid}")
            
            # 调用原始函数
            return self._original_getitem(self, adjusted_grid)
        
        # 绑定 self 到闭包
        gcu_getitem_bound = lambda self, grid: gcu_getitem(self, grid)
        gcu_getitem_bound._adjust_grid = self._adjust_grid
        
        # 应用补丁
        triton.JITFunction.__getitem__ = gcu_getitem_bound
        print("[GCU Config] ✓ Patched Kernel launch (grid auto-adjustment)")
    
    @staticmethod
    def _adjust_grid(grid):
        """调整 Grid Size 以适应 GCU 硬件限制"""
        MAX_GRID = GCUConfig.MAX_GRID_SIZE
        
        if isinstance(grid, (list, tuple)):
            adjusted = []
            for dim in grid:
                adjusted.append(GCUConfig._adjust_single_dim(dim, MAX_GRID))
            return tuple(adjusted)
        else:
            return GCUConfig._adjust_single_dim(grid, MAX_GRID)
    
    @staticmethod
    def _adjust_single_dim(dim, max_grid):
        """调整单个维度"""
        if dim <= max_grid:
            return dim
        elif dim % max_grid == 0:
            # 已经是 max_grid 的倍数
            return dim
        else:
            # 向下取整到 max_grid 的倍数
            adjusted = (dim // max_grid) * max_grid
            if adjusted == 0:
                adjusted = max_grid
            return adjusted
    
    def verify_kernel_config(self, kernel_info):
        """验证 Kernel 配置是否符合 GCU 要求"""
        issues = []
        
        # 检查 grid size
        grid = kernel_info.get('grid')
        if grid:
            for i, dim in enumerate(grid):
                if dim > self.MAX_GRID_SIZE and dim % self.MAX_GRID_SIZE != 0:
                    issues.append(
                        f"Grid dim {i}: {dim} (not ≤{self.MAX_GRID_SIZE} or multiple of {self.MAX_GRID_SIZE})"
                    )
        
        # 检查 num_warps
        num_warps = kernel_info.get('num_warps')
        if num_warps and num_warps != self.OPTIMAL_NUM_WARPS:
            issues.append(f"num_warps: {num_warps} (expected {self.OPTIMAL_NUM_WARPS})")
        
        return issues


# 全局单例
_gcu_config = GCUConfig()


def configure_gcu():
    """便捷函数：配置 GCU"""
    _gcu_config.configure_inductor()
    _gcu_config.apply_gcu_optimizations()
    return _gcu_config

