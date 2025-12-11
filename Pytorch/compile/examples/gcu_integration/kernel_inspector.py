"""
Triton Kernel 检查模块
用于捕获和分析生成的 Triton Kernel
"""
import os
import sys
import re
from io import StringIO
from contextlib import contextmanager


class KernelInspector:
    """Triton Kernel 检查器"""
    
    def __init__(self):
        self.captured_kernels = []
        self.output_dir = "/tmp/gcu_kernels"
    
    @contextmanager
    def capture_kernels(self):
        """上下文管理器：捕获生成的 Kernel 代码"""
        print("\n[Kernel Inspector] Capturing generated Triton kernels...")
        
        # 设置环境变量以输出代码
        old_env = {}
        env_vars = {
            'TORCH_LOGS': '+output_code',
            'TORCH_COMPILE_DEBUG': '1',
        }
        
        for key, value in env_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 捕获 stdout
        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            yield self
        finally:
            # 恢复 stdout
            sys.stdout = old_stdout
            
            # 恢复环境变量
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            
            # 处理捕获的输出
            output = captured_output.getvalue()
            self._parse_kernels(output)
            
            print(f"[Kernel Inspector] ✓ Captured {len(self.captured_kernels)} kernel(s)")
    
    def _parse_kernels(self, output):
        """解析捕获的输出，提取 Kernel 信息"""
        # 简单的解析逻辑（可以根据实际输出格式调整）
        lines = output.split('\n')
        
        current_kernel = None
        for line in lines:
            # 检测 Kernel 定义
            if '@triton.jit' in line or 'def triton_' in line:
                if current_kernel:
                    self.captured_kernels.append(current_kernel)
                current_kernel = {'name': '', 'code': '', 'config': {}}
            
            if current_kernel is not None:
                current_kernel['code'] += line + '\n'
                
                # 提取 Kernel 名称
                match = re.search(r'def (triton_\w+)', line)
                if match:
                    current_kernel['name'] = match.group(1)
                
                # 提取配置信息
                if 'num_warps' in line:
                    match = re.search(r'num_warps[=:](\d+)', line)
                    if match:
                        current_kernel['config']['num_warps'] = int(match.group(1))
                
                if 'grid' in line:
                    match = re.search(r'grid[=:](\([^)]+\)|\d+)', line)
                    if match:
                        current_kernel['config']['grid'] = match.group(1)
        
        if current_kernel:
            self.captured_kernels.append(current_kernel)
    
    def print_summary(self):
        """打印 Kernel 摘要"""
        if not self.captured_kernels:
            print("\n[Kernel Inspector] ⚠ No kernels captured")
            print("[Kernel Inspector] This might be because:")
            print("  1. The model is too simple (no kernel generation)")
            print("  2. Triton backend is not properly configured")
            print("  3. Output format has changed")
            return
        
        print("\n" + "="*70)
        print(f"{'CAPTURED TRITON KERNELS':^70}")
        print("="*70)
        
        for i, kernel in enumerate(self.captured_kernels, 1):
            print(f"\n[Kernel {i}] {kernel.get('name', 'Unknown')}")
            print("-" * 70)
            
            # 打印配置
            config = kernel.get('config', {})
            if config:
                print("Configuration:")
                for key, value in config.items():
                    print(f"  {key}: {value}")
            
            # 打印代码片段（前20行）
            code_lines = kernel.get('code', '').split('\n')
            if code_lines:
                print("\nCode snippet (first 20 lines):")
                for line in code_lines[:20]:
                    if line.strip():
                        print(f"  {line}")
            
            # 验证配置
            self._verify_kernel_config(kernel, i)
        
        print("\n" + "="*70 + "\n")
    
    def _verify_kernel_config(self, kernel, kernel_num):
        """验证单个 Kernel 的配置"""
        from gcu_config import GCUConfig
        
        config = kernel.get('config', {})
        issues = []
        
        # 检查 num_warps
        num_warps = config.get('num_warps')
        if num_warps is not None:
            if num_warps == GCUConfig.OPTIMAL_NUM_WARPS:
                print(f"  ✓ num_warps = {num_warps} (optimal)")
            else:
                print(f"  ✗ num_warps = {num_warps} (expected {GCUConfig.OPTIMAL_NUM_WARPS})")
                issues.append(f"num_warps should be {GCUConfig.OPTIMAL_NUM_WARPS}")
        
        # 检查 grid
        grid_str = config.get('grid')
        if grid_str:
            try:
                # 尝试解析 grid
                if isinstance(grid_str, str):
                    grid_str = grid_str.strip('()')
                    grid_dims = [int(x.strip()) for x in grid_str.split(',') if x.strip().isdigit()]
                else:
                    grid_dims = [grid_str] if isinstance(grid_str, int) else []
                
                for dim in grid_dims:
                    if dim <= GCUConfig.MAX_GRID_SIZE:
                        print(f"  ✓ grid dim = {dim} (≤ {GCUConfig.MAX_GRID_SIZE})")
                    elif dim % GCUConfig.MAX_GRID_SIZE == 0:
                        print(f"  ✓ grid dim = {dim} (multiple of {GCUConfig.MAX_GRID_SIZE})")
                    else:
                        print(f"  ✗ grid dim = {dim} (not ≤{GCUConfig.MAX_GRID_SIZE} or multiple)")
                        issues.append(f"grid dim {dim} violates GCU constraint")
            except Exception as e:
                print(f"  ⚠ Could not parse grid: {e}")
        
        if not issues:
            print(f"  ✓ Kernel {kernel_num} configuration is valid for GCU")
        else:
            print(f"  ✗ Kernel {kernel_num} has {len(issues)} issue(s)")
            for issue in issues:
                print(f"    - {issue}")
    
    def save_kernels(self):
        """保存 Kernel 到文件"""
        if not self.captured_kernels:
            return
        
        print(f"\n[Kernel Inspector] Saving kernels to {self.output_dir}/")
        
        for i, kernel in enumerate(self.captured_kernels, 1):
            name = kernel.get('name', f'kernel_{i}')
            filename = os.path.join(self.output_dir, f"{name}.py")
            
            with open(filename, 'w') as f:
                f.write(f"# Kernel: {name}\n")
                f.write(f"# Configuration: {kernel.get('config', {})}\n\n")
                f.write(kernel.get('code', ''))
            
            print(f"  ✓ Saved: {filename}")
        
        print(f"[Kernel Inspector] ✓ All kernels saved\n")


def inspect_kernels():
    """便捷函数：创建 Kernel 检查器"""
    return KernelInspector()

