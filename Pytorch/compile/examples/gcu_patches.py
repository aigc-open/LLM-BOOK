#!/usr/bin/env python3
"""
GCU Triton 补丁模块

用于解决 triton_gcu 的硬件限制：
1. num_warps <= 4 (GCU 硬件限制)
2. vector_length = 256 (i64 类型向量长度限制)

使用方法：
    # 在任何 torch.compile 之前导入
    import gcu_patches
    
    # 或者手动调用
    from gcu_patches import apply_all_patches
    apply_all_patches()
"""

import os


# =============================================================================
# GCU 硬件限制常量
# =============================================================================

GCU_MAX_WARPS = 4           # num_warps 最大值
# GCU_MAX_VECTOR_LENGTH = 256  # i64 向量最大长度 (暂时禁用)
GCU_MAX_BLOCK_SIZE = 256     # block size 最大值


# =============================================================================
# Patch 函数
# =============================================================================

def patch_triton_num_warps():
    """
    Patch Triton 的 num_warps 限制
    
    GCU 错误信息: "num_warps must not exceed 4"
    
    关键：必须 patch triton.compile 函数，因为 TorchInductor 直接调用它
    """
    patched = []
    
    try:
        import triton
        import triton.compiler.compiler as triton_compiler
        
        # =====================================================================
        # 关键 Patch: triton.compile 函数
        # TorchInductor 直接调用这个函数，必须在这里拦截 num_warps
        # =====================================================================
        if not getattr(triton_compiler, '_gcu_compile_patched', False):
            _original_compile = triton_compiler.compile
            
            def _patched_compile(src, target=None, options=None, **kwargs):
                # 限制 options 中的 num_warps
                if options is not None and hasattr(options, 'num_warps'):
                    if options.num_warps > GCU_MAX_WARPS:
                        options.num_warps = GCU_MAX_WARPS
                
                # 限制 kwargs 中的 num_warps
                if 'num_warps' in kwargs and kwargs['num_warps'] > GCU_MAX_WARPS:
                    kwargs['num_warps'] = GCU_MAX_WARPS
                
                return _original_compile(src, target=target, options=options, **kwargs)
            
            triton_compiler.compile = _patched_compile
            triton.compile = _patched_compile  # 也要 patch 顶层导出
            triton_compiler._gcu_compile_patched = True
            patched.append("triton.compile")
        
        # Patch triton.Config.__init__
        if hasattr(triton, 'Config') and not getattr(triton.Config, '_gcu_patched', False):
            _original_config_init = triton.Config.__init__
            
            def _patched_config_init(self, kwargs=None, num_warps=4, num_stages=2, 
                                     num_ctas=1, maxnreg=None, pre_hook=None):
                num_warps = min(num_warps, GCU_MAX_WARPS)
                return _original_config_init(
                    self, kwargs, num_warps=num_warps, num_stages=num_stages,
                    num_ctas=num_ctas, maxnreg=maxnreg, pre_hook=pre_hook
                )
            
            triton.Config.__init__ = _patched_config_init
            triton.Config._gcu_patched = True
            patched.append("triton.Config")
        
        # Patch triton.JITFunction.run
        if hasattr(triton, 'JITFunction') and not getattr(triton.JITFunction, '_gcu_patched', False):
            _original_jit_run = triton.JITFunction.run
            
            def _patched_jit_run(self, *args, num_warps=4, num_stages=2, 
                                 num_ctas=1, enable_warp_specialization=False,
                                 enable_fp_fusion=True, **kwargs):
                num_warps = min(num_warps, GCU_MAX_WARPS)
                return _original_jit_run(
                    self, *args, num_warps=num_warps, num_stages=num_stages,
                    num_ctas=num_ctas, enable_warp_specialization=enable_warp_specialization,
                    enable_fp_fusion=enable_fp_fusion, **kwargs
                )
            
            triton.JITFunction.run = _patched_jit_run
            triton.JITFunction._gcu_patched = True
            patched.append("triton.JITFunction.run")
            
    except ImportError:
        pass
    except Exception as e:
        print(f"[GCU Patch] ⚠ Error patching triton: {e}")
    
    # =========================================================================
    # Patch TorchInductor triton_heuristics
    # =========================================================================
    try:
        from torch._inductor.runtime import triton_heuristics
        
        if hasattr(triton_heuristics, 'triton_config') and \
           not getattr(triton_heuristics, '_gcu_patched', False):
            _original_triton_config = triton_heuristics.triton_config
            
            def _patched_triton_config(*args, num_warps=4, **kwargs):
                num_warps = min(num_warps, GCU_MAX_WARPS)
                return _original_triton_config(*args, num_warps=num_warps, **kwargs)
            
            triton_heuristics.triton_config = _patched_triton_config
            triton_heuristics._gcu_patched = True
            patched.append("triton_heuristics.triton_config")
            
    except ImportError:
        pass
    except Exception as e:
        print(f"[GCU Patch] ⚠ Error patching triton_heuristics: {e}")
    
    # =========================================================================
    # Patch TorchInductor codegen 的默认 num_warps
    # =========================================================================
    try:
        from torch._inductor.codegen import triton_utils
        
        # 修改默认的 num_warps 配置
        if hasattr(triton_utils, 'config_of') and \
           not getattr(triton_utils, '_gcu_patched', False):
            _original_config_of = triton_utils.config_of
            
            def _patched_config_of(*args, **kwargs):
                config = _original_config_of(*args, **kwargs)
                if hasattr(config, 'num_warps') and config.num_warps > GCU_MAX_WARPS:
                    config.num_warps = GCU_MAX_WARPS
                return config
            
            triton_utils.config_of = _patched_config_of
            triton_utils._gcu_patched = True
            patched.append("triton_utils.config_of")
    except:
        pass
    
    return patched


def patch_gcu_vector_length():
    """
    Patch triton_gcu 的 vector_length
    
    GCU 错误信息: "vector<512xi64>" 不支持，最大支持 "vector<256xi64>"
    
    注意：暂时禁用此补丁
    """
    # 暂时禁用 vector_length 补丁
    return []
    
    # patched = []
    # try:
    #     import triton_gcu.triton.compiler as triton_gcu_compiler
    #     if hasattr(triton_gcu_compiler, 'GCUOptions'):
    #         current_value = getattr(triton_gcu_compiler.GCUOptions, 'vector_length', 512)
    #         if current_value != GCU_MAX_VECTOR_LENGTH:
    #             triton_gcu_compiler.GCUOptions.vector_length = GCU_MAX_VECTOR_LENGTH
    #             patched.append(f"GCUOptions.vector_length: {current_value} -> {GCU_MAX_VECTOR_LENGTH}")
    # except ImportError:
    #     pass
    # except Exception as e:
    #     print(f"[GCU Patch] ⚠ Error patching vector_length: {e}")
    # return patched


def patch_inductor_config():
    """
    配置 TorchInductor 以适应 GCU
    """
    patched = []
    
    try:
        import torch._inductor.config as inductor_config
        
        # 禁用 CUDA graphs（GCU 不支持）
        inductor_config.triton.cudagraphs = False
        inductor_config.triton.cudagraph_trees = False
        patched.append("cudagraphs=False")
        
        # 禁用自动调优（可能产生不兼容的配置）
        inductor_config.max_autotune = False
        inductor_config.triton.autotune_at_compile_time = False
        patched.append("autotune=False")
        
        # 限制 block sizes
        inductor_config.triton.max_tiles = 2
        patched.append("max_tiles=2")
        
        # 关键：禁用子进程编译，这样补丁才能生效
        # 否则子进程不会继承我们的 Python 补丁
        inductor_config.compile_threads = 1
        inductor_config.worker_start_method = "fork"  # fork 会继承补丁
        patched.append("compile_threads=1")
        
    except ImportError:
        pass
    except Exception as e:
        print(f"[GCU Patch] ⚠ Error configuring inductor: {e}")
    
    return patched


def patch_triton_gcu_compiler():
    """
    直接 patch triton_gcu 的编译器来处理 num_warps
    
    这是最可靠的方法，因为它在 assertion 之前修改值
    """
    patched = []
    
    try:
        import triton_gcu.triton.compiler as triton_gcu_compiler
        
        # Patch make_ttir 函数来 clamp num_warps 而不是 assert
        if hasattr(triton_gcu_compiler, 'make_ttir') and \
           not getattr(triton_gcu_compiler, '_gcu_make_ttir_patched', False):
            
            _original_make_ttir = triton_gcu_compiler.make_ttir
            
            def _patched_make_ttir(src, metadata, options):
                # 在调用原函数之前，限制 num_warps
                if hasattr(options, 'num_warps') and options.num_warps > GCU_MAX_WARPS:
                    options.num_warps = GCU_MAX_WARPS
                
                # 也检查 metadata
                if isinstance(metadata, dict) and 'num_warps' in metadata:
                    if metadata['num_warps'] > GCU_MAX_WARPS:
                        metadata['num_warps'] = GCU_MAX_WARPS
                
                return _original_make_ttir(src, metadata, options)
            
            triton_gcu_compiler.make_ttir = _patched_make_ttir
            triton_gcu_compiler._gcu_make_ttir_patched = True
            patched.append("triton_gcu.make_ttir")
        
        # Patch GCUOptions 类来限制 num_warps
        if hasattr(triton_gcu_compiler, 'GCUOptions'):
            GCUOptions = triton_gcu_compiler.GCUOptions
            
            if not getattr(GCUOptions, '_gcu_init_patched', False):
                _original_post_init = getattr(GCUOptions, '__post_init__', None)
                
                def _patched_post_init(self):
                    # 先调用原始 __post_init__
                    if _original_post_init:
                        _original_post_init(self)
                    
                    # 然后限制 num_warps
                    if hasattr(self, 'num_warps') and self.num_warps > GCU_MAX_WARPS:
                        self.num_warps = GCU_MAX_WARPS
                
                GCUOptions.__post_init__ = _patched_post_init
                GCUOptions._gcu_init_patched = True
                patched.append("GCUOptions.__post_init__")
        
    except ImportError:
        pass
    except Exception as e:
        print(f"[GCU Patch] ⚠ Error patching triton_gcu compiler: {e}")
    
    return patched


def set_environment_variables():
    """
    设置 GCU 相关环境变量
    """
    env_vars = {
        'TRITON_NUM_WARPS': str(GCU_MAX_WARPS),
        'TRITON_MAX_BLOCK_SIZE': str(GCU_MAX_BLOCK_SIZE),
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return list(env_vars.keys())


# =============================================================================
# 主入口
# =============================================================================

def apply_all_patches(verbose=True):
    """
    应用所有 GCU 补丁
    
    Args:
        verbose: 是否打印详细信息
    
    Returns:
        dict: 各部分 patch 的结果
    """
    results = {}
    
    if verbose:
        print("[GCU Patches] Applying GCU-specific patches...")
        print(f"[GCU Patches] Limits: num_warps<={GCU_MAX_WARPS}")
    
    # 1. 环境变量
    results['env'] = set_environment_variables()
    if verbose and results['env']:
        print(f"[GCU Patches] ✓ Environment: {', '.join(results['env'])}")
    
    # 2. Inductor config (先配置，禁用子进程编译)
    results['inductor'] = patch_inductor_config()
    if verbose and results['inductor']:
        print(f"[GCU Patches] ✓ Inductor: {', '.join(results['inductor'])}")
    
    # 3. triton_gcu 编译器 (最关键的补丁)
    results['triton_gcu'] = patch_triton_gcu_compiler()
    if verbose and results['triton_gcu']:
        print(f"[GCU Patches] ✓ triton_gcu: {', '.join(results['triton_gcu'])}")
    
    # 4. Triton num_warps
    results['triton'] = patch_triton_num_warps()
    if verbose and results['triton']:
        print(f"[GCU Patches] ✓ Triton: {', '.join(results['triton'])}")
    
    # 5. GCU vector_length
    results['vector'] = patch_gcu_vector_length()
    if verbose and results['vector']:
        print(f"[GCU Patches] ✓ Vector: {', '.join(results['vector'])}")
    
    if verbose:
        total = sum(len(v) for v in results.values())
        print(f"[GCU Patches] ✓ Applied {total} patches successfully")
    
    return results


# =============================================================================
# 模块加载时自动应用补丁
# =============================================================================

# 自动应用（导入即生效）
_patch_results = apply_all_patches(verbose=True)

