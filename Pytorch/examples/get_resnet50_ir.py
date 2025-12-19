"""
获取 ResNet50 IR 文件 - 最简单直接的方式
"""

import torch
import torchvision
import os
import torch._inductor.config as config
config.debug = True
config.trace.enabled = True
config.triton.autotune_at_compile_time = False # 尝试关闭

def main():
    print("="*70)
    print("  ResNet50 IR 获取工具")
    print("="*70)
    print()
    
    # 1. 设置环境
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_LOGS"] = "+inductor"
    
    # 2. 配置 inductor
    import torch._inductor.config as config
    config.debug = True
    
    print("配置:")
    print(f"  TORCH_COMPILE_DEBUG = 1")
    print(f"  inductor.debug = True")
    print()
    
    # 3. 创建模型
    print("创建 ResNet50...")
    model = torchvision.models.resnet50(weights=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)
    model.eval()
    
    # 4. 创建输入
    x = torch.randn(2, 3, 224, 224, dtype=torch.float32, device=device)  # 放到同样的设备
    x = x.to(memory_format=torch.channels_last)
    print(f"输入: {x.shape}, 设备: {x.device}\n")
    
    # 5. 编译
    print("编译模型（这需要几分钟）...")
    compiled_model = torch.compile(model, backend="inductor", mode="default")
    
    # 6. 运行
    # print("运行推理...")
    with torch.no_grad():
        y = compiled_model(x)
    
    # print(f"✓ 输出: {y.shape}\n")
    
    # # 7. 查找文件
    # print("="*70)
    # print("查找生成的文件:")
    # print("="*70)
    
    # debug_dir = "./torch_compile_debug"
    # if os.path.exists(debug_dir):
    #     import glob
        
    #     # 查找所有 .py 文件
    #     py_files = glob.glob(f"{debug_dir}/**/*.py", recursive=True)
    #     if py_files:
    #         print(f"✓ 找到 {len(py_files)} 个 Python 文件:\n")
    #         for f in py_files[-5:]:  # 显示最新的 5 个
    #             print(f"  {f}")
    #         print()
    #     else:
    #         print("✗ 未找到 .py 文件\n")
        
    #     # 查找所有 .log 文件
    #     log_files = glob.glob(f"{debug_dir}/**/*.log", recursive=True)
    #     if log_files:
    #         print(f"✓ 找到 {len(log_files)} 个日志文件:\n")
    #         for f in log_files[-3:]:
    #             print(f"  {f}")
    #         print()
        
    #     # 显示目录结构
    #     print("目录结构:")
    #     os.system(f"find {debug_dir} -type d | head -10")
    #     print()
    # else:
    #     print(f"✗ 目录不存在: {debug_dir}\n")
    
    # print("="*70)
    # print("完成！")
    # print("="*70)


if __name__ == "__main__":
    main()

