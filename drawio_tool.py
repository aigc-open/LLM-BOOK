#!/usr/bin/env python3
"""
DrawIO 命令行工具
功能：安装依赖、导出 drawio 文件
使用 Fire 库提供清晰的命令行接口
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List
import shutil


class DrawIOTool:
    """DrawIO 命令行工具"""
    
    def __init__(self):
        self.colors = {
            'RED': '\033[0;31m',
            'GREEN': '\033[0;32m',
            'YELLOW': '\033[1;33m',
            'BLUE': '\033[0;34m',
            'NC': '\033[0m'  # No Color
        }
    
    def _print(self, msg: str, color: str = 'NC'):
        """打印带颜色的消息"""
        print(f"{self.colors[color]}{msg}{self.colors['NC']}")
    
    def _print_info(self, msg: str):
        """打印信息"""
        self._print(f"[INFO] {msg}", 'GREEN')
    
    def _print_error(self, msg: str):
        """打印错误"""
        self._print(f"[ERROR] {msg}", 'RED')
    
    def _print_warning(self, msg: str):
        """打印警告"""
        self._print(f"[WARNING] {msg}", 'YELLOW')
    
    def _print_step(self, msg: str):
        """打印步骤"""
        print()
        self._print(f"==> {msg}", 'GREEN')
    
    def _run_cmd(self, cmd: List[str], check: bool = True, quiet: bool = False) -> tuple:
        """运行shell命令"""
        try:
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=True,
                text=True
            )
            if not quiet and result.stdout:
                print(result.stdout.strip())
            return result.returncode, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            if not quiet:
                self._print_error(f"命令执行失败: {' '.join(cmd)}")
                if e.stderr:
                    print(e.stderr)
            return e.returncode, e.stdout, e.stderr
    
    def _check_root(self) -> bool:
        """检查是否为root用户"""
        return os.geteuid() == 0
    
    def _command_exists(self, cmd: str) -> bool:
        """检查命令是否存在"""
        return shutil.which(cmd) is not None
    
    def install_deps(self, skip_update: bool = False):
        """
        安装 DrawIO 导出所需的所有依赖
        
        Args:
            skip_update: 跳过 apt-get update（默认：False）
        
        Examples:
            python drawio_tool.py install-deps
            python drawio_tool.py install-deps --skip-update
        """
        self._print_info("DrawIO 命令行导出依赖安装工具")
        self._print_info("=" * 50)
        
        # 检查root权限
        if not self._check_root():
            self._print_error("请使用 root 权限运行此命令")
            self._print_info("使用方法: sudo python drawio_tool.py install-deps")
            sys.exit(1)
        
        # 步骤1: 更新包列表
        if not skip_update:
            self._print_step("1/4 更新软件包列表...")
            ret, _, _ = self._run_cmd(['apt-get', 'update', '-qq'])
            if ret != 0:
                self._print_error("更新软件包列表失败")
                sys.exit(1)
            self._print_info("软件包列表更新完成")
        else:
            self._print_warning("跳过软件包列表更新")
        
        # 步骤2: 安装xvfb
        self._print_step("2/4 安装虚拟显示环境 (xvfb)...")
        if self._command_exists('xvfb-run'):
            self._print_warning("xvfb 已安装，跳过")
        else:
            ret, _, _ = self._run_cmd(['apt-get', 'install', '-y', 'xvfb'])
            if ret != 0:
                self._print_error("xvfb 安装失败")
                sys.exit(1)
            self._print_info("xvfb 安装完成")
        
        # 步骤3: 安装字体
        self._print_step("3/4 安装字体包（中文 + Emoji）...")
        
        fonts = [
            # 中文字体
            'fonts-wqy-microhei',
            'fonts-wqy-zenhei',
            'fonts-noto-cjk',
            'xfonts-wqy',
            # Emoji字体
            'fonts-noto-color-emoji',
            'fonts-symbola'
        ]
        
        self._print_info("安装以下字体包:")
        for font in fonts:
            print(f"  - {font}")
        
        ret, _, _ = self._run_cmd(['apt-get', 'install', '-y'] + fonts)
        if ret != 0:
            self._print_error("字体安装失败")
            sys.exit(1)
        self._print_info("所有字体安装完成")
        
        # 步骤4: 刷新字体缓存
        self._print_step("4/4 刷新字体缓存...")
        ret, _, _ = self._run_cmd(['fc-cache', '-fv'], quiet=True)
        if ret != 0:
            self._print_error("字体缓存刷新失败")
            sys.exit(1)
        self._print_info("字体缓存刷新完成")
        
        # 验证安装
        self._print_step("验证安装...")
        
        if self._command_exists('xvfb-run'):
            self._print_info(f"✓ xvfb-run 已安装: {shutil.which('xvfb-run')}")
        else:
            self._print_error("✗ xvfb-run 未找到")
        
        if self._command_exists('drawio'):
            self._print_info(f"✓ drawio 已安装: {shutil.which('drawio')}")
        else:
            self._print_warning("✗ drawio 未安装")
            self._print_warning("请安装 draw.io 桌面版: https://github.com/jgraph/drawio-desktop/releases")
        
        # 检查中文字体
        ret, stdout, _ = self._run_cmd(['fc-list', ':lang=zh'], quiet=True)
        if ret == 0 and stdout:
            self._print_info("✓ 中文字体已安装")
            for line in stdout.strip().split('\n')[:3]:
                print(f"  {line}")
        
        # 检查Emoji字体
        ret, stdout, _ = self._run_cmd(['fc-list'], quiet=True)
        if ret == 0 and 'emoji' in stdout.lower():
            self._print_info("✓ Emoji 字体已安装")
        
        # 完成
        self._print_step("安装完成！")
        self._print_info("=" * 50)
        self._print_info("现在可以使用 export 命令导出 drawio 文件")
        print()
        self._print_info("示例: python drawio_tool.py export input.drawio output.png")
        self._print_info("=" * 50)
    
    def _export_single_file(
        self,
        input_file: str,
        output_file: str,
        format: str,
        page_index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: Optional[float] = None,
        transparent: bool = False,
        border: Optional[int] = None,
        quiet: bool = False
    ) -> bool:
        """内部方法：导出单个文件"""
        # 构建命令
        cmd = [
            'xvfb-run', '-a',
            'drawio', '--no-sandbox', '--export',
            '--format', format,
            '--output', output_file
        ]
        
        # 添加可选参数
        if page_index is not None:
            cmd.extend(['--page-index', str(page_index)])
        
        if width is not None:
            cmd.extend(['--width', str(width)])
        
        if height is not None:
            cmd.extend(['--height', str(height)])
        
        if scale is not None:
            cmd.extend(['--scale', str(scale)])
        
        if transparent and format == 'png':
            cmd.append('--transparent')
        
        if border is not None:
            cmd.extend(['--border', str(border)])
        
        cmd.append(input_file)
        
        # 打印信息
        if not quiet:
            self._print_info(f"输入: {input_file}")
            self._print_info(f"输出: {output_file}")
            self._print_info(f"格式: {format}")
            
            if page_index is not None:
                self._print_info(f"页面: {page_index}")
            if width:
                self._print_info(f"宽度: {width}px")
            if height:
                self._print_info(f"高度: {height}px")
            if scale:
                self._print_info(f"缩放: {scale}x")
            if transparent:
                self._print_info("透明背景: 是")
            
            print()
        
        # 执行导出
        ret, stdout, stderr = self._run_cmd(cmd, check=False, quiet=quiet)
        
        # 检查是否成功
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size
            if file_size >= 1024 * 1024:
                size_str = f"{file_size / 1024 / 1024:.2f} MB"
            elif file_size >= 1024:
                size_str = f"{file_size / 1024:.2f} KB"
            else:
                size_str = f"{file_size} B"
            
            if not quiet:
                self._print_info(f"✓ 导出成功！文件大小: {size_str}")
                self._print_info(f"保存位置: {Path(output_file).absolute()}")
            return True
        else:
            if not quiet:
                self._print_error("✗ 导出失败")
            return False
    
    def export(
        self,
        drawio_path: str,
        output: Optional[str] = None,
        output_dir: str = "./output",
        format: str = "png",
        page_index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: Optional[float] = None,
        transparent: bool = False,
        border: Optional[int] = None
    ):
        """
        导出 drawio 文件为图片或 PDF，支持通配符匹配
        
        Args:
            drawio_path: drawio 文件路径，支持通配符 (如 *.drawio, dir/**/*.drawio)
            output: 输出文件路径（单文件导出时使用，可选）
            output_dir: 输出目录（批量导出时使用，默认：./output）
            format: 输出格式 (png/jpg/svg/pdf)，默认：png
            page_index: 页面索引（从0开始），用于多页文档
            width: 输出宽度（像素）
            height: 输出高度（像素）
            scale: 缩放比例（如 2 表示 2倍分辨率）
            transparent: PNG 透明背景（仅 PNG 格式）
            border: 边框宽度（像素）
        
        Examples:
            # 单文件导出
            python drawio_tool.py export input.drawio --output=output.png
            
            # 使用通配符导出当前目录所有 drawio 文件
            python drawio_tool.py export "*.drawio" --output-dir=./images
            
            # 使用路径模式导出
            python drawio_tool.py export "diagrams/*.drawio" --output-dir=./output
            
            # 递归匹配
            python drawio_tool.py export "**/*.drawio" --output-dir=./output
            
            # 指定特定文件模式
            python drawio_tool.py export "page*.drawio" --format=png --scale=2
            
            # 高级选项
            python drawio_tool.py export "*.drawio" --output-dir=./out --scale=2 --transparent
        """
        # 检查依赖
        if not self._command_exists('xvfb-run'):
            self._print_error("xvfb-run 未安装")
            self._print_error("请先运行: sudo python drawio_tool.py install-deps")
            sys.exit(1)
        
        if not self._command_exists('drawio'):
            self._print_error("drawio 未安装")
            self._print_error("请安装 draw.io 桌面版: https://github.com/jgraph/drawio-desktop/releases")
            sys.exit(1)
        
        # 验证格式
        valid_formats = ['png', 'jpg', 'jpeg', 'svg', 'pdf']
        if format not in valid_formats:
            self._print_error(f"不支持的格式: {format}")
            self._print_error(f"支持的格式: {', '.join(valid_formats)}")
            sys.exit(1)
        
        # 使用 glob 查找匹配的文件
        import glob
        
        # 支持递归通配符
        matched_files = glob.glob(drawio_path, recursive=True)
        
        # 过滤出 .drawio 文件
        drawio_files = [f for f in matched_files if f.endswith('.drawio') and Path(f).is_file()]
        
        # 如果没有匹配到文件，检查是否是单个文件
        if not drawio_files:
            if Path(drawio_path).exists() and Path(drawio_path).is_file():
                drawio_files = [drawio_path]
            else:
                self._print_error(f"未找到匹配的 drawio 文件: {drawio_path}")
                sys.exit(1)
        
        # 单文件导出
        if len(drawio_files) == 1 and output:
            input_file = drawio_files[0]
            self._print_info("开始导出...")
            
            success = self._export_single_file(
                input_file=input_file,
                output_file=output,
                format=format,
                page_index=page_index,
                width=width,
                height=height,
                scale=scale,
                transparent=transparent,
                border=border,
                quiet=False
            )
            
            if not success:
                sys.exit(1)
            return
        
        # 批量导出
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._print_info(f"找到 {len(drawio_files)} 个 drawio 文件")
        self._print_info(f"输出目录: {output_path.absolute()}")
        print()
        
        success_count = 0
        failed_count = 0
        
        for i, drawio_file in enumerate(drawio_files, 1):
            input_path = Path(drawio_file)
            output_file = output_path / f"{input_path.stem}.{format}"
            
            self._print(f"[{i}/{len(drawio_files)}] {input_path.name}", 'BLUE')
            
            success = self._export_single_file(
                input_file=str(input_path),
                output_file=str(output_file),
                format=format,
                page_index=page_index,
                width=width,
                height=height,
                scale=scale,
                transparent=transparent,
                border=border,
                quiet=True
            )
            
            if success:
                success_count += 1
                file_size = output_file.stat().st_size
                if file_size >= 1024 * 1024:
                    size_str = f"{file_size / 1024 / 1024:.2f} MB"
                elif file_size >= 1024:
                    size_str = f"{file_size / 1024:.2f} KB"
                else:
                    size_str = f"{file_size} B"
                self._print(f"  ✓ {output_file.name} ({size_str})", 'GREEN')
            else:
                failed_count += 1
                self._print(f"  ✗ 导出失败", 'RED')
            
            print()
        
        # 总结
        self._print_step("导出完成！")
        self._print_info(f"成功: {success_count}")
        if failed_count > 0:
            self._print_warning(f"失败: {failed_count}")
        self._print_info(f"输出目录: {output_path.absolute()}")
    
    def batch_export(
        self,
        input_dir: str = ".",
        output_dir: str = "./output",
        format: str = "png",
        recursive: bool = False
    ):
        """
        批量导出目录中的所有 drawio 文件
        
        Args:
            input_dir: 输入目录（默认：当前目录）
            output_dir: 输出目录（默认：./output）
            format: 输出格式（默认：png）
            recursive: 递归查找子目录（默认：False）
        
        Examples:
            # 导出当前目录所有drawio文件
            python drawio_tool.py batch-export
            
            # 导出指定目录
            python drawio_tool.py batch-export --input-dir=./diagrams --output-dir=./images
            
            # 递归导出
            python drawio_tool.py batch-export --recursive
            
            # 导出为PDF
            python drawio_tool.py batch-export --format=pdf
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            self._print_error(f"输入目录不存在: {input_dir}")
            sys.exit(1)
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 查找drawio文件
        pattern = '**/*.drawio' if recursive else '*.drawio'
        drawio_files = list(input_path.glob(pattern))
        
        if not drawio_files:
            self._print_warning(f"在 {input_dir} 中未找到 .drawio 文件")
            return
        
        self._print_info(f"找到 {len(drawio_files)} 个 drawio 文件")
        print()
        
        # 批量导出
        success_count = 0
        failed_count = 0
        
        for i, drawio_file in enumerate(drawio_files, 1):
            output_file = output_path / f"{drawio_file.stem}.{format}"
            
            self._print(f"[{i}/{len(drawio_files)}] 正在导出: {drawio_file.name}", 'BLUE')
            
            try:
                self.export(
                    input_file=str(drawio_file),
                    output_file=str(output_file),
                    format=format
                )
                success_count += 1
            except SystemExit:
                failed_count += 1
                self._print_error(f"导出失败: {drawio_file.name}")
            
            print()
        
        # 总结
        self._print_step("批量导出完成！")
        self._print_info(f"成功: {success_count}")
        if failed_count > 0:
            self._print_warning(f"失败: {failed_count}")
        self._print_info(f"输出目录: {output_path.absolute()}")
    
    def check(self):
        """
        检查系统环境和依赖
        
        Examples:
            python drawio_tool.py check
        """
        self._print_info("系统环境检查")
        self._print_info("=" * 50)
        
        # 检查操作系统
        print(f"\n操作系统: {sys.platform}")
        
        # 检查Python版本
        print(f"Python 版本: {sys.version.split()[0]}")
        
        # 检查依赖命令
        print("\n依赖检查:")
        
        commands = {
            'xvfb-run': '虚拟显示环境',
            'drawio': 'DrawIO 应用程序',
            'fc-cache': '字体缓存工具',
            'fc-list': '字体列表工具'
        }
        
        for cmd, desc in commands.items():
            if self._command_exists(cmd):
                path = shutil.which(cmd)
                self._print(f"  ✓ {desc} ({cmd}): {path}", 'GREEN')
            else:
                self._print(f"  ✗ {desc} ({cmd}): 未安装", 'RED')
        
        # 检查字体
        print("\n字体检查:")
        ret, stdout, _ = self._run_cmd(['fc-list', ':lang=zh'], quiet=True)
        if ret == 0 and stdout:
            zh_fonts = len(stdout.strip().split('\n'))
            self._print(f"  ✓ 中文字体: 找到 {zh_fonts} 个", 'GREEN')
        else:
            self._print(f"  ✗ 中文字体: 未找到", 'RED')
        
        ret, stdout, _ = self._run_cmd(['fc-list'], quiet=True)
        if ret == 0 and 'emoji' in stdout.lower():
            self._print(f"  ✓ Emoji 字体: 已安装", 'GREEN')
        else:
            self._print(f"  ✗ Emoji 字体: 未安装", 'RED')
        
        print()
        self._print_info("=" * 50)
        
        # 建议
        if not self._command_exists('xvfb-run') or not self._command_exists('drawio'):
            print()
            self._print_warning("建议运行: sudo python drawio_tool.py install-deps")


def main():
    """主入口"""
    try:
        import fire
    except ImportError:
        print("错误: 未安装 fire 库")
        print("请运行: pip install fire")
        sys.exit(1)
    
    fire.Fire(DrawIOTool)


if __name__ == '__main__':
    main()

