# DrawIO 命令行工具

一个功能强大的 Python 命令行工具，用于管理 DrawIO 依赖和导出 drawio 文件。

## 🚀 特性

- ✅ 一键安装所有依赖（xvfb、中文字体、Emoji字体）
- ✅ 支持多种导出格式（PNG、JPG、SVG、PDF）
- ✅ 批量导出功能
- ✅ 丰富的导出参数（缩放、透明背景、页面选择等）
- ✅ 清晰的命令行接口（基于 Fire）
- ✅ 环境检查功能

## 📦 安装

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

或直接安装：

```bash
pip install fire
```

### 2. 安装系统依赖

```bash
sudo python drawio_tool.py install-deps
```

这会自动安装：
- xvfb（虚拟显示环境）
- 中文字体（WenQuanYi 微米黑、文泉驿正黑、思源黑体等）
- Emoji 字体（Noto Color Emoji、Symbola）

## 📖 使用方法

### 基本命令

```bash
# 查看帮助
python drawio_tool.py --help

# 查看子命令帮助
python drawio_tool.py export --help
```

### 1. 检查环境

```bash
python drawio_tool.py check
```

输出示例：
```
[INFO] 系统环境检查
==================================================

操作系统: linux
Python 版本: 3.9.2

依赖检查:
  ✓ 虚拟显示环境 (xvfb-run): /usr/bin/xvfb-run
  ✓ DrawIO 应用程序 (drawio): /usr/bin/drawio
  ✓ 字体缓存工具 (fc-cache): /usr/bin/fc-cache
  ✓ 字体列表工具 (fc-list): /usr/bin/fc-list

字体检查:
  ✓ 中文字体: 找到 28 个
  ✓ Emoji 字体: 已安装
```

### 2. 安装依赖

```bash
# 完整安装
sudo python drawio_tool.py install-deps

# 跳过 apt-get update
sudo python drawio_tool.py install-deps --skip-update
```

### 3. 导出文件

#### 单文件导出

```bash
# 基本导出
python drawio_tool.py export input.drawio --output=output.png

# 导出为 PDF
python drawio_tool.py export input.drawio --output=output.pdf --format=pdf

# 导出为 SVG
python drawio_tool.py export input.drawio --output=output.svg --format=svg

# 高清导出（2倍分辨率）
python drawio_tool.py export input.drawio --output=output@2x.png --scale=2

# 透明背景
python drawio_tool.py export input.drawio --output=output.png --transparent
```

#### 使用通配符批量导出（推荐）

```bash
# 导出当前目录所有 drawio 文件
python drawio_tool.py export "*.drawio"

# 导出指定目录的 drawio 文件
python drawio_tool.py export "diagrams/*.drawio" --output-dir=./images

# 递归导出所有子目录的 drawio 文件
python drawio_tool.py export "**/*.drawio" --output-dir=./output

# 匹配特定模式的文件
python drawio_tool.py export "page*.drawio" --output-dir=./pages

# 指定输出格式和参数
python drawio_tool.py export "*.drawio" --format=pdf --output-dir=./pdfs

# 高清批量导出
python drawio_tool.py export "**/*.drawio" --output-dir=./out --scale=2 --transparent
```

#### 高级选项

```bash
# 导出指定页面（第2页，索引从0开始）
python drawio_tool.py export input.drawio --output=page2.png --page-index=1

# 指定宽度和高度
python drawio_tool.py export input.drawio --output=output.png --width=1920 --height=1080

# 添加边框
python drawio_tool.py export input.drawio --output=output.png --border=10

# 组合使用多个参数
python drawio_tool.py export "*.drawio" \
  --output-dir=./out \
  --format=png \
  --scale=2 \
  --transparent \
  --border=20
```


## 🎯 使用场景

### 场景 1: 单文件高清导出

```bash
# 创建演示用的高清图片
python drawio_tool.py export presentation.drawio \
  --output=slide.png \
  --scale=2 \
  --transparent
```

### 场景 2: 批量导出当前项目

```bash
# 导出当前目录所有 drawio 文件
python drawio_tool.py export "*.drawio" --output-dir=./images
```

### 场景 3: 递归导出整个项目

```bash
# 递归导出所有子目录的 drawio 文件
python drawio_tool.py export "**/*.drawio" --output-dir=./output --scale=2
```

### 场景 4: 按模式匹配导出

```bash
# 只导出 page 开头的文件
python drawio_tool.py export "page*.drawio" --output-dir=./pages

# 导出特定目录的架构图
python drawio_tool.py export "docs/architecture/*.drawio" --output-dir=./images/arch
```

### 场景 5: 多页文档导出

```bash
# 导出第1页
python drawio_tool.py export document.drawio --output=page1.png --page-index=0

# 导出第2页
python drawio_tool.py export document.drawio --output=page2.png --page-index=1

# 使用循环导出所有页面
for i in {0..4}; do
  python drawio_tool.py export document.drawio \
    --output="page-$((i+1)).png" \
    --page-index=$i
done
```

### 场景 6: CI/CD 集成

```yaml
# .gitlab-ci.yml 示例
generate-diagrams:
  stage: build
  script:
    - pip install fire
    - sudo python drawio_tool.py install-deps --skip-update
    - python drawio_tool.py export "**/*.drawio" --output-dir=./output --scale=2
  artifacts:
    paths:
      - output/
```

### 场景 7: GitHub Actions 集成

```yaml
# .github/workflows/export-diagrams.yml
name: Export Diagrams
on: [push]
jobs:
  export:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: |
          pip install fire
          sudo apt-get update
          sudo python drawio_tool.py install-deps --skip-update
      - name: Export diagrams
        run: |
          python drawio_tool.py export "**/*.drawio" --output-dir=./output
      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: diagrams
          path: output/
```

## 📋 完整命令参考

### install-deps

安装所有系统依赖

**参数：**
- `--skip-update`: 跳过 apt-get update（可选）

**示例：**
```bash
sudo python drawio_tool.py install-deps
sudo python drawio_tool.py install-deps --skip-update
```

### export

导出 drawio 文件，支持通配符匹配

**参数：**
- `drawio_path`: drawio 文件路径，支持通配符（必需）
  - 单文件：`input.drawio`
  - 通配符：`*.drawio`、`page*.drawio`
  - 递归：`**/*.drawio`
  - 路径匹配：`docs/**/*.drawio`
- `--output`: 输出文件路径（单文件导出时使用，可选）
- `--output-dir`: 输出目录（批量导出时使用，默认：./output）
- `--format`: 输出格式（png/jpg/svg/pdf，默认：png）
- `--page-index`: 页面索引，从0开始（可选）
- `--width`: 输出宽度（像素，可选）
- `--height`: 输出高度（像素，可选）
- `--scale`: 缩放比例（可选）
- `--transparent`: 透明背景，仅PNG（可选）
- `--border`: 边框宽度（像素，可选）

**单文件导出示例：**
```bash
# 基本导出
python drawio_tool.py export input.drawio --output=output.png

# 高级选项
python drawio_tool.py export input.drawio --output=output.png --scale=2 --transparent
```

**批量导出示例：**
```bash
# 当前目录所有文件
python drawio_tool.py export "*.drawio"

# 递归所有子目录
python drawio_tool.py export "**/*.drawio" --output-dir=./output

# 指定模式和格式
python drawio_tool.py export "page*.drawio" --format=pdf --output-dir=./pdfs
```

### check

检查系统环境和依赖

**示例：**
```bash
python drawio_tool.py check
```

## 🔧 常见问题

### Q: 提示 "未安装 fire 库"

**A:** 运行以下命令安装：
```bash
pip install fire
```

### Q: 中文或 Emoji 显示为方框

**A:** 运行依赖安装命令：
```bash
sudo python drawio_tool.py install-deps
```

### Q: 提示 "请使用 root 权限运行"

**A:** 在命令前加 `sudo`：
```bash
sudo python drawio_tool.py install-deps
```

### Q: 提示 "drawio 未安装"

**A:** 需要先安装 DrawIO 桌面版：
- 下载地址: https://github.com/jgraph/drawio-desktop/releases
- 或使用包管理器安装

### Q: 如何指定特定页面导出？

**A:** 使用 `--page-index` 参数（索引从0开始）：
```bash
python drawio_tool.py export input.drawio --output=page1.png --page-index=0
python drawio_tool.py export input.drawio --output=page2.png --page-index=1
```

### Q: 如何使用通配符导出？

**A:** drawio_path 参数支持 glob 通配符：
```bash
# 当前目录所有 drawio 文件
python drawio_tool.py export "*.drawio"

# 递归所有子目录
python drawio_tool.py export "**/*.drawio" --output-dir=./output

# 特定模式
python drawio_tool.py export "page*.drawio" --output-dir=./pages
```

**注意**: 使用通配符时要用引号括起来，避免被 shell 展开。

## 💡 技巧

### 1. 创建别名简化命令

在 `~/.bashrc` 或 `~/.zshrc` 中添加：

```bash
alias drawio='python /path/to/drawio_tool.py'
alias drawio-export='python /path/to/drawio_tool.py export'
```

然后可以直接使用：
```bash
drawio export "*.drawio" --output-dir=./images
drawio check
```

### 2. 使用脚本批量导出多页文档

```bash
#!/bin/bash
# export-all-pages.sh

DRAWIO_FILE="document.drawio"
TOTAL_PAGES=5

for i in $(seq 0 $((TOTAL_PAGES - 1))); do
    python drawio_tool.py export "$DRAWIO_FILE" \
        --output="page-$((i+1)).png" \
        --page-index=$i
done
```

### 3. 在 Makefile 中集成

```makefile
# Makefile

.PHONY: diagrams clean install-deps

diagrams:
	python drawio_tool.py export "**/*.drawio" --output-dir=./docs/images

clean:
	rm -rf ./docs/images

install-deps:
	sudo python drawio_tool.py install-deps

# 生成高清版本
diagrams-hd:
	python drawio_tool.py export "**/*.drawio" \
		--output-dir=./docs/images \
		--scale=2 \
		--transparent
```

使用：
```bash
make diagrams      # 普通导出
make diagrams-hd   # 高清导出
make clean         # 清理
```

### 4. 监控文件变化自动导出

使用 `inotifywait` 监控文件变化：

```bash
#!/bin/bash
# watch-and-export.sh

while inotifywait -e modify,create *.drawio; do
    echo "检测到文件变化，开始导出..."
    python drawio_tool.py export "*.drawio" --output-dir=./images
done
```

## 🌟 完整示例

### 生成项目架构图

```bash
# 1. 检查环境
python drawio_tool.py check

# 2. 导出架构图（高清、透明背景）
python drawio_tool.py export architecture.drawio \
  --output=architecture@2x.png \
  --scale=2 \
  --transparent \
  --border=20

# 3. 同时生成 PDF 版本用于打印
python drawio_tool.py export architecture.drawio \
  --output=architecture.pdf \
  --format=pdf
```

### 批量生成文档图片

```bash
# 递归导出所有图表
python drawio_tool.py export "docs/**/*.drawio" \
  --output-dir=./docs/images \
  --scale=2
```

### 项目发布流程

```bash
#!/bin/bash
# release-diagrams.sh

echo "开始生成发布图表..."

# 1. 清理旧文件
rm -rf ./release/diagrams

# 2. 导出 PNG（网页用）
python drawio_tool.py export "**/*.drawio" \
  --output-dir=./release/diagrams/png \
  --format=png \
  --scale=2 \
  --transparent

# 3. 导出 PDF（打印用）
python drawio_tool.py export "**/*.drawio" \
  --output-dir=./release/diagrams/pdf \
  --format=pdf

# 4. 导出 SVG（编辑用）
python drawio_tool.py export "**/*.drawio" \
  --output-dir=./release/diagrams/svg \
  --format=svg

echo "图表生成完成！"
```

## 📝 系统要求

- Python 3.6+
- Linux (Debian/Ubuntu)
- DrawIO 桌面版
- Root 权限（仅安装依赖时需要）

## 📚 相关链接

- [DrawIO 官网](https://www.diagrams.net/)
- [DrawIO Desktop Releases](https://github.com/jgraph/drawio-desktop/releases)
- [Python Fire 文档](https://github.com/google/python-fire)

## 📄 许可证

MIT License

---

**提示**: 首次使用请先运行 `sudo python drawio_tool.py install-deps` 安装依赖。

