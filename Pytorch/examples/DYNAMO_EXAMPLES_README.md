# TorchDynamo 实战示例集

这些示例配合 `Pytorch/high/06_TorchDynamo实战调试指南.md` 使用，帮助你从零开始理解 TorchDynamo。

## 运行环境

```bash
# 确保安装了 PyTorch 2.0+
python -c "import torch; print(torch.__version__)"
```

## 示例列表

### 1. dynamo_01_simple.py - 最简单的例子

**学习目标**：理解 torch.compile 的基本用法

```bash
python dynamo_01_simple.py
```

**预期输出**：
```
==================================================
原始执行:
==================================================
结果: tensor([...])

==================================================
torch.compile 执行:
==================================================
结果: tensor([...])

结果是否一致: True
```

**关键点**：
- torch.compile 的基本用法
- 编译前后结果一致性

---

### 2. dynamo_02_debug.py - 开启调试日志

**学习目标**：查看 TorchDynamo 内部发生了什么

```bash
python dynamo_02_debug.py
```

**预期输出**：大量调试日志，包括：
- Frame Hook 拦截信息
- 字节码分析
- Guard 生成
- FX Graph 构建

**关键点**：
- 理解编译流程
- 理解缓存机制
- 理解 Guard 的作用

---

### 3. dynamo_03_bytecode.py - 查看字节码

**学习目标**：理解字节码和 TorchDynamo 的关系

```bash
python dynamo_03_bytecode.py
```

**预期输出**：
- Python 字节码反汇编
- TorchDynamo 如何处理字节码

**关键点**：
- 理解 Python 字节码
- 理解字节码指令到 FX Graph 的转换

---

### 4. dynamo_04_graph_break.py - Graph Break 分析

**学习目标**：理解什么会导致 Graph Break

```bash
python dynamo_04_graph_break.py
```

**预期输出**：
```
函数: with_print
  Graph 数量: 2
  Graph Break 数量: 1
  Break 原因:
    [1] call to print (side effect)

函数: with_item
  Graph 数量: 2
  Graph Break 数量: 1
  Break 原因:
    [1] call to tensor.item() (data-dependent control flow)

函数: no_break
  Graph 数量: 1
  Graph Break 数量: 0
  [√] 没有 Graph Break！
```

**关键点**：
- 识别 Graph Break
- 理解 Graph Break 的原因
- 如何避免 Graph Break

---

### 5. dynamo_05_full_debug.py - 完整调试流程

**学习目标**：完整观察一次编译过程

```bash
python dynamo_05_full_debug.py > debug_log.txt 2>&1
```

然后查看日志：
```bash
# 查看编译触发
grep "CONVERT_FRAME" debug_log.txt

# 查看字节码
grep -A 20 "Bytecode:" debug_log.txt

# 查看 Guard
grep -A 10 "Guard" debug_log.txt

# 查看生成的图
grep -A 30 "FX Graph" debug_log.txt
```

**关键点**：
- 完整的编译流程
- 缓存的使用
- 形状改变时的重新编译

---

## 学习路径

### Day 1: 基础理解
1. 运行 `dynamo_01_simple.py`
2. 理解 torch.compile 的基本用法
3. 对比编译前后的区别

### Day 2: 深入机制
1. 运行 `dynamo_02_debug.py`
2. 阅读日志输出
3. 理解编译流程的每个步骤

### Day 3: 字节码分析
1. 运行 `dynamo_03_bytecode.py`
2. 学习 Python 字节码
3. 理解字节码到 FX Graph 的转换

### Day 4: Graph Break
1. 运行 `dynamo_04_graph_break.py`
2. 理解什么会导致 Graph Break
3. 学习如何避免 Graph Break

### Day 5: 综合实践
1. 运行 `dynamo_05_full_debug.py`
2. 查看完整的日志
3. 尝试修改代码，观察变化

---

## 调试技巧

### 1. 开启详细日志

```python
import torch._dynamo as dynamo
import logging

torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True
```

### 2. 查看生成的代码

```python
torch._dynamo.config.output_code = True
```

### 3. 打印 Guard

```python
torch._dynamo.config.print_guards = True
```

### 4. 分析 Graph Break

```python
explanation = dynamo.explain(your_function)
print(f"Graph Break 数量: {explanation.graph_break_count}")
for reason in explanation.break_reasons:
    print(f"原因: {reason}")
```

### 5. 禁用缓存（调试时）

```python
torch._dynamo.config.cache_size_limit = 1
```

### 6. 重置缓存

```python
torch._dynamo.reset()
```

---

## 常见问题

### Q: 为什么第一次运行很慢？
A: 第一次需要编译（分析字节码、生成图、编译 kernel）。后续运行使用缓存，会非常快。

### Q: 如何确认使用了缓存？
A: 开启 `verbose=True`，会看到 `[Cache Hit]` 或类似的日志。

### Q: 什么时候会重新编译？
A: 当输入的形状、类型改变，或者控制流条件改变时（Guard 失败）。

### Q: Graph Break 影响性能吗？
A: 是的！Graph Break 会把图切成多个小图，降低融合优化的机会。应该尽量避免。

### Q: 如何查看生成的 FX Graph？
A: 设置 `torch._dynamo.config.output_code = True`，会打印生成的图代码。

---

## 下一步学习

完成这些示例后，建议：

1. 阅读 `Pytorch/high/05_TorchDynamo源码深度剖析.md`
2. 阅读 `Pytorch/high/06_TorchDynamo实战调试指南.md`
3. 学习 AOTAutograd（反向传播优化）
4. 学习 TorchInductor（代码生成）
5. 尝试自定义编译后端

---

## 实用资源

- [PyTorch 2.0 官方文档](https://pytorch.org/docs/stable/torch.compiler.html)
- [TorchDynamo GitHub](https://github.com/pytorch/pytorch/tree/main/torch/_dynamo)
- [PEP 523 - Frame Evaluation API](https://peps.python.org/pep-0523/)

---

**记住**：最好的学习方法是边运行、边调试、边修改代码！🚀

