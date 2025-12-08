# PyTorch 深度学习框架详解

## 学习指南

本系列文档将带你从零开始深入理解 PyTorch 2.0 的核心技术，特别是围绕 `torch.compile` 的一整套编译栈技术。

### 适合人群
- PyTorch 初学者和进阶用户
- 想要深入理解 PyTorch 编译技术的开发者
- 需要为自定义硬件（如国产芯片）适配 PyTorch 的工程师

## 章节目录

### 第零章：[技术对比与选型指南](./00_技术对比与选型指南.md) 新增
- 各技术的优缺点详细对比
- 能解决什么问题
- 如何选择合适的技术
- 适用场景分析
- 决策树和最佳实践

### 第一章：[PyTorch 2.0 新特性概览](./01_PyTorch2.0新特性概览.md)
- PyTorch 2.0 的重大变革
- torch.compile 的核心价值
- 性能提升原理
- 使用示例

### 第二章：[TorchScript 详解](./02_TorchScript详解.md)
- 什么是 TorchScript
- 为什么需要 TorchScript
- TorchScript 的两种模式：trace 和 script
- 实际应用场景
- 局限性

### 第三章：[TorchFX 图捕获技术](./03_TorchFX图捕获技术.md)
- TorchFX 的设计理念
- 符号追踪（Symbolic Tracing）
- IR（中间表示）详解
- 图变换和优化
- 与 TorchScript 的区别

### 第四章：[Lazy Tensor 延迟执行机制](./04_LazyTensor延迟执行机制.md)
- Lazy Tensor 的概念
- 延迟执行的优势
- LazyTensor 的工作流程
- 在不同后端的应用
- XLA 集成

### 第五章：[Torch Dynamo 动态图编译](./05_TorchDynamo动态图编译.md)
- Dynamo 的核心作用
- 字节码捕获技术
- Guards 机制
- 动态图到静态图的转换
- 源码分析

### 第六章：[AOTAutograd 自动微分](./06_AOTAutograd自动微分.md)
- Ahead-of-Time Autograd 原理
- 前向和反向图的生成
- 与传统 Autograd 的区别
- 联合编译优化
- 实现细节

### 第七章：[PyTorch 调度机制](./07_PyTorch调度机制.md)
- Dispatcher 的架构设计
- 多后端支持原理
- DispatchKey 详解
- 算子注册机制
- 自定义算子实现

### 第八章：[国产芯片适配实战指南](./08_国产芯片适配实战.md)
- torch.compile 适配流程
- 自定义后端开发
- 算子库映射
- 性能优化策略
- 完整适配案例

## 技术栈关系图

```
PyTorch 2.0 架构
|
|-- torch.compile (用户接口)
| |
| |-- Torch Dynamo (图捕获层)
| | |-- 捕获 Python 字节码，构建动态图
| |
| |-- AOTAutograd (自动微分层)
| | |-- 提前生成前向/反向计算图
| |
| |-- TorchFX (图表示层)
| | |-- IR 中间表示和图变换
| |
| |-- 后端编译器
| |-- TorchInductor (默认后端)
| |-- TorchScript
| |-- ONNX Runtime
| |-- 自定义后端 (如国产芯片)
|
|-- PyTorch Dispatcher (底层分发)
|-- 多后端算子调度
```

## 学习路线建议

### 快速入门（推荐新手）
**第 0 章（技术对比） -> 第 1 章（torch.compile）**
- 先看技术对比，了解全貌
- 学会使用 torch.compile
- 这就能解决 80% 的性能问题！

### 阶段一：基础理解（第 0-3 章）
先理解 PyTorch 2.0 的整体架构和 TorchScript、TorchFX 等基础技术。

### 阶段二：深入原理（第 4-7 章）
学习 Lazy Tensor、Dynamo、AOTAutograd 和 Dispatcher 的工作原理。

### 阶段三：实战应用（第 8 章）
将所学知识应用到国产芯片适配的实际场景中。

## 前置知识要求

- 基本的 Python 编程能力
- 了解神经网络基础概念
- 有一定的 PyTorch 使用经验（知道如何训练简单模型）

## 重要概念速查

| 概念 | 简单解释 | 详细章节 |
| ------ | ---------- | ---------- |
| torch.compile | PyTorch 2.0 的核心编译接口 | 第 1 章 |
| TorchScript | 将 Python 代码转为可优化的静态图 | 第 2 章 |
| TorchFX | 图捕获和变换工具 | 第 3 章 |
| Lazy Tensor | 延迟执行的 Tensor 系统 | 第 4 章 |
| Dynamo | 动态捕获 Python 执行图 | 第 5 章 |
| AOTAutograd | 提前计算自动微分图 | 第 6 章 |
| Dispatcher | 算子到硬件后端的分发机制 | 第 7 章 |

## 快速问题定位

### 我想要...

**️ 加速现有模型**
-> 看第 0 章和第 1 章，使用 `torch.compile`

**️ C++ 部署模型**
-> 看第 2 章，使用 TorchScript

**️ 自定义图优化**
-> 看第 3 章，使用 TorchFX

**️ 在 TPU 上训练**
-> 看第 4 章，使用 PyTorch/XLA

**️ 理解 torch.compile 原理**
-> 看第 5-6 章，学习 Dynamo 和 AOTAutograd

**️ 适配国产芯片**
-> 看第 7-8 章，学习 Dispatcher 和适配方法

**️ 不知道选哪个**
-> 看第 0 章技术对比！

## 性能提升速查

| 技术 | 训练加速 | 推理加速 | 易用性 | 推荐度 |
| ------ | --------- | --------- | -------- | -------- |
| **torch.compile** | 20-50% | 30-100%+ |
| **TorchScript** | 10-20% | 20-40% |
| **TorchFX** | 自定义 | 自定义 |
| **Lazy Tensor** | 15-40% | 20-50% |

## 参考资源

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch 2.0 发布博客](https://pytorch.org/get-started/pytorch-2.0/)
- [TorchDynamo 设计文档](https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361)
- [PyTorch Dispatcher RFC](https://github.com/pytorch/rfcs/blob/master/RFC-0001-pytorch-dispatcher.md)

## 文档说明

- 每个章节都包含：理论讲解、代码示例、图示说明、实践练习
- 代码示例均可直接运行（需要安装 PyTorch 2.0+）
- 涉及源码时会标注具体的文件路径，方便查阅
- **所有文档都是为小白设计，通俗易懂**

## 特别推荐

### 对于新手
**必看：第 0 章 + 第 1 章**
- 第 0 章告诉你每个技术的优缺点
- 第 1 章教你立即上手 torch.compile
- 其他章节按需学习

### 对于希望适配芯片的工程师
**按顺序全部学习**
- 第 0 章：了解技术全貌
- 第 1-7 章：理解底层原理
- 第 8 章：实战适配

### 对于性能优化工程师
**重点：第 0, 1, 3, 5, 6 章**
- 第 0 章：技术选型
- 第 1 章：torch.compile 使用
- 第 3 章：图优化技术
- 第 5-6 章：编译原理

---

** 开始学习之旅吧！**

推荐路线：
1. [第零章：技术对比与选型](./00_技术对比与选型指南.md) <- 从这里开始！
2. [第一章：PyTorch 2.0 新特性](./01_PyTorch2.0新特性概览.md)
3. 继续后续章节...

**小贴士：** 如果你是小白，先看第 0 章和第 1 章，就能解决大部分问题了！
