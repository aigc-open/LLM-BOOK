# Liger-Kernel 项目文档

欢迎查阅 Liger-Kernel 项目分析文档！本目录包含了对 Liger-Kernel 项目的全面分析和使用指南。

## 📚 文档导航

### 1. [快速入门指南](./快速入门指南.md) ⭐ 推荐新手阅读

**适合人群**：想快速上手使用 Liger-Kernel 的开发者

**内容概览**：
- ✅ 三种使用方式对比
- ✅ 完整训练示例代码
- ✅ 常见问题快速解答
- ✅ 最佳实践建议

**阅读时间**：15-20 分钟

---

### 2. [NPU适配实践指南](./NPU适配实践指南.md) 🔧 实战指南 ⭐ 新增

**适合人群**：需要将 Liger-Kernel 适配到 NPU 的开发者

**内容概览**：
- ✅ 如何为 NPU 创建特化算子
- ✅ 保持客户使用方式完全不变
- ✅ 设备自动检测和切换
- ✅ 完整的代码实现示例
- ✅ 测试和性能分析方法

**核心解决方案**：通过条件 Monkey Patch 和设备感知模块，让客户代码零修改的同时享受 NPU 优化。

**阅读时间**：30-40 分钟

---

### 3. [Liger-Kernel项目深度分析](./Liger-Kernel项目深度分析.md) 📖 完整参考

**适合人群**：需要深入了解项目实现原理的开发者、研究人员

**内容概览**：

#### 第一部分：项目概述
- 项目作用与解决的问题
- 核心特性与支持的模型
- 三种使用方法的详细说明
- 性能提升数据

#### 第二部分：技术实现
- **算子注册机制**
  - Triton JIT 编译原理
  - PyTorch Autograd Function 封装
  - 与 PyTorch 的集成方式

- **Monkey Patch 实现机制**
  - 模块级 Patch
  - 实例级 Patch
  - 自动检测机制
  - 如何添加新的 Monkey Patch

#### 第三部分：使用流程
- 完整训练流程示例
- 使用方法对比
- 分布式训练（FSDP、DeepSpeed）
- 后训练（对齐）流程

#### 第四部分：NPU 兼容性分析
- 硬件支持现状
- 技术依赖分析
- 迁移挑战与可行性评估
- 迁移路径建议（理论分析）

> 💡 **实战指南**：如果 NPU 已支持 Triton，请阅读 [NPU适配实践指南](./NPU适配实践指南.md)

#### 第五部分：测试体系
- 测试框架介绍
- 正确性、性能、收敛性测试
- 运行测试的各种方法
- 编写自定义测试

**阅读时间**：60-90 分钟

---

## 🎯 根据需求选择文档

### 我想...

#### 快速开始使用 Liger-Kernel
👉 阅读：[快速入门指南](./快速入门指南.md)  
重点关注：「快速开始」和「完整训练示例」章节

#### 了解项目的核心价值
👉 阅读：[快速入门指南](./快速入门指南.md) - 「核心问答 Q1」  
或：[深度分析](./Liger-Kernel项目深度分析.md) - 「第 1 章」

#### 理解算子注册机制
👉 阅读：[深度分析](./Liger-Kernel项目深度分析.md) - 「第 2 章：算子注册机制」  
关键结论：**不是 Torch 注册机制，而是基于 Triton + Autograd**

#### 学习如何添加 Monkey Patch
👉 阅读：[深度分析](./Liger-Kernel项目深度分析.md) - 「第 3 章：Monkey Patch 实现机制」  
特别关注：「3.5 如何添加新的 Monkey Patch」

#### 了解完整的使用流程
👉 阅读：[深度分析](./Liger-Kernel项目深度分析.md) - 「第 4 章：项目使用流程」  
包含：训练、分布式、后训练的完整示例

#### 为 NPU 适配 Liger-Kernel（NPU 已支持 Triton）⭐
👉 阅读：[NPU适配实践指南](./NPU适配实践指南.md)  
**核心方案**：
- 创建 NPU 特化算子（`rms_norm_npu.py`）
- 使用设备感知的 Module 自动切换
- 客户代码完全不变！

#### 评估 NPU 迁移可行性（NPU 不支持 Triton）
👉 阅读：[深度分析](./Liger-Kernel项目深度分析.md) - 「第 5 章：NPU 兼容性分析」  
关键结论：
- ✅ Monkey Patch 机制可迁移
- ❌ Triton 算子需要重写
- ⚠️ 预计 3-6 个月工作量

#### 运行和编写测试
👉 阅读：[深度分析](./Liger-Kernel项目深度分析.md) - 「第 6 章：单元测试使用方法」  
包含：pytest 使用、测试类型、编写自定义测试

---

## 📋 六大核心问题速查

根据您的原始问题，这里是快速导航：

| 问题 | 快速答案 | 详细文档 |
|------|---------|---------|
| **1. 项目的作用与使用方法** | 提升 20% 速度，减少 60% 内存，一行代码集成 | [快速入门 - 快速开始](./快速入门指南.md#快速开始) <br> [深度分析 - 第1章](./Liger-Kernel项目深度分析.md#1-项目作用与解决的问题) |
| **2. 是 Torch 注册的算子吗** | **不是**，基于 Triton JIT + PyTorch Autograd | [快速入门 - Q2](./快速入门指南.md#q2-是-torch-注册的算子吗) <br> [深度分析 - 第2章](./Liger-Kernel项目深度分析.md#2-算子注册机制) |
| **3. 如何添加 Monkey Patch** | 三步：创建 patch 函数 → 注册 → 导出 | [快速入门 - Q3](./快速入门指南.md#q3-如何添加-monkey-patch) <br> [深度分析 - 3.5节](./Liger-Kernel项目深度分析.md#35-如何添加新的-monkey-patch) |
| **4. 整个项目的使用流程** | 安装 → Patch → 加载模型 → 训练 | [快速入门 - Q4](./快速入门指南.md#q4-整个项目使用流程) <br> [深度分析 - 第4章](./Liger-Kernel项目深度分析.md#4-项目使用流程) |
| **5. Monkey Patch 能否迁移到 NPU** | **Patch 机制可以**，算子需适配 | **NPU 支持 Triton**：[NPU适配指南](./NPU适配实践指南.md) ⭐<br>**NPU 不支持 Triton**：[深度分析 - 第5章](./Liger-Kernel项目深度分析.md#5-npu-兼容性分析)（需重写，3-6月）|
| **6. unittest 如何使用** | pytest 框架，支持正确性/性能/收敛性测试 | [快速入门 - Q6](./快速入门指南.md#q6-如何使用-unittest) <br> [深度分析 - 第6章](./Liger-Kernel项目深度分析.md#6-单元测试使用方法) |

---

## 🔍 代码示例索引

### 基本使用

```python
# 最简单的使用方式
from liger_kernel.transformers import AutoLigerKernelForCausalLM
model = AutoLigerKernelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
```

👉 更多示例见：[快速入门 - 三种使用方式](./快速入门指南.md#2-三种使用方式)

### Monkey Patch

```python
# 为新模型添加 Liger 优化
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
```

👉 详细实现见：[深度分析 - 3.5节](./Liger-Kernel项目深度分析.md#35-如何添加新的-monkey-patch)

### 分布式训练

```python
# FSDP 训练
training_args = TrainingArguments(
    fsdp="full_shard auto_wrap",
    fsdp_config={"fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"]},
)
```

👉 完整示例见：[深度分析 - 4.3节](./Liger-Kernel项目深度分析.md#43-分布式训练流程)

### 测试

```bash
# 运行所有测试
pytest test/

# 运行特定测试
pytest test/transformers/test_rms_norm.py
```

👉 详细命令见：[深度分析 - 6.3节](./Liger-Kernel项目深度分析.md#63-运行测试)

---

## 📊 性能数据速查

| 指标 | HuggingFace | Liger-Kernel | 提升 |
|------|------------|--------------|------|
| **训练速度** | 2000 tokens/s | 2400 tokens/s | +20% |
| **内存占用** | 40 GB (4K ctx) | 16 GB (4K ctx) | -60% |
| **最大上下文** | 4K (8B 模型) | 16K (8B 模型) | 4x ↑ |
| **后训练内存** | 80 GB (DPO) | 16 GB (DPO) | -80% |

> 测试条件：LLaMA 3-8B, Batch Size=8, bf16, 8×A100

---

## 🛠️ 支持的硬件平台

| 平台 | 状态 | 说明 |
|------|------|------|
| **NVIDIA GPU** | ✅ 完全支持 | 推荐使用 |
| **AMD GPU** | ✅ 完全支持 | ROCm 5.0+ |
| **Intel GPU** | ⚠️ 实验性支持 | XPU |
| **其他 NPU** | ❌ 需要移植 | 见「NPU 兼容性分析」 |

---

## 📚 支持的模型

完整列表（30+ 模型）：

- ✅ LLaMA 2/3/4
- ✅ Mistral / Mixtral
- ✅ Gemma 1/2/3
- ✅ Qwen 2/2.5/3 (Text & VL)
- ✅ Phi3 / Phi3.5
- ✅ Granite 3.0/3.1
- ✅ GLM-4
- ✅ InternVL3
- ... 更多

👉 完整列表见：[快速入门 - 支持的模型](./快速入门指南.md#支持的模型)

---

## 🧩 核心算子

### 训练算子

- RMSNorm / LayerNorm
- RoPE（位置编码）
- SwiGLU / GeGLU（激活函数）
- CrossEntropy（损失函数）
- **FusedLinearCrossEntropy**（融合损失，节省 80% 内存）

### 对齐算子

- DPO / ORPO / CPO
- SimPO / KTO
- JSD / KL Divergence

👉 详细说明见：[快速入门 - 核心算子](./快速入门指南.md#核心算子)

---

## ❓ 常见问题

### 会影响模型精度吗？
**不会**。所有算子计算精确，经过严格测试。

### 兼容 Flash Attention 吗？
**完全兼容**，可以同时使用。

### 推理阶段可以用吗？
**可以，但收益有限**。主要优化训练阶段。

### 支持量化吗？
**支持**，可以与 4-bit/8-bit 量化配合使用。

👉 更多问题见：[快速入门 - 常见问题](./快速入门指南.md#常见问题)

---

## 🔗 外部资源

- 🌐 **官方文档**：https://linkedin.github.io/Liger-Kernel/
- 💻 **GitHub 仓库**：https://github.com/linkedin/Liger-Kernel
- 📄 **技术论文**：https://arxiv.org/pdf/2410.10989
- 📝 **官方博客**：https://www.linkedin.com/blog/engineering/open-source/liger-kernel-open-source-ecosystem-for-efficient-llm-training
- 🎥 **CUDA MODE 演讲**：https://youtu.be/gWble4FreV4
- 💬 **Discord 社区**：https://discord.gg/gpumode

---

## 🆕 最新更新

### 2025-11-07 v1.1
- ✨ 新增 [NPU适配实践指南](./NPU适配实践指南.md)
- 📝 解决 NPU 已支持 Triton 场景下的算子适配问题
- 🎯 提供保持客户代码不变的完整解决方案

### 2025-11-07 v1.0
- 📚 初始版本：完整的项目分析文档
- 📖 包含快速入门、深度分析、测试指南

---

## 📝 文档信息

- **创建日期**：2025-11-07
- **当前版本**：v1.1
- **作者**：基于 Liger-Kernel 源码分析生成

---

## 🤝 贡献

如果您发现文档中有错误或需要补充的内容，欢迎提 Issue 或 PR！

---

## 📖 推荐阅读顺序

### 新手入门（20 分钟）
1. [快速入门指南](./快速入门指南.md) - 完整阅读
2. 运行一个简单示例
3. 查阅「常见问题」章节

### 进阶学习（90 分钟）
1. [深度分析 - 第1章](./Liger-Kernel项目深度分析.md#1-项目作用与解决的问题) - 项目概述
2. [深度分析 - 第2章](./Liger-Kernel项目深度分析.md#2-算子注册机制) - 算子机制
3. [深度分析 - 第3章](./Liger-Kernel项目深度分析.md#3-monkey-patch-实现机制) - Monkey Patch
4. [深度分析 - 第4章](./Liger-Kernel项目深度分析.md#4-项目使用流程) - 使用流程

### 高级应用
1. [深度分析 - 第5章](./Liger-Kernel项目深度分析.md#5-npu-兼容性分析) - NPU 迁移
2. [深度分析 - 第6章](./Liger-Kernel项目深度分析.md#6-单元测试使用方法) - 测试框架
3. 阅读源码：`src/liger_kernel/`
4. 贡献代码

---

**祝您使用愉快！** 🚀

