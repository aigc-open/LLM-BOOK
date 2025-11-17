# LLM-BOOK：大模型优化技术学习手册

🚀 一个全面的大模型推理与训练优化技术学习资源库，包含 FlashInfer、Liger-Kernel、LMCache 等关键项目的深度学习指南。

## 📚 项目简介

本仓库专注于大模型（LLM）的性能优化技术，涵盖推理加速、训练优化、内存管理等核心领域。我们提供了详细的学习路径、实践指南和设备适配方案，帮助你从入门到精通。

## 🎯 包含内容

### 1️⃣ FlashInfer - 高性能推理加速库

FlashInfer 是一个专为大模型推理优化的 CUDA 库，实现了 FlashAttention 和 PageAttention 算法。

📖 **包含文档**：
- [仓库概述](./flashinfer/仓库概述.md) - FlashInfer 是什么，有什么用
- [项目介绍](./flashinfer/项目介绍.md) - 核心概念与技术创新
- [项目架构](./flashinfer/项目架构.md) - 代码组织与设计模式
- [快速上手](./flashinfer/快速上手.md) - 安装使用与实践示例
- [二次开发指南](./flashinfer/二次开发指南.md) - 扩展与定制开发
- [第三方设备集成指南](./flashinfer/第三方设备集成指南.md) - NPU/XPU 设备适配

🔗 **官方资源**：
- 官网: https://flashinfer.ai
- GitHub: https://github.com/flashinfer-ai/flashinfer
- 论文: https://arxiv.org/abs/2501.01005

**适合人群**：LLM 应用开发者、推理优化工程师、系统研究人员

---

### 2️⃣ Liger-Kernel - 高效训练内核库

Liger-Kernel 提供了一系列优化的训练算子，专注于提升大模型训练效率和内存使用。

📖 **包含文档**：
- [快速入门指南](./Liger-Kernel/快速入门指南.md) - 基础使用方法
- [项目深度分析](./Liger-Kernel/Liger-Kernel项目深度分析.md) - 技术原理解析
- [NPU适配实践指南](./Liger-Kernel/NPU适配实践指南.md) - 国产芯片适配

🔗 **官方资源**：
- GitHub: https://github.com/linkedin/Liger-Kernel

**适合人群**：LLM 训练工程师、算子优化开发者、AI 芯片适配工程师

---

### 3️⃣ LMCache - KV-Cache 分布式管理框架

LMCache 是一个用于管理和共享大模型 KV-Cache 的分布式系统，支持跨 GPU 共享，减少重复计算。

📖 **包含文档**：
- [框架深度解析](./LMCache/LMCache框架深度解析-架构与二次开发指南.md) - 架构设计与开发指南
- [实现细节与数据流转完全解析](./LMCache/LMCache实现细节与数据流转完全解析.md) - 核心实现原理
- [P2P-GPU通信深度解析](./LMCache/LMCache-P2P-GPU通信深度解析.md) - GPU 间通信机制
- [Python替代CUDA-Kernel可行性分析](./LMCache/LMCache-用Python替代CUDA-Kernel的可行性分析.md) - 性能对比分析
- [NPU和国产AI加速器适配指南](./LMCache/LMCache适配NPU和国产AI加速器完全指南.md) - 国产芯片适配

🔗 **官方资源**：
- GitHub: https://github.com/LMCache/LMCache

**适合人群**：分布式系统开发者、LLM 服务优化工程师、缓存系统研究人员

---

## 🗺️ 学习路径推荐

### 🔰 初学者路径
1. 从 FlashInfer 的《仓库概述》开始，了解推理优化基础
2. 阅读《快速上手》，动手实践基础示例
3. 学习 Liger-Kernel《快速入门指南》，了解训练优化
4. 探索 LMCache《框架深度解析》，理解缓存管理

**预计时间**：8-12 小时

### 👨‍💻 应用开发者路径
1. 快速浏览各项目的 README 和概述文档
2. 重点阅读《快速上手》和《入门指南》
3. 根据需求深入特定功能模块
4. 参考实践案例进行集成

**预计时间**：5-8 小时

### 🔧 系统开发者路径
1. 系统学习《项目架构》和《深度分析》
2. 深入《二次开发指南》，理解设计原理
3. 研读《实现细节》，掌握核心技术
4. 实践《设备适配指南》，进行移植开发

**预计时间**：15-25 小时

### 🎓 研究人员路径
1. 阅读所有技术深度解析文档
2. 研究各项目的技术创新点
3. 对比不同方案的设计权衡
4. 探索优化方向和研究课题

**预计时间**：20-30 小时

---

## 💡 核心技术点

### 推理优化
- ✅ FlashAttention / PageAttention 实现
- ✅ KV-Cache 管理与优化
- ✅ 批处理与动态 Batching
- ✅ 低精度推理（FP16/INT8）
- ✅ Tensor Core 加速

### 训练优化
- ✅ Fused Kernels（融合算子）
- ✅ 内存优化技术
- ✅ 梯度计算优化
- ✅ 混合精度训练

### 分布式与缓存
- ✅ 跨 GPU KV-Cache 共享
- ✅ P2P 通信优化
- ✅ 分布式缓存管理
- ✅ 序列化与反序列化

### 设备适配
- ✅ 华为昇腾 NPU 适配
- ✅ AMD/Intel GPU 支持
- ✅ 国产 AI 加速器适配
- ✅ CUDA 到其他平台的移植策略

---

## 🛠️ 技术栈

- **语言**: Python, C++, CUDA, C
- **框架**: PyTorch, CUDA Runtime
- **构建**: CMake, setuptools, pybind11
- **设备**: NVIDIA GPU, 华为昇腾, AMD GPU, Intel GPU
- **关键技术**: CUDA Kernels, Tensor Cores, 模板元编程, JIT 编译

---

## 📖 使用建议

### 📌 边学边练
- 每个示例都运行一遍，修改参数观察变化
- 尝试组合不同功能，理解交互效果
- 对比优化前后的性能差异

### 📌 查阅文档
- 遇到问题先查项目文档
- 参考 tests/ 目录下的测试用例
- 阅读源码注释和类型定义

### 📌 循序渐进
- 先掌握基础用法和核心概念
- 再尝试高级功能和性能调优
- 最后进行二次开发和设备移植

### 📌 积极交流
- 加入各项目的官方社区
- 在 GitHub Discussions 提问讨论
- 分享你的实践经验和优化心得

---

## 🚀 快速开始

### 选择你的起点：

| 你的目标 | 推荐文档 |
|---------|---------|
| 了解推理优化 | [FlashInfer 仓库概述](./flashinfer/仓库概述.md) |
| 快速使用推理加速 | [FlashInfer 快速上手](./flashinfer/快速上手.md) |
| 优化训练性能 | [Liger-Kernel 快速入门](./Liger-Kernel/快速入门指南.md) |
| 管理 KV-Cache | [LMCache 框架解析](./LMCache/LMCache框架深度解析-架构与二次开发指南.md) |
| 适配国产芯片 | [第三方设备集成指南](./flashinfer/第三方设备集成指南.md) |
| 深入系统开发 | [FlashInfer 项目架构](./flashinfer/项目架构.md) |

---

## 🤝 贡献与反馈

如果你发现文档中的问题或有改进建议：

1. 📝 提交 Issue 描述问题
2. 💬 在 Discussions 中讨论
3. 🔧 提交 Pull Request 贡献改进
4. ⭐ 给项目加星，分享给更多人

---

## 📜 许可证

本项目采用 MIT 许可证，详见 [LICENSE](./LICENSE) 文件。

---

## 📅 更新记录

- **2025-11**: 初始版本发布，包含 FlashInfer、Liger-Kernel、LMCache 完整学习资料

---

**祝学习愉快！如有问题欢迎交流！** 🎉

---

### ⭐ 如果这个项目对你有帮助，请给个 Star！