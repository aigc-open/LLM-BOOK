# cuTile Python 中文教程

欢迎来到 cuTile Python 中文教程！本教程深入讲解 NVIDIA cuTile Python API 的核心概念和实践技巧。

## 📚 教程目录

### 基础篇

1. **[快速入门](quick-start.MD)**
   - cuTile 简介
   - 环境配置
   - 第一个 cuTile 程序
   - 基本概念

### 核心概念

2. **[执行模型](execution-model.MD)**
   - 线程层次结构
   - CTA（协作线程数组）
   - Grid 和 Block 组织
   - 线程同步机制

3. **[数据模型](data-model.MD)**
   - Tile 抽象
   - 数据布局
   - 多维数组处理
   - 数据类型系统

4. **[内存模型](memory-model.MD)**
   - Memory Order（内存顺序）
   - Memory Scope（内存范围）
   - 原子操作
   - 生产者-消费者模式
   - 内存同步最佳实践

### 高级优化

5. **[性能调优](performance-tuning.MD)** ⭐ 新增
   - 架构特定配置（ByTarget）
   - 加载/存储性能提示
   - latency 和 allow_tma 优化
   - 性能调优策略
   - 实战案例与最佳实践

## 🎯 学习路线

### 初学者路线
```
快速入门 → 执行模型 → 数据模型
```

### 进阶路线
```
内存模型 → 性能调优
```

### 完整路线
```
快速入门 → 执行模型 → 数据模型 → 内存模型 → 性能调优
```

## 🚀 快速导航

### 按主题查找

#### 性能优化
- [架构特定配置](performance-tuning.MD#架构特定配置bytarget)
- [内存访问优化](performance-tuning.MD#加载存储性能提示)
- [TMA 使用指南](performance-tuning.MD#allow_tmatma-使用控制)

#### 内存管理
- [内存同步](memory-model.MD#memory-order内存顺序)
- [原子操作](memory-model.MD#实战示例)
- [生产者-消费者](memory-model.MD#常见模式与最佳实践)

#### 基础概念
- [Tile 操作](data-model.MD)
- [线程管理](execution-model.MD)
- [内核启动](quick-start.MD)

## 📖 文档特点

✅ **中文讲解**：全中文深度解析，便于理解  
✅ **实战导向**：大量代码示例和实际案例  
✅ **系统完整**：从基础到高级，循序渐进  
✅ **图文并茂**：配合图表和可视化说明  
✅ **最佳实践**：总结常见陷阱和优化技巧  

## 🔗 相关资源

- [NVIDIA cuTile 官方文档](https://docs.nvidia.com/cuda/cutile/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## 💡 使用建议

1. **按顺序学习**：建议先阅读基础篇，再深入核心概念
2. **动手实践**：每个章节都有代码示例，建议运行并修改
3. **反复查阅**：作为参考手册，随时查找需要的内容
4. **结合官方文档**：本教程是官方文档的补充和深化

## 📝 更新日志

- **2025-12-29**: 新增《性能调优》专题
- **2025-12**: 完成《内存模型》专题
- **2025-12**: 完成基础专题（快速入门、执行模型、数据模型）

## 🤝 贡献

欢迎提交问题和改进建议！

---

**最后更新：** 2025-12-29  
**维护者：** LLM-BOOK 项目组

