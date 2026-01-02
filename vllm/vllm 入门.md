# vLLM 快速入门指南

## 目录
1. [vLLM 简介](#vllm-简介)
2. [核心技术：分页注意力机制](#核心技术分页注意力机制)
3. [安装部署](#安装部署)
4. [模型支持](#模型支持)
5. [使用方式](#使用方式)
6. [参数详解](#参数详解)
7. [实战示例](#实战示例)
8. [最佳实践](#最佳实践)

---

## vLLM 简介

vLLM 是一个用于快速高效部署大型语言模型（LLMs）的开源库。它特别优化了高吞吐量推理，使其在生产环境中部署语言模型时非常受欢迎。

### vLLM 解决的核心问题

LLMs 通常需要大量的计算和内存资源，尤其是在同时处理大量用户请求时。传统方法存在以下问题：
- 推理速度慢
- 硬件利用率低下
- KV缓存管理效率低
- 内存浪费严重

### vLLM 的主要特点

- **高吞吐量**：通过分页注意力机制提高内存利用率
- **易于使用**：兼容 OpenAI API 接口
- **灵活部署**：支持多种硬件和部署方式
- **广泛支持**：支持主流的开源模型

---

## 核心技术：分页注意力机制

**分页注意力机制（PagedAttention）** 是 vLLM 的核心创新。受操作系统中虚拟内存和分页概念的启发，它能够更有效地管理 LLMs 中注意力机制使用的内存（即 KV 缓存）。

### 工作原理

传统方法：将每个提示的 KV 缓存强制存储在一个长而连续的缓冲区中

分页注意力机制：
1. **页面池（Page Pool）**：KV 张量存储在 GPU 内存中的块中（例如，16 个令牌 × 隐藏维度）
2. **页面表（Page Table）**：每个序列保持一个小列表，将其逻辑令牌索引映射到物理页面
3. **注意力内核（Attention Kernel）**：当新的查询向量到达时，内核会：
   - 查找包含相关键值的页面
   - 将这些页面流式传输到片上寄存器/共享内存中
   - 计算 softmax 加权和

### 优势

- 页面不需要相邻存储，可以灵活分配
- 运行时可以密集地打包数千个请求
- 显著减少内存碎片
- 提高 GPU 利用率

---

## 安装部署

### 环境准备

支持的平台：
- Linux（推荐）
- macOS（支持 M1/M2 芯片）
- Windows（WSL2）

### 安装步骤

#### 1. 创建 Python 环境

```bash
# 创建工作目录
mkdir vllm_project
cd vllm_project

# 创建虚拟环境（Python 3.8+）
python3 -m venv venv

# 激活环境
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows
```

#### 2. 安装 vLLM

**标准安装（GPU）：**
```bash
pip install vllm
```

**从源码安装：**
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

**macOS 安装（M1/M2）：**
```bash
pip install vllm
```

#### 3. 验证安装

```bash
python -c "import vllm; print(vllm.__version__)"
```

---

## 模型支持

vLLM 支持大量主流开源模型，包括：

- **GPT 系列**：GPT-2, GPT-J, GPT-NeoX
- **LLaMA 系列**：LLaMA, LLaMA-2, LLaMA-3, Vicuna, Alpaca
- **Mistral 系列**：Mistral, Mixtral
- **其他模型**：
  - Phi 系列（Microsoft）
  - Qwen 系列（阿里）
  - Baichuan 系列
  - ChatGLM 系列
  - Falcon 系列

完整支持列表：[vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

---

## 使用方式

### 方式一：OpenAI 兼容 API 服务器

#### 启动服务器

```bash
vllm serve microsoft/phi-1_5 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```

**常用启动参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 服务器监听地址 | 0.0.0.0 |
| `--port` | 服务器端口 | 8000 |
| `--trust-remote-code` | 允许执行自定义模型代码 | False |
| `--tensor-parallel-size` | 张量并行大小（多GPU） | 1 |
| `--gpu-memory-utilization` | GPU 内存利用率 | 0.9 |
| `--max-model-len` | 最大序列长度 | 模型配置 |
| `--dtype` | 数据类型（auto/half/float16/bfloat16） | auto |

#### 使用 OpenAI SDK 调用

```python
from openai import OpenAI

# 配置客户端
client = OpenAI(
    api_key="EMPTY",  # vLLM 不需要 API key
    base_url="http://localhost:8000/v1"
)

# 文本补全
completion = client.completions.create(
    model="microsoft/phi-1_5",
    prompt="San Francisco is a",
    max_tokens=50,
    temperature=0.8,
    top_p=0.95
)

print(completion.choices[0].text)

# 聊天补全
chat_completion = client.chat.completions.create(
    model="microsoft/phi-1_5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7
)

print(chat_completion.choices[0].message.content)
```

### 方式二：vLLM 原生 Python API

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="microsoft/phi-1_5",
    trust_remote_code=True,
    tensor_parallel_size=1,  # GPU 数量
    gpu_memory_utilization=0.9
)

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# 批量推理
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

outputs = llm.generate(prompts, sampling_params)

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
```

### 方式三：LangChain 集成

```python
from langchain_community.llms import VLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 初始化 vLLM
llm = VLLM(
    model="microsoft/phi-1_5",
    trust_remote_code=True,
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8
)

# 简单调用
response = llm.invoke("What is the capital of France?")
print(response)

# 使用链
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(prompt=prompt, llm=llm)

result = chain.run("What is machine learning?")
print(result)
```

---

## 参数详解

### 模型初始化参数（LLM）

```python
llm = LLM(
    model="model_name",                    # 模型名称或路径
    tokenizer="tokenizer_name",            # 分词器（默认与模型相同）
    tokenizer_mode="auto",                 # 分词器模式：auto/slow
    trust_remote_code=False,               # 是否信任远程代码
    tensor_parallel_size=1,                # 张量并行大小（GPU数量）
    dtype="auto",                          # 数据类型
    quantization=None,                     # 量化方法：awq/gptq/squeezellm
    gpu_memory_utilization=0.9,            # GPU内存利用率（0-1）
    max_model_len=None,                    # 最大序列长度
    seed=0,                                # 随机种子
    download_dir=None,                     # 模型下载目录
    load_format="auto",                    # 加载格式
    revision=None,                         # 模型版本
)
```

**关键参数说明：**

- **`model`**：Hugging Face 模型名称或本地路径
- **`tensor_parallel_size`**：在多个 GPU 上分割模型（需要多张 GPU）
- **`gpu_memory_utilization`**：控制 GPU 内存使用，建议 0.85-0.95
- **`max_model_len`**：限制最大序列长度，减少内存占用
- **`quantization`**：启用量化以减少内存使用

### 采样参数（SamplingParams）

```python
sampling_params = SamplingParams(
    n=1,                          # 每个prompt生成的序列数
    best_of=None,                 # 生成best_of个序列，返回最好的n个
    presence_penalty=0.0,         # 存在惩罚（-2.0到2.0）
    frequency_penalty=0.0,        # 频率惩罚（-2.0到2.0）
    repetition_penalty=1.0,       # 重复惩罚（1.0表示不惩罚）
    temperature=1.0,              # 温度（0-2，越高越随机）
    top_p=1.0,                    # 核采样概率（0-1）
    top_k=-1,                     # Top-K采样（-1表示禁用）
    min_p=0.0,                    # 最小概率阈值
    use_beam_search=False,        # 是否使用束搜索
    length_penalty=1.0,           # 长度惩罚（束搜索）
    early_stopping=False,         # 早停（束搜索）
    stop=None,                    # 停止字符串列表
    stop_token_ids=None,          # 停止token ID列表
    ignore_eos=False,             # 是否忽略EOS token
    max_tokens=16,                # 最大生成token数
    logprobs=None,                # 返回log概率的token数
    prompt_logprobs=None,         # 返回prompt的log概率
    skip_special_tokens=True,     # 跳过特殊token
)
```

**核心采样参数：**

- **`temperature`**：
  - 0.0：贪心解码，确定性输出
  - 0.1-0.7：较确定的输出，适合事实性任务
  - 0.8-1.2：平衡创造性和连贯性
  - 1.5+：高随机性，创造性输出

- **`top_p`**：核采样，保留累计概率达到 p 的最小 token 集合
  - 0.9-0.95：推荐值，平衡多样性和质量

- **`top_k`**：只考虑概率最高的 k 个 token
  - 40-100：常用范围

- **`max_tokens`**：控制输出长度，影响响应时间

- **`repetition_penalty`**：
  - 1.0：无惩罚
  - 1.1-1.5：减少重复

### 服务器启动参数

```bash
vllm serve MODEL_NAME \
    --host HOST \                          # 监听地址
    --port PORT \                          # 端口号
    --trust-remote-code \                  # 信任远程代码
    --tensor-parallel-size N \             # 张量并行
    --pipeline-parallel-size N \           # 流水线并行
    --gpu-memory-utilization RATIO \       # GPU内存利用率
    --max-model-len LENGTH \               # 最大序列长度
    --dtype TYPE \                         # 数据类型
    --quantization METHOD \                # 量化方法
    --max-num-seqs NUM \                   # 最大并发序列数
    --max-num-batched-tokens NUM \         # 批处理token数
    --disable-log-requests \               # 禁用请求日志
    --served-model-name NAME               # 服务模型名称
```

---

## 实战示例

### 示例 1：本地部署 LLaMA-2-7B

```bash
# 启动服务（需要约14GB显存）
vllm serve meta-llama/Llama-2-7b-chat-hf \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096
```

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "解释什么是量子计算"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### 示例 2：批量推理优化

```python
from vllm import LLM, SamplingParams
import time

# 初始化模型
llm = LLM(
    model="microsoft/phi-1_5",
    trust_remote_code=True
)

# 准备大量提示词
prompts = [f"Write a story about {topic}" 
           for topic in ["space", "ocean", "forest", "city", "mountain"] * 20]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# 批量推理
start = time.time()
outputs = llm.generate(prompts, sampling_params)
end = time.time()

print(f"生成 {len(prompts)} 个响应耗时: {end-start:.2f} 秒")
print(f"吞吐量: {len(prompts)/(end-start):.2f} 个请求/秒")
```

### 示例 3：多 GPU 部署

```bash
# 使用 2 张 GPU
vllm serve meta-llama/Llama-2-13b-chat-hf \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9
```

### 示例 4：量化模型部署

```python
from vllm import LLM, SamplingParams

# 使用 AWQ 量化模型（减少显存需求）
llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-AWQ",
    quantization="awq",
    dtype="half",
    gpu_memory_utilization=0.85
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

prompts = ["Explain quantum computing in simple terms."]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### 示例 5：流式输出

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

stream = client.chat.completions.create(
    model="microsoft/phi-1_5",
    messages=[
        {"role": "user", "content": "写一首关于春天的诗"}
    ],
    stream=True,  # 启用流式输出
    max_tokens=200
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 最佳实践

### 性能优化

1. **合理设置 GPU 内存利用率**
   ```python
   # 单模型：0.9
   # 多模型/应用共享：0.7-0.8
   gpu_memory_utilization=0.9
   ```

2. **使用量化降低显存需求**
   ```bash
   # AWQ/GPTQ量化可将显存需求减半
   --quantization awq
   ```

3. **批处理优化**
   ```python
   # 批量处理多个请求提高吞吐量
   outputs = llm.generate(batch_prompts, sampling_params)
   ```

4. **多 GPU 并行**
   ```bash
   # 大模型使用张量并行
   --tensor-parallel-size 2
   ```

### 部署建议

1. **生产环境配置**
   - 使用反向代理（Nginx/Traefik）
   - 配置负载均衡
   - 启用监控和日志
   - 设置请求限流

2. **错误处理**
   ```python
   try:
       response = client.chat.completions.create(...)
   except Exception as e:
       print(f"请求失败: {e}")
   ```

3. **资源监控**
   - 监控 GPU 利用率（nvidia-smi）
   - 监控请求延迟和吞吐量
   - 设置内存和超时限制

### 常见问题

**Q: 显存不足怎么办？**
- 降低 `gpu_memory_utilization`
- 减小 `max_model_len`
- 使用量化模型
- 使用更小的模型

**Q: 如何提高推理速度？**
- 增加 GPU 数量使用张量并行
- 使用批处理
- 降低 `max_tokens`
- 调整 `max_num_batched_tokens`

**Q: 如何部署私有模型？**
```python
llm = LLM(model="/path/to/local/model")
```

**Q: 如何自定义停止词？**
```python
sampling_params = SamplingParams(
    stop=["<|endoftext|>", "\n\n", "###"]
)
```

---

## 总结

vLLM 通过创新的分页注意力机制，为大语言模型的部署提供了高效的解决方案。它具有以下优势：

✅ **高性能**：显著提高吞吐量和 GPU 利用率  
✅ **易用性**：兼容 OpenAI API，无缝迁移  
✅ **灵活性**：支持多种部署方式和模型  
✅ **可扩展**：支持多 GPU 和分布式部署  

无论是研究实验还是生产部署，vLLM 都是部署大语言模型的理想选择。

---

## 参考资源

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [支持的模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)

---

**更新时间**：2026年1月  
**作者**：admin

