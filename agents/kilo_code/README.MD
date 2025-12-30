# 🚀 Kilo Code AI Agent 使用指南

> 让 AI 成为你的编程伙伴，用自然语言完成编程任务！

## 💡 什么是 Kilo Code？

Kilo Code 是一个强大的 VS Code AI 编程助手，它能够：

- 🎯 **自然语言编程**：用中文或英文描述需求，AI 自动生成代码
- 🔍 **智能代码分析**：理解项目结构，提供上下文相关的建议
- 🛠️ **全流程协助**：从代码生成、调试、重构到文档编写
- 🤖 **自主执行任务**：AI 可以读写文件、执行命令、浏览网页
- 🌐 **MCP 协议支持**：扩展视觉理解、联网搜索等强大功能

## ✨ Kilo Code 能做什么？

### 1️⃣ 代码生成
**你只需描述：** "创建一个 Flask API，实现用户注册登录功能"  
**Kilo 会：** 生成完整的代码文件、数据库模型、路由处理

### 2️⃣ 代码重构
**你只需说：** "优化这个函数的性能"  
**Kilo 会：** 分析代码瓶颈，重写高效实现，并解释改进点

### 3️⃣ 调试修复
**你只需说：** "这个报错怎么解决"  
**Kilo 会：** 分析错误原因，提供修复方案，直接修改代码

### 4️⃣ 项目分析
**你只需问：** "这个项目是做什么的？"  
**Kilo 会：** 阅读项目文件，生成技术架构和功能说明

### 5️⃣ 文档生成
**你只需说：** "为这个模块写文档"  
**Kilo 会：** 生成 README、API 文档、使用示例

---

## 📦 快速开始

### 第一步：安装插件

1. 打开 VS Code
2. 按 `Ctrl+Shift+X` 打开扩展市场
3. 搜索 `Kilo Code AI Agent`
4. 点击 **Install** 安装
5. 安装完成后，信任开发者

![插件安装](./videos/01-kilo安装.png)

### 第二步：配置 API

Kilo Code 需要配置 AI 模型 API 才能使用。这里以 PH8.co 为例（支持 OpenAI 兼容的 API）。

**步骤 1：获取 API Key**

1. 访问 API 提供商网站（如 [ph8.co](https://ph8.co/)）
2. 注册并登录
3. 在密钥管理中生成 API Key

![获取密钥](./videos/02-ph8大模型密钥获取.png)

**步骤 2：在 Kilo Code 中配置**

1. 点击 Kilo Code 图标
2. 点击右上角 **Settings**
3. 配置以下参数：

| 配置项 | 示例值 |
|--------|---------|
| **API Provider** | `OpenAI Compatible` |
| **Base URL** | `https://ph8.co/v1` |
| **API Key** | 你的 API Key |
| **Model** | `claude-sonnet-4.5` |

![配置密钥](./videos/03-配置密钥供应商.png)

> ⚠️ 注意：Base URL 末尾不要有空格

**步骤 3：配置权限（可选但推荐）**

为了让 Kilo Code 更流畅地工作，建议配置自动审批权限：

在 Auto-Approve 部分，勾选以下权限：
- ✅ **Read** - 允许读取文件
- ✅ **Write** - 允许写入文件
- ✅ **Browser** - 允许浏览器操作
- ✅ **Execute** - 允许执行命令
- ✅ **MCP** - 允许使用 MCP 扩展

![配置权限](./videos/04-kilo配置系统读写mcp等权限.png)

> 💡 提示：只勾选你信任的操作。如果不勾选，每次操作都会弹窗请求授权。

**步骤 4：测试配置**

1. 点击 **Save** 保存
2. 在聊天框输入：`写一个 Hello World 程序`
3. AI 成功响应即配置成功 ✅

---

## 💡 提示词编写技巧（重点）

提示词就是你和 AI 沟通的语言。写好提示词，AI 才能准确理解你的需求。

### 📝 提示词基本原则

#### ✅ 原则 1：具体明确
**❌ 不好的提示词：**
```
写个登录功能
```

**✅ 好的提示词：**
```
创建一个 Flask 登录功能：
1. 用户通过邮箱和密码登录
2. 密码使用 bcrypt 加密
3. 登录成功返回 JWT token
4. 登录失败返回 401 状态码和错误信息
5. 添加登录次数限制（5次/小时）
```

**为什么好？** 明确了技术栈、具体要求、安全措施

---

#### ✅ 原则 2：提供上下文
**❌ 不好的提示词：**
```
这个函数有bug
```

**✅ 好的提示词：**
```
这个 calculate_discount 函数在处理会员折扣时有bug：
- 当用户是 VIP 时，折扣应该是 0.8（8折）
- 但实际计算结果不对
- 测试数据：原价 100，期望结果 80，实际结果 90

请帮我找出问题并修复
```

**为什么好？** 说明了问题现象、期望结果、实际结果

---

#### ✅ 原则 3：分步骤描述
**❌ 不好的提示词：**
```
做个电商网站
```

**✅ 好的提示词：**
```
第一步：创建用户模型
- 包含字段：id, username, email, password, created_at
- 使用 SQLAlchemy ORM
- password 需要自动加密

完成后，我们再继续做商品模型
```

**为什么好？** 任务清晰，AI 可以聚焦当前步骤

---

#### ✅ 原则 4：指定技术栈
**❌ 不好的提示词：**
```
写个 API 接口
```

**✅ 好的提示词：**
```
使用 FastAPI 创建一个 REST API 接口：
- 端点：POST /api/users
- 接收 JSON：{"name": "xxx", "email": "xxx"}
- 使用 Pydantic 进行数据验证
- 返回格式：{"success": true, "user_id": 123}
- 添加异常处理和日志记录
```

**为什么好？** 明确了框架、数据格式、验证方式

---

### 🎯 实战提示词模板

#### 模板 1：生成新代码
```
创建 [功能名称]：
技术栈：[列出使用的库/框架]
功能要求：
1. [具体需求1]
2. [具体需求2]
3. [具体需求3]
注意事项：[特殊说明]
```

**实际例子：**
```
创建一个数据爬虫：
技术栈：Python + requests + BeautifulSoup4
功能要求：
1. 爬取指定网页的新闻标题和链接
2. 处理翻页（最多5页）
3. 数据保存到 CSV 文件
4. 添加请求间隔（避免被封）
5. 添加异常处理和重试机制
注意事项：需要设置 User-Agent 伪装浏览器
```

---

#### 模板 2：优化代码
```
优化这段代码：
目标：[性能/可读性/安全性]
当前问题：[描述问题]
期望效果：[期望达到什么效果]
限制条件：[有什么限制]
```

**实际例子：**
```
优化这段代码：
目标：提升性能
当前问题：处理 10万条数据需要 30 秒，太慢
期望效果：希望能在 5 秒内完成
限制条件：不能改变输出结果，数据结构保持不变
```

---

#### 模板 3：修复 Bug
```
修复这个 Bug：
问题描述：[发生了什么]
复现步骤：[如何触发]
期望行为：[应该怎样]
实际行为：[实际怎样]
错误信息：[贴上报错]
```

**实际例子：**
```
修复这个 Bug：
问题描述：用户注册时邮箱验证失败
复现步骤：
1. 输入邮箱 test@example.com
2. 点击注册
3. 显示"邮箱格式错误"
期望行为：邮箱格式正确，应该通过验证
实际行为：总是提示格式错误
错误信息：无报错，但验证函数返回 False
```

---

### 📸 提示词实战演示

![聊天效果](./videos/05-chat效果.png)

---

## 🛠️ 如何用 Kilo Code 改代码（重点）

代码修改是日常开发中最频繁的操作。掌握这个技巧，效率翻倍！

### 方式一：选中代码 + 描述需求

这是最常用的方式！

**步骤：**
1. 在编辑器中**选中**要修改的代码
2. 在 Kilo Code 聊天框输入需求
3. AI 会针对选中的代码进行修改

---

#### 示例 1：性能优化

**原始代码：**
```python
def find_users(user_list, target_age):
    result = []
    for user in user_list:
        if user['age'] == target_age:
            result.append(user)
    return result
```

**选中代码后，输入提示词：**
```
优化这个函数的性能，数据量可能达到 10 万条
```

**Kilo 会输出：**
```python
def find_users(user_list, target_age):
    # 使用列表推导式，性能更好
    return [user for user in user_list if user.get('age') == target_age]
```

**AI 还会解释：**
- 列表推导式比循环 append 快 30%
- 使用 get() 避免 KeyError
- 如果数据量更大，建议使用数据库索引

---

#### 示例 2：添加错误处理

**原始代码：**
```python
def divide(a, b):
    return a / b
```

**选中代码后，输入提示词：**
```
添加错误处理：
1. 处理除零错误
2. 处理非数字类型
3. 添加日志记录
```

**Kilo 会输出：**
```python
import logging

def divide(a, b):
    """
    除法运算，包含完整的错误处理
    """
    try:
        # 类型检查
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError(f"参数必须是数字类型，得到 {type(a)}, {type(b)}")
        
        # 除零检查
        if b == 0:
            raise ValueError("除数不能为零")
        
        result = a / b
        logging.info(f"计算成功: {a} / {b} = {result}")
        return result
        
    except (TypeError, ValueError) as e:
        logging.error(f"计算失败: {e}")
        raise
```

---

#### 示例 3：重构代码结构

**原始代码（一个大函数）：**
```python
def process_order(order_data):
    # 验证订单
    if not order_data.get('user_id'):
        return {'error': 'Missing user_id'}
    if not order_data.get('items'):
        return {'error': 'No items'}
    
    # 计算价格
    total = 0
    for item in order_data['items']:
        total += item['price'] * item['quantity']
    
    # 应用折扣
    if order_data.get('coupon'):
        total = total * 0.9
    
    # 保存订单
    db.save_order(order_data)
    
    return {'success': True, 'total': total}
```

**选中代码后，输入提示词：**
```
重构这个函数：
1. 拆分成多个小函数
2. 提高可读性和可测试性
3. 每个函数只做一件事
```

**Kilo 会输出：**
```python
def validate_order(order_data):
    """验证订单数据"""
    if not order_data.get('user_id'):
        raise ValueError('Missing user_id')
    if not order_data.get('items'):
        raise ValueError('No items')
    return True

def calculate_total(items):
    """计算商品总价"""
    return sum(item['price'] * item['quantity'] for item in items)

def apply_discount(total, coupon):
    """应用优惠券"""
    if coupon:
        return total * 0.9
    return total

def process_order(order_data):
    """处理订单主流程"""
    try:
        # 验证
        validate_order(order_data)
        
        # 计算价格
        total = calculate_total(order_data['items'])
        total = apply_discount(total, order_data.get('coupon'))
        
        # 保存
        db.save_order(order_data)
        
        return {'success': True, 'total': total}
    except ValueError as e:
        return {'error': str(e)}
```

---

### 方式二：描述问题 + 贴代码

当代码太长或者在其他文件时使用。

**格式：**
```
我有这段代码：
```python
[贴上代码]
```

[描述要修改的内容]
```

---

#### 示例 4：修复 Bug

**输入提示词：**
```
我有这段代码：

```python
def get_user_age(user_id):
    user = db.query(User).filter(User.id == user_id)
    return user.age
```

问题：当用户不存在时会报错
请修复：找不到用户时返回 None
```

**Kilo 会输出：**
```python
def get_user_age(user_id):
    """
    获取用户年龄
    
    Args:
        user_id: 用户ID
        
    Returns:
        int: 用户年龄，如果用户不存在返回 None
    """
    user = db.query(User).filter(User.id == user_id).first()  # 添加 .first()
    if user is None:
        return None
    return user.age
```

---

### 方式三：增量修改

先让 AI 改一部分，再逐步完善。

**第一轮：**
```
添加参数验证
```

**第二轮：**
```
再添加日志记录
```

**第三轮：**
```
添加单元测试
```

这样可以更好地控制修改过程，每一步都清晰可见。

---

### 🎯 代码修改最佳实践

#### ✅ 做法 1：明确修改目标
```
优化性能 → 具体到哪个性能指标（速度/内存）
添加功能 → 具体要添加什么功能
修复 bug → 描述 bug 的表现和期望行为
```

#### ✅ 做法 2：一次改一件事
```
❌ 不要说：优化性能，添加日志，重构结构
✅ 应该说：先优化性能（完成后再说下一步）
```

#### ✅ 做法 3：提供约束条件
```
不要改变接口签名
保持向后兼容
不使用外部库
```

#### ✅ 做法 4：要求解释
```
解释一下为什么这样改
这样改有什么好处
还有其他优化方案吗
```

---



## 💡 使用技巧总结

### 🎯 核心要点

1. **提示词要具体明确**
   - ❌ "优化代码" → ✅ "优化性能，减少内存占用"
   - ❌ "修复bug" → ✅ "修复除零错误，添加异常处理"

2. **选中代码再提问**
   - 直接选中要修改的部分
   - AI 会针对性地处理
   - 避免改错地方

3. **分步骤进行**
   - 复杂任务拆分成多个小任务
   - 每一步都清晰可控
   - 出问题容易定位

4. **提供约束条件**
   - 说明哪些不能改
   - 明确性能要求
   - 指定技术栈

5. **要求 AI 解释**
   - "为什么这样改？"
   - "有没有更好的方案？"
   - "这样改有什么风险？"

---

## 📈 效率提升对比

| 任务 | 传统方式 | 使用 Kilo Code | 提升 |
|------|---------|----------------|------|
| 生成样板代码 | 30 分钟 | 2 分钟 | 15倍 ⚡ |
| 添加错误处理 | 20 分钟 | 1 分钟 | 20倍 ⚡ |
| 重构代码结构 | 60 分钟 | 5 分钟 | 12倍 ⚡ |
| 编写单元测试 | 40 分钟 | 3 分钟 | 13倍 ⚡ |
| 生成 API 文档 | 30 分钟 | 2 分钟 | 15倍 ⚡ |

**平均效率提升：10-15 倍！**

---

## 🔗 相关资源

- 🛠️ [Kilo Code 官方插件](https://marketplace.visualstudio.com/items?itemName=kilocode.kilo-code)
- 📘 [Kilo Code 使用文档](https://marketplace.visualstudio.com/items?itemName=kilocode.kilo-code)
- 💬 [问题反馈与讨论](https://github.com/kilocode)
- 🌐 API 提供商：[ph8.co](https://ph8.co/) 或其他 OpenAI 兼容服务

---

---

## 🎨 更多实战场景

### 场景 1：数据处理任务

**你的需求：**
```
读取 sales.csv，统计每个产品的销售额，生成柱状图
```

**Kilo 会：**
1. 生成 pandas 读取代码
2. 实现数据统计逻辑
3. 使用 matplotlib 绘图
4. 提供完整可运行脚本

---

### 场景 2：学习开源项目

**你的需求：**
```
分析这个开源项目的架构，帮我理解它是如何工作的
```

**Kilo 会：**
1. 阅读 README 和核心文件
2. 分析项目结构
3. 总结核心功能和技术栈
4. 解释关键实现逻辑

---

### 场景 3：自动化文档

**你的需求：**
```
为这个 API 模块生成完整文档
```

**Kilo 会：**
1. 分析所有函数和类
2. 生成 Markdown 文档
3. 包含参数说明和示例
4. 添加返回值类型说明

---

## 📋 快速参考速查表

### 常用提示词模板

| 场景 | 提示词模板 |
|------|-----------|
| **生成代码** | `创建 [功能]，使用 [技术栈]，要求：1. xxx 2. xxx` |
| **修改代码** | 选中代码 + `优化 [目标]，注意 [约束]` |
| **修复Bug** | `修复这个bug：[问题描述]，期望 [行为]，实际 [行为]` |
| **添加功能** | `给这段代码添加 [功能]，要求 [具体要求]` |
| **重构代码** | `重构这段代码：[目标]，要求 [约束条件]` |
| **理解代码** | `explain这段代码的 [某部分]，重点说明 [关注点]` |
| **生成测试** | `为这个函数生成单元测试，覆盖 [场景]` |

### API 配置参数

| 配置项 | 说明 |
|--------|------|
| **API Provider** | 选择 `OpenAI Compatible` |
| **Base URL** | API 端点地址（末尾不要有空格） |
| **API Key** | 从 API 提供商获取的密钥 |
| **Model** | 具体的模型名称 |

---

## 🚀 总结

掌握 Kilo Code 的核心要点：

### 1. 提示词是关键 🎯
- 具体描述需求（不要含糊）
- 提供完整上下文（技术栈、约束条件）
- 分步骤表达（复杂任务拆分）

### 2. 改代码很简单 ✏️
- **选中代码** + **描述需求** = 精准修改
- 一次改一件事，避免混乱
- 要求 AI 解释改动原因

### 3. 分步骤迭代 🔄
- 先实现基本功能
- 再添加错误处理
- 最后优化性能
- 每一步都可控

### 4. 理解 AI 的代码 📚
- 不要盲目接受
- 理解修改的逻辑
- 学习新的写法
- AI 是助手，不是替代品

---

## 🔗 相关资源

- 🛠️ [Kilo Code 官方插件](https://marketplace.visualstudio.com/items?itemName=kilocode.kilo-code)
- 📘 [Kilo Code 使用文档](https://marketplace.visualstudio.com/items?itemName=kilocode.kilo-code)
- 🌐 API 提供商示例：[ph8.co](https://ph8.co/) 或其他 OpenAI 兼容服务

---

<div align="center">

**💡 优秀的程序员不是写更多代码，而是用更聪明的方式解决问题！**

**祝你用 Kilo Code 写出更好的代码！🚀**

</div>

