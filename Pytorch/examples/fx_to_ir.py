import torch
import torch.fx as fx
from typing import Callable, List, Dict, Any
from dataclasses import dataclass

    
# ============================================================
# 第一部分：模拟 IR 节点类型
# ============================================================
@dataclass
class ExprNode:
    """表达式树节点基类"""
    pass

@dataclass
class Load(ExprNode):
    """加载操作"""
    buffer: str
    index: str

@dataclass
class Constant(ExprNode):
    """常量"""
    value: float
    
@dataclass
class BinOp(ExprNode):
    """二元操作"""
    op: str
    lhs: ExprNode
    rhs: ExprNode

@dataclass
class UnaryOp(ExprNode):
    """一元操作"""
    op: str
    arg: ExprNode

class ops:
    """符号化操作（返回表达式节点，不执行计算）"""
    @staticmethod
    def load(buffer: str, index: str):
        """"
        符号化加载
        """
        return Load(buffer=buffer, index=index)
    @staticmethod
    def constant(value: float):
        """常量"""
        return Constant(value=value)
    @staticmethod
    def add(lhs: ExprNode, rhs: ExprNode):
        """
        符号化加法
        """
        return BinOp(op='add', lhs=lhs, rhs=rhs)
    @staticmethod
    def mul(lhs: ExprNode, rhs: ExprNode):
        """
        符号化乘法
        """
        return BinOp(op='mul', lhs=lhs, rhs=rhs)
    @staticmethod
    def maximum(lhs: ExprNode, rhs: ExprNode):
        """
        符号化 maximum
        """
        return BinOp(op='maximum', lhs=lhs, rhs=rhs)
    @staticmethod
    def neg(arg: ExprNode):
        """
        符号化取负
        """
        return UnaryOp(op='neg', arg=arg)
    @staticmethod
    def exp(arg: ExprNode):
        """
        符号化指数
        """
        return UnaryOp(op='exp', arg=arg)
    @staticmethod
    def truediv(lhs: ExprNode, rhs: ExprNode):
        """
        符号化除法
        """
        return BinOp(op='truediv', lhs=lhs, rhs=rhs)
    
    @staticmethod
    def relu(arg: ExprNode):
        """
        符号化 ReLU
        """
        return UnaryOp(op='relu', arg=arg)
    @staticmethod
    def sigmoid(arg: ExprNode):
        """
        """
        return UnaryOp(op='sigmoid', arg=arg)

# ============================================================
# 第二部分：IR 节点类型
# ============================================================

class IRNode:
    """IR 节点基类"""
    def get_device(self):
        return self.device
    def get_dtype(self):
        return self.dtype
    def get_size(self):
        return self.ranges

class InputBuffer(IRNode):
    """输入缓冲区"""
    def __init__(self, name: str, device: str, dtype: str, size: List[int]):
        self.name = name
        self.device = device
        self.dtype = dtype
        self.ranges = size
    def __repr__(self):
        return f"InputBuffer(name='{self.name}', shape={self.ranges})"
class Pointwise(IRNode):
    """逐点操作"""
    def __init__(self, device: str, dtype: str, inner_fn: Callable, ranges: List[int]):
        self.device = device
        self.dtype = dtype
        self.inner_fn = inner_fn
        self.ranges = ranges
    def __repr__(self):
        return f"Pointwise(device='{self.device}', dtype='{self.dtype}', shape={self.ranges})"
class Reduction(IRNode):
    """归约操作"""
    def __init__(self, device: str, dtype: str, inner_fn: Callable, ranges: List[int], reduction_ranges: List[int], reduction_type: str):
        self.device = device
        self.dtype = dtype
        self.inner_fn = inner_fn
        self.ranges = ranges
        self.reduction_ranges = reduction_ranges
        self.reduction_type = reduction_type
    def __repr__(self):
        return f"Reduction(device='{self.device}', dtype='{self.dtype}', shape={self.ranges}, reduction_ranges={self.reduction_ranges}, reduction_type='{self.reduction_type}')"


class TensorBox:
    """张量包装器"""
    def __init__(self, data: IRNode):
        self.data = data
        
    def __repr__(self):
        return f"TensorBox({self.data})"
    
    @staticmethod
    def create(data):
        if isinstance(data, TensorBox):
            return data
        return TensorBox(data)
    
    def get_device(self):
        return self.data.get_device()
    def get_dtype(self):
        return self.data.get_dtype()
    def get_size(self):
        return self.data.get_size()
    
    
# ============================================================
# 第三部分：Lowering 函数注册
# ============================================================

lowerings: Dict[Any, Callable] = {}


def register_lowering(aten_op):
    """装饰器：注册 lowering 函数"""
    def decorator(fn):
        lowerings[aten_op] = fn
        return fn
    return decorator


@register_lowering('aten.add')
def add_tensor(x: TensorBox, y):
    """
    符号化加法
    y 可以是 TensorBox 或常量（int/float）
    """
    def inner_fn(idx):
        x_val = ops.load(x.data.name if isinstance(x.data, InputBuffer) else 'buf', idx)
        
        # 判断 y 是 TensorBox 还是常量
        if isinstance(y, TensorBox):
            y_val = ops.load(y.data.name if isinstance(y.data, InputBuffer) else 'buf', idx)
        else:
            # y 是常量（int/float）
            y_val = ops.constant(float(y))
        
        return ops.add(x_val, y_val)
    
    return TensorBox.create(Pointwise(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size())))
    
@register_lowering('aten.mul')
def mul_tensor(x: TensorBox, y):
    """
    符号化乘法
    y 可以是 TensorBox 或常量（int/float）
    """
    def inner_fn(idx):
        x_val = ops.load(x.data.name if isinstance(x.data, InputBuffer) else 'buf', idx)
        
        # 判断 y 是 TensorBox 还是常量
        if isinstance(y, TensorBox):
            y_val = ops.load(y.data.name if isinstance(y.data, InputBuffer) else 'buf', idx)
        else:
            # y 是常量（int/float）
            y_val = ops.constant(float(y))
        
        return ops.mul(x_val, y_val)
    
    return TensorBox.create(Pointwise(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size())))
@register_lowering('aten.relu')
def relu_tensor(x: TensorBox):
    """
    符号化 ReLU
    """
    def inner_fn(idx):
        x_val = ops.load(x.data.name if isinstance(x.data, InputBuffer) else 'buf', idx)
        return ops.relu(x_val)
    return TensorBox.create(Pointwise(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size())))
    
@register_lowering('aten.sigmoid')
def sigmoid_tensor(x: TensorBox):
    """
    符号化 Sigmoid
    """
    def inner_fn(idx):
        x_val = ops.load(x.data.name if isinstance(x.data, InputBuffer) else 'buf', idx)
        return ops.sigmoid(x_val)
    return TensorBox.create(Pointwise(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size())))
    
def linearize_expr(expr: ExprNode) -> list:
    """
    将表达式树线性化为操作序列
    
    返回: [(op_name, details, var_name), ...]
    """
    operations = []
    var_counter = [0]  # 使用列表以便在嵌套函数中修改
    var_map = {}  # 表达式节点 -> 临时变量名
    
    def visit(node):
        """后序遍历表达式树"""
        if id(node) in var_map:
            return var_map[id(node)]
        
        if isinstance(node, Load):
            var_name = f"tmp{var_counter[0]}"
            var_counter[0] += 1
            operations.append(('load', f"'{node.buffer}', {node.index}", var_name))
            var_map[id(node)] = var_name
            return var_name
        
        elif isinstance(node, Constant):
            var_name = f"tmp{var_counter[0]}"
            var_counter[0] += 1
            operations.append(('constant', f"{node.value}, dtype", var_name))
            var_map[id(node)] = var_name
            return var_name
        
        elif isinstance(node, BinOp):
            lhs_var = visit(node.lhs)
            rhs_var = visit(node.rhs)
            var_name = f"tmp{var_counter[0]}"
            var_counter[0] += 1
            operations.append((node.op, f"{lhs_var}, {rhs_var}", var_name))
            var_map[id(node)] = var_name
            return var_name
        
        elif isinstance(node, UnaryOp):
            arg_var = visit(node.arg)
            var_name = f"tmp{var_counter[0]}"
            var_counter[0] += 1
            operations.append((node.op, arg_var, var_name))
            var_map[id(node)] = var_name
            return var_name
        
        else:
            return "unknown"
    
    visit(expr)
    return operations

    
class MockGraphLowering:
    def __init__(self, gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
        self.gm = gm
        self.graph = gm.graph
        self.example_inputs = example_inputs
        
        # 核心数据结构：FX Node → IR Node 的映射
        self.env: Dict[fx.Node, TensorBox] = {}
        
    def run(self):
        """运行 lowering 过程,遍历fx graph,调用对应的lowering函数"""
        input_idx = 0
        for node in self.graph.nodes:
            if node.op == "placeholder":
                self.placeholder(node, input_idx)
                input_idx += 1
            elif node.op == "call_function":
                self.call_function(node)
            elif node.op == "call_method":
                self.call_method(node)
            elif node.op == "output":
                self.output(node)
            else:
                raise NotImplementedError(f"Unknown op: {node.op}")
            
    def placeholder(self, node: fx.Node, idx: int):
        """处理输入占位符"""
        example_value = self.example_inputs[idx]
        input_buffer = InputBuffer(
            name=node.name,
            device=str(example_value.device),
            dtype=str(example_value.dtype),
            size=list(example_value.shape))
        self.env[node] = TensorBox.create(input_buffer)
        
    def call_function(self, node: fx.Node):
        """处理函数调用"""
        lowering_fn = self._get_node_fn(node)
        args = self._get_node_args(node)
        result = lowering_fn(*args)
        self.env[node] = result
        
    def call_method(self, node: fx.Node):
        """处理方法调用"""
        lowering_fn = self._get_node_fn(node)
        args = self._get_node_args(node)
        result = lowering_fn(*args)
        self.env[node] = result
        
    def output(self, node: fx.Node):
        """处理输出"""
        # output 节点不需要存储到 env，只需要记录输出是哪个节点
        # 不添加到 env，避免在 visualize_ir 中处理
        pass
        
    def _get_node_fn(self, node: fx.Node):
        """获取节点函数"""
        target_name = "aten." + str(node.target.__name__)
        return lowerings[target_name]
        
        
    def _get_node_args(self, node: fx.Node):
        """获取节点参数"""
        args = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                args.append(self.env[arg])
            else:
                args.append(arg)
        return args
        
    def ir_pre(self):
        """可视化 IR Graph - 先收集到列表，最后统一打印"""
        lines = []  # 存储所有输出行
        
        # 构建依赖关系图
        node_index = {}
        for idx, (fx_node, _) in enumerate(self.env.items()):
            node_index[fx_node] = idx
        
        for op_id, (fx_node, tensor_box) in enumerate(self.env.items()):
            ir_node = tensor_box.data
            # lines.append(f"\n{'='*80}")
            # lines.append(f"op{op_id}: {type(ir_node).__name__}")
            # lines.append(f"{'='*80}")
            
            # 1. 依赖关系
            dependencies = []
            unmet_deps = []
            for arg in fx_node.args:
                if isinstance(arg, fx.Node) and arg in self.env:
                    if arg in node_index and node_index[arg] < op_id:
                        dependencies.append(arg.name)
                    else:
                        unmet_deps.append(arg.name)
            
            lines.append(f"op{op_id}.writes = ['{fx_node.name}']")
            lines.append(f"op{op_id}.met_dependencies = {dependencies}")
            lines.append(f"op{op_id}.unmet_dependencies = {unmet_deps}")
            
            # 2. 输出信息
            lines.append(f"op{op_id}.outputs = [")
            lines.append(f"    {fx_node.name}: {type(ir_node).__name__}")
            lines.append(f"]")
            
            # 3. 布局信息
            if hasattr(ir_node, 'device') and hasattr(ir_node, 'ranges'):
                lines.append(f"    {fx_node.name}.layout = FixedLayout(")
                lines.append(f"        device='{ir_node.device}',")
                lines.append(f"        dtype={ir_node.dtype},")
                lines.append(f"        size={ir_node.ranges}")
                lines.append(f"    )")
            
            # 4. 用户信息（哪些节点使用这个输出）
            users = []
            for other_node, _ in self.env.items():
                if fx_node in other_node.args:
                    users.append(other_node.name)
            if users:
                lines.append(f"    {fx_node.name}.users = {users}")
            lines.append(f"]")
            
            # 5. 循环体（计算逻辑）
            if isinstance(ir_node, Pointwise):
                lines.append(f"\nop{op_id}.group.device = {ir_node.device}")
                lines.append(f"op{op_id}.group.iteration = ({ir_node.ranges}, ())")
                lines.append(f"op{op_id}.sizes = ({ir_node.ranges}, [])")
                
                # 详细的计算逻辑
                lines.append(f"\nclass op{op_id}_loop_body:")
                lines.append(f"    var_ranges = {{idx: {ir_node.ranges}}}")
                lines.append(f"    def body(self, ops):")
                
                # 符号执行 inner_fn，展开表达式序列
                try:
                    expr = ir_node.inner_fn('idx')
                    operations = linearize_expr(expr)
                    for i, (op_name, details, var_name) in enumerate(operations):
                        lines.append(f"        {var_name} = ops.{op_name}({details})")
                    if operations:
                        last_var = operations[-1][2]
                        lines.append(f"        return {last_var}")
                except Exception as e:
                    lines.append(f"        # (表达式解析失败: {e})")
                    
            elif isinstance(ir_node, InputBuffer):
                lines.append(f"\nop{op_id}.type = InputBuffer")
                lines.append(f"op{op_id}.name = '{ir_node.name}'")
                lines.append(f"op{op_id}.shape = {ir_node.ranges}")
            
            lines.append(f"]")
        
        # 统一打印所有行
        for line in lines:
            print(line)
        
        # 返回列表，方便后续处理
        return lines


    def ir_post(self):
        """可视化融合后的 IR Graph（模拟 ir_post_fusion.txt）"""
        lines = []
        
        # 步骤1：执行融合，将可融合的节点分组
        fused_groups = self._fuse_nodes()
        
        # 步骤2：生成融合后的 IR 描述
        for group_id, group in enumerate(fused_groups):
            # 获取组的第一个和最后一个节点
            first_node = group[0]
            last_node = group[-1]
            first_data = self.env[first_node].data
            
            # lines.append(f"\n{'='*80}")
            # lines.append(f"op{group_id}: SchedulerNode(ComputedBuffer)")
            # lines.append(f"{'='*80}")
            
            # 1. 依赖关系
            dependencies = []
            for arg in first_node.args:
                if isinstance(arg, fx.Node) and arg not in group:
                    dependencies.append(arg.name)
            
            lines.append(f"op{group_id}.writes = [MemoryDep('{last_node.name}', c0, {{c0: ...}})]")
            lines.append(f"op{group_id}.unmet_dependencies = []")
            lines.append(f"op{group_id}.met_dependencies = {dependencies}")
            
            # 2. 输出信息
            lines.append(f"op{group_id}.outputs = [")
            lines.append(f"    {last_node.name}: ComputedBuffer")
            
            if isinstance(first_data, Pointwise):
                lines.append(f"    {last_node.name}.layout = FixedLayout(")
                lines.append(f"        device='{first_data.device}',")
                lines.append(f"        dtype={first_data.dtype},")
                lines.append(f"        size={first_data.ranges},")
                lines.append(f"        stride=[...]")
                lines.append(f"    )")
                
                # 用户信息
                users = []
                for other_group in fused_groups:
                    if other_group == group:
                        continue
                    for node in other_group:
                        if last_node in node.args:
                            users.append(f"NodeUser(node=SchedulerNode(name='op{fused_groups.index(other_group)}'), can_inplace=False)")
                            break
                
                if users:
                    lines.append(f"    {last_node.name}.users = [")
                    for user in users:
                        lines.append(f"        {user},")
                    lines.append(f"    ]")
            elif isinstance(first_data, InputBuffer):
                lines.append(f"    {first_node.name}.layout = FixedLayout(")
                lines.append(f"        device='{first_data.device}',")
                lines.append(f"        dtype={first_data.dtype},")
                lines.append(f"        size={first_data.ranges}")
                lines.append(f"    )")
            
            lines.append(f"]")
            
            # 3. 融合后的循环体
            if isinstance(first_data, Pointwise):
                lines.append(f"op{group_id}.group.device = {first_data.device}")
                lines.append(f"op{group_id}.group.iteration = ({first_data.ranges}, ())")
                lines.append(f"op{group_id}.sizes = ({first_data.ranges}, [])")
                
                # 生成融合后的 loop_body
                lines.append(f"\nclass op{group_id}_loop_body:")
                lines.append(f"    var_ranges = {{idx: {first_data.ranges}}}")
                lines.append(f"    def body(self, ops):")
                
                # 如果是融合组，展开所有操作
                if len(group) > 1:
                    lines.append(f"        # 融合了 {len(group)} 个操作: {' -> '.join([n.name for n in group])}")
                
                # 展开融合组中的所有操作
                last_var = None
                for node in group:
                    tensor_box = self.env[node]
                    if isinstance(tensor_box.data, Pointwise):
                        lines.append(f"        # --- {node.name} ---")
                        try:
                            expr = tensor_box.data.inner_fn('idx')
                            operations = linearize_expr(expr)
                            for op_name, details, var_name in operations:
                                lines.append(f"        {var_name} = ops.{op_name}({details})")
                                last_var = var_name
                        except Exception as e:
                            lines.append(f"        # (表达式解析失败: {e})")
                
                if last_var:
                    lines.append(f"        return {last_var}")
                
            elif isinstance(first_data, InputBuffer):
                lines.append(f"\nop{group_id}.type = InputBuffer")
                lines.append(f"op{group_id}.name = '{first_data.name}'")
                lines.append(f"op{group_id}.shape = {first_data.ranges}")
            
            lines.append(f"]")
        
        # 统一打印
        for line in lines:
            print(line)
        
        return lines
    
    def _fuse_nodes(self):
        """
        简单的融合逻辑：将连续的 Pointwise 操作融合
        
        融合规则：
        1. 只融合 Pointwise 操作
        2. 形状必须相同
        3. 设备必须相同
        4. 必须是生产者-消费者关系（单一使用）
        """
        nodes = list(self.env.items())
        fused_groups = []
        visited = set()
        
        for i, (node_i, tensor_box_i) in enumerate(nodes):
            if node_i in visited:
                continue
            
            # 创建新的融合组
            group = [node_i]
            visited.add(node_i)
            
            # 尝试向后融合
            current_node = node_i
            for j in range(i + 1, len(nodes)):
                node_j, tensor_box_j = nodes[j]
                
                if node_j in visited:
                    continue
                
                # 检查是否可以融合
                if self._can_fuse_with_group(group, node_j, tensor_box_j):
                    group.append(node_j)
                    visited.add(node_j)
                    current_node = node_j
                else:
                    # 无法继续融合，结束当前组
                    break
            
            fused_groups.append(group)
        
        return fused_groups
    
    def _can_fuse_with_group(self, group, candidate_node, candidate_box):
        """判断节点是否可以融合到组中"""
        last_node = group[-1]
        last_box = self.env[last_node]
        
        # 检查 1: 必须是 Pointwise 操作
        if not isinstance(last_box.data, Pointwise):
            return False
        if not isinstance(candidate_box.data, Pointwise):
            return False
        
        # 检查 2: 必须是直接依赖关系（candidate 依赖 last）
        if last_node not in candidate_node.args:
            return False
        
        # 检查 3: last_node 只能被 candidate_node 使用（单一消费者）
        user_count = 0
        for other_node, _ in self.env.items():
            if last_node in other_node.args:
                user_count += 1
                if user_count > 1:
                    return False
        
        # 检查 4: 形状必须相同
        if last_box.data.ranges != candidate_box.data.ranges:
            return False
        
        # 检查 5: 设备必须相同
        if last_box.data.device != candidate_box.data.device:
            return False
        
        return True

# ============================================================
# 第四部分：测试
# ============================================================

class CustomModel(torch.nn.Module):
    def forward(self, x):
        y = x * 2
        z = y + 1
        w = torch.sigmoid(z)
        return w

def demo_fx_to_ir():
    model = CustomModel()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    example_input = torch.randn(4, 64, device=device)
    traced = fx.symbolic_trace(model)
    
    print("="*80)
    print("步骤 1: 得到 Pytorch FX Graph")
    print("="*80)
    print(traced.graph)
    
    # 运行 lowering
    lowering = MockGraphLowering(traced, [example_input])
    lowering.run()
    
    # 可视化融合前的 IR（ir_pre_fusion）
    print("\n" + "="*80)
    print("步骤 2: ir_pre_fusion - GraphLowering 输出（未融合）")
    print("="*80)
    ir_pre_lines = lowering.ir_pre()
    
    # 可视化融合后的 IR（ir_post_fusion）
    print("\n" + "="*80)
    print("步骤 3: ir_post_fusion - Scheduler 融合后输出")
    print("="*80)
    ir_post_lines = lowering.ir_post()
    
    # 统计对比
    print("\n" + "="*80)
    print("融合效果对比")
    print("="*80)
    
    # 统计融合前的节点数
    pre_nodes = len([l for l in ir_pre_lines if l.startswith('op') and '.writes' in l])
    post_nodes = len([l for l in ir_post_lines if l.startswith('op') and '.writes' in l])
    
    print(f"融合前节点数: {pre_nodes}")
    print(f"融合后节点数: {post_nodes}")
    
    if post_nodes < pre_nodes:
        fusion_rate = ((pre_nodes - post_nodes) / pre_nodes) * 100
        print(f"融合率: {fusion_rate:.1f}% ({pre_nodes - post_nodes} 个节点被融合)")
        print(f"性能提升: 减少了 {pre_nodes - post_nodes} 次 kernel 启动")
    
    print("="*80)

if __name__ == '__main__':
    demo_fx_to_ir()