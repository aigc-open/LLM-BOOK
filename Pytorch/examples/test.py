import torch
import torch.fx as fx

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        # 当 x 是 Proxy 对象时:
        y = self.linear(x)      # 记录 call_module 节点
        z = torch.relu(y)       # 记录 call_function 节点  
        return z                # 记录 output 节点

# 符号追踪
model = SimpleModel()


traced = fx.symbolic_trace(model)

print("-"*20)
print(traced.graph)
print(traced.graph.python_code(root_module='self'))

# 修改计算图（例如：算子融合）
for node in traced.graph.nodes:
    if node.op == 'call_function' and node.target == torch.relu:
        # 将 linear + relu 融合成 linear + inplace relu
        with traced.graph.inserting_after(node):
            new_node = traced.graph.call_function(
                torch.relu_,  # inplace 版本
                args=(node.args[0],)
            )
            node.replace_all_uses_with(new_node)
        traced.graph.erase_node(node)

# 查看生成的图
print("-"*20)
print(traced.graph)
print(traced.graph.python_code(root_module='self'))