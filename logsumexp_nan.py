import torch

# 输入包含inf的张量
x = torch.tensor([-float('inf'), -float('inf'), 1], requires_grad=True)
y = torch.tensor([-float('inf'), 1, 1], requires_grad=True)

# 计算logsumexp
output = torch.logsumexp(x + y, dim=0)

# 反向传播
with torch.autograd.detect_anomaly():
    output.backward()
print(x.grad)

######## again ########

# 输入包含inf的张量
x = torch.tensor([-float('inf'), -float('inf')], requires_grad=True)
y = torch.tensor([-float('inf'), 1], requires_grad=True)

# 计算logsumexp
output = torch.logsumexp(x + y, dim=0)

# 反向传播
with torch.autograd.detect_anomaly():
    output.backward()
print(x.grad)