from MoBoAligner import MoBoAligner
import torch

# 设置随机种子以确保结果可复现
torch.manual_seed(1234)

# 初始化文本和mel嵌入向量
text_embeddings = torch.randn(2, 5, 10, requires_grad=True)  # 批量大小为2，文本令牌数为5，嵌入维度为10
mel_embeddings = torch.randn(2, 800, 10, requires_grad=True)  # 批量大小为2，mel帧数为800，嵌入维度为10
temperature_ratio = 0.5  # Gumbel噪声的温度比率

# 初始化MoBoAligner模型
aligner = MoBoAligner()

# 前向传播
gamma, expanded_text_embeddings = aligner(text_embeddings, mel_embeddings, temperature_ratio)

# 打印软对齐（gamma）的形状和扩展的文本嵌入向量
print("Soft alignment (gamma):")
print(gamma.shape)
print("Expanded text embeddings:")
print(expanded_text_embeddings)

# 反向传播测试
gamma.sum().backward()
print("Gradient for text_embeddings:")
print(text_embeddings.grad)
print("Gradient for mel_embeddings:")
print(mel_embeddings.grad)