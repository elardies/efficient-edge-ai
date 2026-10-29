import torch

print("PyTorch version:", torch.__version__)
print("PyTorch working:", torch.rand(2, 3))


x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([0.5, 0.3, 0.2])

y = torch.dot(x, w)

print(y)