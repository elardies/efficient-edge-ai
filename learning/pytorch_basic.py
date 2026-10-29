import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Device:", device)

x = torch.tensor([1.0, 2.0, 3.0], device=device)
w = torch.tensor([0.5, 0.3, 0.2], device=device)

y = torch.dot(x, w)

print("Result:", y.item())
print("Tensor device:", y.device)

##
weight = torch.tensor(2.0, requires_grad=True)

loss = (weight - 5) ** 2

loss.backward()

print("Loss: ", loss.item())
print("Gradient: ", weight.grad.item())
