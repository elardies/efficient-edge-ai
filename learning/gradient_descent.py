import torch

weight = torch.tensor(2.0, requires_grad=True)
learning_rate = 0.1

for epoch in range(20):
    loss = (weight - 5) ** 2
    loss.backward()

    with torch.no_grad():
        weight -= learning_rate * weight.grad

    weight.grad.zero_()

    print(f"Epoch {epoch + 1}: Weight = {weight.item():.4f}, Loss = {loss.item():.4f}")