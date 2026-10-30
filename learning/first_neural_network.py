import torch
from torch import nn

torch.manual_seed(42)

X = torch.linspace(-5, 5, 100).reshape(-1, 1)
y = 3 * X + 2

print("X:", X.shape)
print("y:", y.shape)
print("First five inputs", X[:5])
print("First five outputs", y[:5])

model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1),)

print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    prediction = model(X)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch: {epoch + 1}: Loss: {loss.item():.6f}")

model.eval()

with torch.no_grad():
    test_input = torch.tensor([[4.0]])
    prediction = model(test_input)

print("Predicted value:", prediction.item())
print("Expected value:", 14.0)