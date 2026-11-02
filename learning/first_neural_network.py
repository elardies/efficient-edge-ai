import torch
from torch import nn

torch.manual_seed(42)

X = torch.linspace(-5, 5, 100).reshape(-1, 1)
y = 3 * X + 2

indices = torch.randperm(100)

train_idx = indices[:70]
val_idx = indices[70:85]
test_idx = indices[85:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


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
    prediction = model(X_train)
    loss = loss_fn(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch: {epoch + 1}: Loss: {loss.item():.6f}")

model.eval()

#Example inference using the trained model
with torch.no_grad():
    test_input = torch.tensor([[4.0]])
    prediction = model(test_input)

print("Predicted value:", prediction.item())
print("Expected value:", 14.0)

with torch.no_grad():
    val_predictions = model(X_val)
    val_loss = loss_fn(val_predictions, y_val)

print("Validation MSE:", val_loss.item())

with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = loss_fn(test_predictions, y_test)

print("Test MSE:", test_loss.item())