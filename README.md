# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: JEEVANESH S

### Register Number: 212222243002
```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(70)

# Generate noisy data from 1 to 70
X = torch.linspace(1, 70, 70).reshape(-1, 1)
e = torch.randint(-8, 9, (70, 1), dtype=torch.float)
y = 2 * X + 1 + e

plt.scatter(X.numpy(), y.numpy(), color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

torch.manual_seed(59)
model = Model(1, 1)

initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()

print("\nName:Jeevanesh")
print("Register No: 212222243002")
print(f'Initial Weight: {initial_weight:.8f}, Initial Bias: {initial_bias:.8f}\n')

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

epochs = 70
losses = []

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_function(y_pred, y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print(f'epoch: {epoch:2} loss: {loss.item():10.8f}, '
          f'weight: {model.linear.weight.item():10.8f}, '
          f'bias: {model.linear.bias.item():10.8f}')

plt.plot(range(epochs), losses, color='blue')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Curve')
plt.show()

final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()

print("\nName:Jeevanesh")
print("Register No: 212222543002")
print(f'\nFinal Weight: {final_weight:.8f}, Final Bias: {final_bias:.8f}')

x1 = torch.tensor([X.min().item(), X.max().item()])
y1 = x1 * final_weight + final_bias

plt.scatter(X.numpy(), y.numpy(), label="Original Data")
plt.plot(x1.numpy(), y1.numpy(), 'r', label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()

x_new = torch.tensor([[120.0]])
y_new_pred = model(x_new).item()

print("\nName:Jeevanesh")
print("Register No: 212222243002")
print(f"\nPrediction for x = 120: {y_new_pred:.8f}")

# Save the trained model
torch.save(model.state_dict(), 'linear_regression_model.pth')
print("\nModel saved as 'linear_regression_model.pth'")

```
### Dataset Information
Include screenshot of the generated data

### OUTPUT
![image](https://github.com/user-attachments/assets/4070a821-fb42-4025-b8b8-2f9da14bc862)
![image](https://github.com/user-attachments/assets/c83b3f7c-dd39-4eee-a378-27ff12ac06f9)
![image](https://github.com/user-attachments/assets/3bcd0731-bcf4-48a6-99f5-3c2653f953c8)


### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/bea4c44d-d88d-488a-b859-7d7a2352fbb5)

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
