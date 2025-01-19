import torch
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())

tensor2d = torch.tensor([[1, 2, 3], 
                         [4, 5, 6]])

# create a 3D tensor from a nested Python list
tensor3d = torch.tensor([[[1, 2, 0], [3, 4, 0]], 
                           [[5, 6, 0], [7, 8, 0]]])

print(tensor2d.shape)
print(tensor2d)
print(tensor2d.T)
print("--------------")
print(tensor3d.shape)
print(tensor3d)
print(tensor3d.mT)


import torch.nn.functional as F

y = torch.tensor([1.0])  # true label
x1 = torch.tensor([1.1]) # input feature
w1 = torch.tensor([2.2]) # weight parameter
b = torch.tensor([0.0])  # bias unit

z = x1 * w1 + b          # net input
a = torch.sigmoid(z)     # activation & output

loss = F.binary_cross_entropy(a, y)
print(loss)

print("--------------")
print("CREATING A NEURAL NETWORK")
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
                
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

model = NeuralNetwork(50, 3)
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)
print("model.layers[0].weight:", model.layers[0].weight)
print("model.layers[0].weight.shape:", model.layers[0].weight.shape)
print("--------------")
torch.manual_seed(123)

X = torch.rand((1, 50))
out = model(X)
print(out)

with torch.no_grad():
    out = model(X)
print(out)

with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)
print("--------------")
