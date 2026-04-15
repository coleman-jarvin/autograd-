import torch 
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import SGD



model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)


prediction = model(data) # forward pass

loss = (prediction - labels).sum() 
loss.backward() # backward pass

## load optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9)

optimizer.step() # gradient descent

## differentiation 

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3 * a**3 - b**2 
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

#check if collected graidents are coreect 
print(9*a**2 == a.grad)
print(-2*b == b.grad)

x = torch.rand(5,5)
y = torch.rand(5,5)
z = torch.rand((5,5), requires_grad=True)

a = x = y 
print(f"Does 'a' require gradients?:{a.requires_grad}")
b = x + z
print(f"Does 'b' require gradients?:{b.requires_grad}")


model = resnet18(weights=ResNet18_Weights.DEFAULT)

## freeze network params

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 10)

# Optimize only the classifier

optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)




