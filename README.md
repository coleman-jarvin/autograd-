Autograd

Working through PyTorch's official autograd tutorial — understanding how automatic differentiation drives neural network training.

---

## What is `torch.autograd`

`torch.autograd` is PyTorch's automatic differentiation engine. It computes gradients automatically by tracking operations on tensors through a dynamic computation graph built at runtime.

This is the mechanism behind backpropagation — every weight update in a neural network flows through autograd.

---

## Core Concepts

**`requires_grad`** — flags a tensor for gradient tracking. Any operation on it gets recorded.

**Computation Graph** — PyTorch builds a directed acyclic graph (DAG) of operations during the forward pass. Each node is an operation; leaves are input tensors.

**`.backward()`** — traverses the graph in reverse, computing gradients via the chain rule and accumulating them in each tensor's `.grad` attribute.

**`grad_fn`** — each non-leaf tensor stores the function that created it. This is how autograd knows how to differentiate backwards through each operation.

**`torch.no_grad()`** — context manager that disables gradient tracking. Used during inference and evaluation to reduce memory overhead.

---

## Model Used

ResNet18 pretrained on ImageNet — used to demonstrate a forward pass, loss computation, and backward pass on a real architecture.

```python
model = torchvision.models.resnet18(pretrained=True)
```

---

## Training Loop Structure

```
Forward pass     → compute prediction
Loss             → measure error
.backward()      → compute gradients
optimizer.step() → update weights
optimizer.zero_grad() → clear gradients for next step
```

---

## Stack

- Python 3.x
- PyTorch
- TorchVision

---
