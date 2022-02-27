# 6.2. Convolutions for Images

## 6.2.1. The Cross-Correlation Operation

Implementation of cross-correlation operation.

```python
import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y
```

```python
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

```python
tensor([[19., 25.],
        [37., 43.]])
```


## 6.2.2. Convolutional Layers

```python
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

## 6.2.3. Object Edge Detection in Images

```python
X = torch.ones((6, 8))
X[:, 2:6] = 0
X
```

```python
tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.]])
```

```python
Y = corr2d(X, K)
Y
```

```python
tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])
```

We can now apply the kernel to the transposed image. 

As expected, it vanishes. The kernel K only detects vertical edges.

```python
corr2d(X.t(), K)
```


```python
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
```

## 6.2.4. Learning a Kernel

Now let us see whether we can learn the kernel that generated Y from X by looking at the input–output pairs only. 

We first construct a convolutional layer and initialize its kernel as a random tensor. 

Next, in each iteration, we will use the squared error to compare Y with the output of the convolutional layer. 

We can then calculate the gradient to update the kernel. 

For the sake of simplicity, in the following we use the built-in class for two-dimensional convolutional layers and ignore the bias.

```python
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Learning rate

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
```

```python
epoch 2, loss 10.459
epoch 4, loss 2.395
epoch 6, loss 0.664
epoch 8, loss 0.219
epoch 10, loss 0.081
```

Note that the error has dropped to a small value after 10 iterations. 

Now we will take a look at the kernel tensor we learned.

```python
conv2d.weight.data.reshape((1, 2))
```

```python
tensor([[ 0.9573, -1.0137]])
```

Indeed, the learned kernel tensor is remarkably close to the kernel tensor K we defined earlier.
