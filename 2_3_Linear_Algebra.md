# 2.3. Linear Algebra

## 2.3.1. Scalars

```python
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```python
(tensor(5.), tensor(6.), tensor(1.5000), tensor(9.))
```

```python
x[3]
```
```python
tensor(3)
```

### 2.3.2.1. Length, Dimensionality, and Shape

```python
len(x)
```

```python
4
```

```python
x.shape
```
```python
torch.Size([4])
```

## 2.3.3. Matrices

```python
A = torch.arange(20).reshape(5, 4)
A
```

```python
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]])
```

```python
A.T
```

```python
tensor([[ 0,  4,  8, 12, 16],
        [ 1,  5,  9, 13, 17],
        [ 2,  6, 10, 14, 18],
        [ 3,  7, 11, 15, 19]])
```

Here we define a symmetric matrix **B**.

```python
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```python
tensor([[1, 2, 3],
        [2, 0, 4],
        [3, 4, 5]])
```

Now we compare B with its transpose.

```python
B == B.T
```

```python
tensor([[True, True, True],
        [True, True, True],
        [True, True, True]])
```

## 2.3.4. Tensors

Tensors will become more important when we start working with images, which arrive as n-dimensional arrays with 3 axes corresponding to the height, width, and a channel axis for stacking the color channels (red, green, and blue). For now, we will skip over higher order tensors and focus on the basics.

```python
X = torch.arange(24).reshape(2, 3, 4)
X
```

```python
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
```


## 2.3.5. Basic Properties of Tensor Arithmetic

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```python
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]]),
 tensor([[ 0.,  2.,  4.,  6.],
         [ 8., 10., 12., 14.],
         [16., 18., 20., 22.],
         [24., 26., 28., 30.],
         [32., 34., 36., 38.]]))
```

```python
A * B
```

```python
tensor([[  0.,   1.,   4.,   9.],
        [ 16.,  25.,  36.,  49.],
        [ 64.,  81., 100., 121.],
        [144., 169., 196., 225.],
        [256., 289., 324., 361.]])
```


Multiplying or adding a tensor by a scalar also does not change the shape of the tensor, where each element of the operand tensor will be added or multiplied by the scalar.

```python
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```python
(tensor([[[ 2,  3,  4,  5],
          [ 6,  7,  8,  9],
          [10, 11, 12, 13]],

         [[14, 15, 16, 17],
          [18, 19, 20, 21],
          [22, 23, 24, 25]]]),
 torch.Size([2, 3, 4]))
```

## 2.3.6. Reduction


```python
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```

```python
(tensor([0., 1., 2., 3.]), tensor(6.))
```

```python
A.shape, A.sum()
```

```python
(torch.Size([5, 4]), tensor(190.))
```

Addition along an axis.

```python
A = torch.arange(0,24).reshape(2,3,4)
A
```

```python
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
```

```python
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0.shape
```

```python
tensor([[12, 14, 16, 18],
        [20, 22, 24, 26],
        [28, 30, 32, 34]])
torch.Size([3, 4])
```

```python
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1.shape
```

```python
tensor([[12, 15, 18, 21],
        [48, 51, 54, 57]])

torch.Size([2, 4])
```


```python
A_sum_axis2 = A.sum(axis=2)
A_sum_axis2.shape
```

```python
tensor([[ 6, 22, 38],
        [54, 70, 86]])

torch.Size([2, 3])
```

### 2.3.6.1. Non-Reduction Sum

However, sometimes it can be useful to keep the number of axes unchanged when invoking the function for calculating the sum or mean.
```python
sum_A = A.sum(axis=1, keepdims=True)
sum_A

```

```python
tensor([[[12, 15, 18, 21]],

        [[48, 51, 54, 57]]])

torch.Size([2, 1, 4])
```

For instance, since sum_A still keeps its two axes after summing each row, we can divide A by sum_A with broadcasting.

```python
A / sum_A
```

```python
tensor([[[0.0000, 0.0667, 0.1111, 0.1429],
         [0.3333, 0.3333, 0.3333, 0.3333],
         [0.6667, 0.6000, 0.5556, 0.5238]],

        [[0.2500, 0.2549, 0.2593, 0.2632],
         [0.3333, 0.3333, 0.3333, 0.3333],
         [0.4167, 0.4118, 0.4074, 0.4035]]])
```

```python
A.cumsum(axis=0)
```

```python
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 14, 16, 18],
         [20, 22, 24, 26],
         [28, 30, 32, 34]]])
```

```python
C = A.reshape(4,6)
D = C.cumsum(axis=0)
```
```python
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23]])

tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  8, 10, 12, 14, 16],
        [18, 21, 24, 27, 30, 33],
        [36, 40, 44, 48, 52, 56]])
```

## 2.3.7. Dot Products

Mathematics Formula: $ \mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i $

```python
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```python
(tensor([0., 1., 2., 3.]), tensor([1., 1., 1., 1.]), tensor(6.))
```

Note that we can express the dot product of two vectors equivalently by performing an elementwise multiplication and then a sum:

```python
torch.sum(x * y)
```

```python
tensor(6.)
```

## 2.3.8. Matrix-Vector Products

Mathematics Formula:

$ \begin{split}\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.\end{split} $

```python
A.shape, x.shape, torch.mv(A, x)
```

```python
(torch.Size([5, 4]), torch.Size([4]), tensor([ 14.,  38.,  62.,  86., 110.]))
```

## 2.3.9. Matrix-Matrix Multiplication

Mathematics Formula: 

$ \begin{split}\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.\end{split} $

```python
B = torch.ones(4, 3)
torch.mm(A, B)
```

```python
tensor([[ 6.,  6.,  6.],
        [22., 22., 22.],
        [38., 38., 38.],
        [54., 54., 54.],
        [70., 70., 70.]])
```

Matrix-matrix multiplication can be simply called **matrix multiplication**, and should not be confused with the Hadamard product.

## 2.3.10. Norms

```python
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```python
tensor(5.)
```

```python
torch.abs(u).sum()
```

```python
tensor(7.)
```

```python
torch.norm(torch.ones((4, 9)))
```

```python
tensor(6.)
```
