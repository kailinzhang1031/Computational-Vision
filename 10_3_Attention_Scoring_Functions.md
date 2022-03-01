# 10.3. Attention Scoring Functions

```python
import math
import torch
from torch import nn
from d2l import torch as d2l
```

## 10.3.1. Masked Softmax Operation

To get an attention pooling over only meaningful tokens as values, we can specify a valid sequence length (in number of tokens) to filter out those beyond this specified range when computing softmax. 

In this way, we can implement such a **masked softmax operation** in the following **masked_softmax** function, where any value beyond the valid length is masked as zero.

```python
#@save
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

```python
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
```

```python
tensor([[[0.5366, 0.4634, 0.0000, 0.0000],
         [0.5877, 0.4123, 0.0000, 0.0000]],

        [[0.2743, 0.3506, 0.3751, 0.0000],
         [0.3856, 0.2694, 0.3450, 0.0000]]])
```

Similarly, we can also use a two-dimensional tensor to specify valid lengths for every row in each matrix example.

```python
tensor([[[1.0000, 0.0000, 0.0000, 0.0000],
         [0.2555, 0.3955, 0.3490, 0.0000]],

        [[0.5460, 0.4540, 0.0000, 0.0000],
         [0.3326, 0.2032, 0.1953, 0.2689]]])
```

## 10.3.2. Additive Attention

```python
#@save
class AdditiveAttention(nn.Module):
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of `queries`: (`batch_size`, no. of
        # queries, 1, `num_hiddens`) and shape of `keys`: (`batch_size`, 1,
        # no. of key-value pairs, `num_hiddens`). Sum them up with
        # broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of `self.w_v`, so we remove the last
        # one-dimensional entry from the shape. Shape of `scores`:
        # (`batch_size`, no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of `values`: (`batch_size`, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

valid_length: for queries, number of considering (key, value), avoiding padding value in sequence.

queries: batch_size * num_queries * num_keys
keys: batch_size * num_keys * h

- queries: batch_size * queries(1) * h
- keys: batch_size(1) * num_keys * h
- features: batch_size * num_queries * num_keys * h(1)
- scores: batch_size * num_queries * num_keys

```python
torch.bmm(self.dropout(self.attention_weights), values)
```

dropout: turning more terms to 0, droping more (key,value) pair.

- output: batch_size * num_queries * num_keys * value


- queries: batch_size * num_queries * length_queries = 2 * 1 * 20
- keys: batch_size * num_keys * length_keys = 2 * 10 * 2
- value: batch_size * num_value * length_value = 2 * 10 * 4
- values: 1st visible (key,value), 2nd (key,value)

same values, different queries, keys, different scores.

Although additive attention contains learnable parameters, since every key is the same in this example, the attention weights are uniform, determined by the specified valid lengths.


```python
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## 10.3.3. Scaled Dot-Product Attention

Only learnable hyperparameter: **dropout** 

```python
#@save
class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

To demonstrate the above DotProductAttention class, we use the same keys, values, and valid lengths from the earlier toy example for additive attention. For the dot product operation, we make the feature size of queries the same as that of keys.

```python
queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens)
```


```python
tensor([[[ 2.0000,  3.0000,  4.0000,  5.0000]],

        [[10.0000, 11.0000, 12.0000, 13.0000]]])
```

Same as in the additive attention demonstration, since **keys** contains the same element that cannot be differentiated by any query, uniform attention weights are obtained.

```python
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```




