import numpy as np
import math

L = 8
d_model = 512 
num_heads = 2  
head_dim = d_model // num_heads  


query_matrix = np.random.randn(L, d_model)
key_matrix = np.random.randn(L, d_model)
value_matrix = np.random.randn(L, d_model)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1)

# scaled dot-product attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    attention_scores = np.matmul(Q, K.T) / math.sqrt(Q.shape[-1])
    if mask is not None:
        attention_scores += mask
    attention_weights = softmax(attention_scores)
    return np.matmul(attention_weights, V)

# masking
mask = np.tril(np.ones((L, L)))
mask[mask == 0] = -np.inf
mask[mask == 1] = 0

# spliting the single matrix for multiple heads
def split_heads(matrix, num_heads, head_dim):
    return matrix.reshape(L, num_heads, head_dim).transpose(1, 0, 2)  # (num_heads, L, head_dim)

# rearranging the heads as original one
def merge_heads(matrix):
    return matrix.transpose(1, 0, 2).reshape(L, -1)  # (L, d_model)

# Split Query, Key, and Value into heads
Q_heads = split_heads(query_matrix, num_heads, head_dim)
K_heads = split_heads(key_matrix, num_heads, head_dim)
V_heads = split_heads(value_matrix, num_heads, head_dim)

# Perform attention for each head
head_outputs = []
for i in range(num_heads):
    head_output = scaled_dot_product_attention(Q_heads[i], K_heads[i], V_heads[i], mask)
    head_outputs.append(head_output)

# Add al multihead answers 
multi_head_output = merge_heads(np.array(head_outputs))

print("Multi-Head Attention Output:\n", multi_head_output)
