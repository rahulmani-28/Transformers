import numpy as np
import math
# Define constants
L = 8  # Sequence length
d_model = 512  # Model dimension
num_heads = 2  # Number of attention heads
head_dim = d_model // num_heads  # Dimension of each head

# input data
query_matrix = np.random.randn(L, d_model)
key_matrix = np.random.randn(L, d_model)
value_matrix = np.random.randn(L, d_model)

# Softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1)

# # Masking for attention
mask = np.tril(np.ones((L, L)))
mask[mask == 0] = -np.inf
mask[mask == 1] = 0

# Scaled dot-product attention function
def scaled_dot_product_attention(Q, K, V, mask=None):
    attention_scores = np.matmul(Q, K.T) / math.sqrt(Q.shape[-1])
    if mask is not None:
        attention_scores += mask
    attention_weights = softmax(attention_scores)
    return np.matmul(attention_weights, V)


# Function to split the input matrix into multiple heads
def split_heads(matrix, num_heads, head_dim):
    return matrix.reshape(L, num_heads, head_dim).transpose(1, 0, 2)


# Function to merge the attention heads back into a single matrix
def merge_heads(matrix):
    return matrix.transpose(1, 0, 2).reshape(L, -1)  # (L, d_model)


# Function to add positional encoding to the input matrix
def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len).reshape(-1, 1)
    div_term = np.power(10000, -np.arange(0, d_model, 2) / d_model)
    pos_enc = np.zeros((seq_len, d_model))
    pos_enc[:, 0::2] = np.sin(positions / div_term)
    pos_enc[:, 1::2] = np.cos(positions / div_term)
    return pos_enc


# Function to normalize input with layer normalization
def layer_normalization(x, epsilon=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)


# Residual connection function
def residual_connection(x, sublayer, dropout_prob=0.1):
    return x + np.random.binomial(1, dropout_prob, size=x.shape) * sublayer(layer_normalization(x))


# Feedforward network function
def feedforward(x, d_model, d_ff, dropout_prob=0.1):
    # Linear 1
    x = np.dot(x, np.random.randn(d_model, d_ff))
    x = np.maximum(x, 0)  # ReLU activation
    x = np.random.binomial(1, dropout_prob, size=x.shape) * x  # Dropout
    # Linear 2
    x = np.dot(x, np.random.randn(d_ff, d_model))
    return x

 
# Encoder block function with self-attention and feedforward
def encoder_block(x, self_attention_block, feed_forward_block, mask, dropout_prob):
    # Self-attention
    x = residual_connection(x, lambda x: self_attention_block(x, x, x, mask), dropout_prob)

    # Feedforward
    x = residual_connection(x, feed_forward_block, dropout_prob)
    return x


# Transformer Encoder function
def transformer_encoder(input_ids, d_model, vocab_size, num_heads, d_ff, num_layers, dropout_prob=0.1):
    # Create an embedding for the input (simple random initialization here)
    embedding = np.random.randn(vocab_size, d_model)  # Embedding matrix 10000*512
    x = embedding[input_ids] #8*512
    # Add positional encoding
    pos_enc = positional_encoding(x.shape[0], d_model)  # Shape of input: (L, d_model) 8*512
    x = x + pos_enc  # Add positional encoding to input
    # print(pos_enc)
    # print(x)
    # Create the mask
    mask = np.tril(np.ones((x.shape[0], x.shape[0])))  # Mask shape: (L, L)

    # Multi-layer Transformer encoder
    self_attention_block = lambda q, k, v, m: scaled_dot_product_attention(q, k, v, m)
    feed_forward_block = lambda x: feedforward(x, d_model, d_ff, dropout_prob)

    for _ in range(num_layers):
        x = encoder_block(x, self_attention_block, feed_forward_block, mask, dropout_prob)
    return x


vocab_size = 10000  # Vocabulary size
input_ids = np.random.randint(0, vocab_size, L)  # Input tokens (8 tokens)
d_ff = 2048  # Feedforward dimension
num_layers = 6  # Number of transformer layers


output = transformer_encoder(input_ids, d_model, vocab_size, num_heads, d_ff, num_layers)

print("Transformer Encoder Output:")
print(output)