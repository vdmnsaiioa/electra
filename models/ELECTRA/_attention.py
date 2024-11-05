import torch
import torch.nn as nn


class BatchMultiHeadAttention(nn.Module):
    def __init__(self, num_heads=9, input_dim=9):
        super(BatchMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, rows, cols = x.size()

        # Flatten each 3x3 matrix to a 9-dimensional vector
        x = x.view(batch_size, -1)  # Shape: (batch_size, 9)

        # Apply linear transformations and split into heads
        queries = self.query(x).view(batch_size, self.num_heads, self.dim_per_head)
        keys = self.key(x).view(batch_size, self.num_heads, self.dim_per_head)
        values = self.value(x).view(batch_size, self.num_heads, self.dim_per_head)

        # Transpose to get dimensions: (num_heads, batch_size, dim_per_head)
        queries = queries.transpose(0, 1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (
                    self.dim_per_head ** 0.5)  # Shape: (num_heads, batch_size, batch_size)
        attn_weights = self.softmax(scores)  # Shape: (num_heads, batch_size, batch_size)
        attn_output = torch.matmul(attn_weights, values)  # Shape: (num_heads, batch_size, dim_per_head)

        # Transpose back and concatenate heads
        attn_output = attn_output.transpose(0, 1).contiguous().view(batch_size, -1)  # Shape: (batch_size, 9)

        # Reshape back to original shape
        attn_output = attn_output.view(batch_size, rows, cols)  # Shape: (batch_size, 3, 3)
        return attn_output