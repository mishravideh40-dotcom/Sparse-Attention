import torch

from sparse_attention.dense_attention import causal_mask, dense_attention
from sparse_attention.sliding_window import sliding_window_attention, sliding_window_mask

torch.set_printoptions(precision=3, sci_mode=False)

# 4 tokens, head_dim=2. Q = K here just to keep the demo numbers simple
# (in a real model Q and K come from different learned projections).
Q = torch.tensor([[1.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 1.0],
                   [-1.0, 1.0]]).unsqueeze(0)  # (1, 4, 2)
K = Q.clone()

# distinct, easy-to-recognize "content" for each token
V = torch.tensor([[1.0, 10.0],
                   [2.0, 20.0],
                   [3.0, 30.0],
                   [4.0, 40.0]]).unsqueeze(0)

print("=== DENSE (causal) ===")
mask = causal_mask(4)
print("mask:\n", mask)

raw_scores = (Q @ K.transpose(-2, -1)) / (2 ** 0.5)
print("\nraw scores Q.K^T / sqrt(d):\n", raw_scores)

out, weights = dense_attention(Q, K, V, mask=mask)
print("\nattention weights (after mask + softmax):\n", weights)
print("\noutput (weighted avg of V):\n", out)

print("\n=== SLIDING WINDOW (window_size=2) ===")
sw_mask = sliding_window_mask(4, 2)
print("mask:\n", sw_mask)

sw_out, sw_weights = sliding_window_attention(Q, K, V, window_size=2)
print("\nattention weights:\n", sw_weights)
print("\noutput:\n", sw_out)
