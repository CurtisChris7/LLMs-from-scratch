from importlib.metadata import version

print("torch version:", version("torch"))

import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]  # 2nd input token is the query



attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2)
print("--------------------------------")
res = 0.
for idx, val in enumerate(inputs[0]):
    res += inputs[0][idx] * inputs[1][idx]

print(res)
print(torch.dot(inputs[0], query))

print("--------------------------------")
print("\tInputs")
print(inputs)
print("Transpose:\n",inputs.T)

print("--------------------------------")
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

print("--------------------------------")
query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)

print("--------------------------------")
print("\tAttention Scores")
attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print("Iterative Computation\n", attn_scores)
attn_scores = inputs @ inputs.T
print("Matrix Multiplication\n",attn_scores)

print("--------------------------------")
print("\tAttention Weights")
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

print("--------------------------------")
print("\tContext Vectors")
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)


print("--------------------------------")
torch.manual_seed(123)
x_2 = inputs[1] # second input element
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print("\tQuery PW Matrix\n",W_query)
print("\tKey PW Matrix\n",W_key)
print("\tValue PW Matrix\n",W_value)

print("--------------------------------")
keys = inputs @ W_key 
values = inputs @ W_value
queries = inputs @ W_query

print("keys.shape:", keys.shape)
print("\tKey Matrix\n", keys)
print("values.shape:", values.shape)
print("\tValues Matrix\n", values)
print("queries.shape:", queries.shape)
print("\tqueries Matrix\n", queries)

print("--------------------------------")
print("Vector Level Passthrough - 2nd Input Element")
query_2 = x_2 @ W_query # _2 because it's with respect to the 2nd input element
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print("attn_scores_2:", attn_scores_2)

d_k = keys.shape[1]
print("Scaling Factor:", d_k)
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("attn_weights_2:", attn_weights_2)
context_vec_2 = attn_weights_2 @ values
print("context_vec_2:", context_vec_2)

print("--------------------------------")

attn_scores = queries @ keys.T
print("attn_scores:", attn_scores)
attn_weights  = torch.softmax(attn_scores / d_k**0.5, dim=-1)
print("attn_weights:", attn_weights)
context_vec = attn_weights @ values
print("context_vec:", context_vec)

print("--------------------------------")
import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print("\tsa_v1(inputs):\n",sa_v1(inputs))

print("--------------------------------")

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print("\tsa_v2(inputs):\n",sa_v2(inputs))

print("--------------------------------")
print("\t\tEX 3.1 SOLUTION:")
sa_v1.W_query = nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key = nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = nn.Parameter(sa_v2.W_value.weight.T)
print(sa_v1(inputs))

print("--------------------------------")
print("Masking")
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs) 
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print("attn_weights:\n",attn_weights)

context_length = attn_scores.shape[0]
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("attn_weights:\n",masked)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print("attn_weights:\n",attn_weights)

print("--------------------------------")
print("Dropout")
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) # dropout rate of 50%
example = torch.ones(6, 6) # create a matrix of ones

print(example)
print(dropout(example))
torch.manual_seed(123)
print(dropout(attn_weights))
batch = torch.stack((inputs, inputs), dim=0)
print("batch:\n",batch)
print("batch.shape:\n",batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3

print("--------------------------------")
print("CausalAttention")
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        #print("keys", keys, keys.shape)
        #print("queries", queries)
        #print("values", values)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        
        #print("attn_scores", attn_scores, attn_scores.shape)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        #print("attn_weights", attn_weights, attn_weights.shape)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)

context_vecs = ca(batch)

print("d_in", d_in, "d_out", d_out, "context_length", context_length)
print("\tcontext_vecs:\n",context_vecs)
print("context_vecs.shape:", context_vecs.shape)

print("--------------------------------")
print("MultiHeadAttentionWrapper")
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print("context_vecs:\n",context_vecs)
print("context_vecs.shape:", context_vecs.shape)

print("--------------------------------")
print("\t\tEX 3.2 SOLUTION:")
d_in, d_out = 3, 1
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print("context_vecs:\n",context_vecs)
print("context_vecs.shape:", context_vecs.shape)

print("--------------------------------")
print("\tTransposition:")
# (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

print(a,a.shape)
a=a.transpose(2, 3)
print(a,a.shape)

print("--------------------------------")
print("MultiHeadAttention")
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        print("x.shape:", x.shape)

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)


        print("d_out:", d_out)
        print("W_query", self.W_query.weight.shape)
        print("W_key:", self.W_key.weight.shape)
        print("W_value", self.W_value.weight.shape)

        #print("keys:", keys.shape)
        #print("queries:", queries)
        #print("values:", values)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        print("num_heads:", self.num_heads, "head_dim:", self.head_dim)
        #print("keys:", keys, keys.shape)
        print("queries:", queries.shape)
        print("values:", values.shape)
        print("keys:", keys.shape)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        print("T queries:", queries.shape)
        print("T values:", values.shape)
        print("T keys:", keys.shape)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        print("attn_weights:", attn_weights.shape)

        # Shape: (b, num_tokens, num_heads, head_dim)
        print("(attn_weights @ values)", (attn_weights @ values).shape)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        #print("context_vec\n", context_vec, context_vec.shape)
        print("context_vec:", context_vec.shape)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        #context_vec = self.out_proj(context_vec) # optional projection
        #print("context_vec:", context_vec, context_vec.shape)
        print("context_vec:", context_vec.shape)
        #print("out_proj:\n", self.out_proj)

        return context_vec

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print("context_vecs:",context_vecs)
print("context_vecs.shape:", context_vecs.shape)

print("--------------------------------")
print("\t\tEX 3.3 SOLUTION:")

d_in = d_out = 768
inputs = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
#batch = torch.stack(tuple([inputs for _ in range(25)]), dim=0)
batch = torch.stack((inputs, inputs), dim=0)
print("batch:\n",batch)
print("batch.shape:\n",batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3

mha = MultiHeadAttention(d_in, d_out, context_length=1024, dropout=0.0, num_heads=12)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

print("--------------------------------")