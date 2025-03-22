import tiktoken
import torch
from importlib.metadata import version
from previous_chapters import GPTModel,generate_text_simple

pkgs = ["matplotlib", 
        "numpy", 
        "tiktoken", 
        "torch",
        "tensorflow" # For OpenAI's pretrained weights
       ]
for p in pkgs:
    print(f"{p} version: {version(p)}")

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Disable dropout during inference

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


inputs = torch.tensor([
    [16833,3626,6100], # every effort moves
    [40,1107,588] # I really like
])
targets = torch.tensor([
    [3626,6100,345], # effort moves you
    [1107,588,11311] # really like chocolate
])

with torch.no_grad():
    logits = model(inputs)
probs = torch.softmax(logits,dim=-1)
print(probs.shape)
token_ids = torch.argmax(probs,dim=-1,keepdim=True)
print("Token IDs:\n",token_ids)

print(f"Targets batch 1:{token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:{token_ids_to_text(token_ids[0].flatten(), tokenizer)}")


text_idx = 0
target_probs_1 = probs[text_idx, [0,1,2], targets[text_idx]]
print("Text 1:", target_probs_1)

text_idx = 1
target_probs_2 = probs[text_idx, [0,1,2], targets[text_idx]]
print("Text 2:", target_probs_2)

log_probs = torch.log(torch.cat((target_probs_1,target_probs_2)))
print(log_probs)

print("="  * 50)

