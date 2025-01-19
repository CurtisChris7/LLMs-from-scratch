from importlib.metadata import version

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

#----------

import os
import urllib.request

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))
print(raw_text[:99])

#----------

import re

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print(len(preprocessed))

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

#----------

import tiktoken
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

strings = tokenizer.decode(integers)

print(strings)

#----------

text = ("Akwirw ier")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

strings = tokenizer.decode(integers)

print(strings)

#----------

import torch
print("PyTorch version:", torch.__version__)

from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    #raw_text, batch_size=7, max_length=8, stride=8, shuffle=False
)


print(raw_text[:1000])

data_iter = iter(dataloader)
first_batch = next(data_iter)
print("first_batch:", first_batch)

for i, x in enumerate(first_batch):
    for row in  x.tolist(): 
        print(row, "---->", tokenizer.decode(row))

second_batch = next(data_iter)
print("second_batch:", second_batch)

for i, x in enumerate(second_batch):
    for row in  x.tolist(): 
        print(row, "---->", tokenizer.decode(row))

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

print("--------")

vocab_size = 50257
output_dim = 256

torch.manual_seed(123)

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("Token Weight Matrix:\n", token_embedding_layer.weight)

print("Token Embedding for first token:\n", "First Token:", inputs[0][0], "\n", token_embedding_layer(inputs[0][0]), "\nsize:", len(token_embedding_layer(inputs[0][0])))
print("Token Embedding for first batch:\n", "First Batch:", inputs[0], "\n", token_embedding_layer(inputs[0]), "\nsize:", len(token_embedding_layer(inputs[0])))

token_embeddings = token_embedding_layer(inputs)
print("Token Embedding Dimsesion:", token_embeddings.shape)
print("Token Embedding:\n", token_embedding_layer(inputs))

print("--------")

pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)

print("pos_embedding_layer:\n",pos_embedding_layer.weight)

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print("Position Embedding:", pos_embeddings.shape, "\n", pos_embeddings)

input_embeddings = token_embeddings + pos_embeddings
print("New Input Shapes", input_embeddings.shape)

print("Input Embedding:\n", input_embeddings)

