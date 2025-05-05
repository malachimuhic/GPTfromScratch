import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 16 #how many independant sequences will we process in parallel
block_size = 128 # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
device = 'cpu'
eval_iters = 200
n_embd = 256 # embedding dimension
n_head = 4
n_layer = 4
dropout = 0.2
# =============================

torch.manual_seed(1337)

# input.txt is a text file containing the training data
with open('input.txt', 'r') as f:
    text = f.read()

# here are all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create mapping from character to integers
stoi = {ch: i for i,ch in enumerate(chars)}
itos = {i: ch for i,ch in enumerate(chars)}
# encode the text into integers
encode = lambda s: [stoi[c] for c in s] # turn string into int
decode = lambda l: ''.join([itos[i] for i in l]) # turn int into string

# train and test splits
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data)) # first 98%? will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # random starting indices
    x = torch.stack([data[i:i+block_size] for i in ix]) # (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # (batch_size, block_size)
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(): # average loss over multiple batches
    out = {}
    model.eval() # set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) 
        for k in range(eval_iters): # loop over eval_iters batches
            # get a batch of data
            X, Y = get_batch(split) # (batch_size, block_size)
            # forward pass
            logits, loss = model(X, Y) # (batch_size, block_size, vocab_size), (batch_size)
            # reshape the logits and targets to compute the loss
            losses[k] = loss.item() # (batch_size) # move to device
        out[split] = losses.mean() # mean loss
    model.train() # set model back to training mode
    # return the average loss for train and val sets
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # linear transformation of the previous layer
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.layernorm1(x))
        x = x + self.ffwd(self.layernorm2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # (vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # (block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final Layer Norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # (n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape # (batch_size, block_size)
        # idx is (B, T) array of indices in the current context
        # get the token and position embeddings

        token_embeddings = self.token_embedding_table(idx) # (batch_size, block_size, vocab_size)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (block_size, n_embd)
        x = token_embeddings + position_embeddings # (batch_size, block_size, n_embd)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (batch_size, block_size, vocab_size)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # (B*T, C)
            targets = targets.view(B*T) # (B*T)
            loss = F.cross_entropy(logits, targets) # cross entropy loss
            return logits, loss
        return logits
    
    def generate(self, idx, max_new_tokens):
      # idx is (B, T) array of indices in the current context
      for _ in range(max_new_tokens):
        # crop idx to the lasst block_size tokens
        idx_cond = idx[:, -block_size:]  # correct slicing
        # get the predictions
        logits = self(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
      return idx

model = BigramLanguageModel() # create the model
# move the model to the device (GPU or CPU)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = model(xb, yb)
    # zero the gradients
    optimizer.zero_grad(set_to_none=True)
    # backpropagation
    loss.backward()
    # update the weights
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # (1, 1)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))
# The above code will generate 500 characters of text based on the trained model.