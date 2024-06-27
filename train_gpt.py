import torch
import torch.nn as nn
from dataclasses import dataclass
import tiktoken 
import torch.nn.functional as F
import math
import time


class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super(MultiHeadAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.NANOGPT_SCALE_INIT = 1 #Following karpathy GPT2 training script for initialization of residual weights

        self.nb_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
                                     .view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        # x: B, T, C
        B, T, C = x.size()
        qkv = self.c_attn(x)  # B, T, 3 * n_embd
        q, k, v = torch.chunk(qkv, 3, dim=-1) # B, T, n_embd

        q = q.view(B, T, self.nb_head, self.n_embd // self.nb_head).permute(0, 2, 1, 3) # B, nb_head, T, headsize
        k = k.view(B, T, self.nb_head, self.n_embd // self.nb_head).permute(0, 2, 1, 3) # B, nb_head, T, headsize
        v = v.view(B, T, self.nb_head, self.n_embd // self.nb_head).permute(0, 2, 1, 3) # B, nb_head, T, headsize

        # attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # attention =  attention.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # attention = torch.nn.functional.softmax(attention, dim=-1) # B, nb_head, T, T
        # out = torch.matmul(attention, v) # B, nb_head, T, headsize
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.n_embd) # B, T, n_embd
        out = self.c_proj(out) # B, T, n_embd
        
        return out
    
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.act = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.NANOGPT_SCALE_INIT = 1 #Following karpathy GPT2 training script for initialization of residual weights


    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)

        return x    

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # parameter sharing with wte and lm_head
        self.transformer.wte.weight = self.lm_head.weight
        
        # Parameter initialization
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        # x is (B, T)
        B, T = x.size()
        assert T <= self.transformer['wpe'].weight.size(0), "Cannot forward, model has been trained with T=%d, but current sequence has length T=%d" % (self.transformer['wpe'].weight.size(0), T)
        token_encoding = self.transformer['wte'](x) # (B, T, C)
        position_encoding = self.transformer['wpe'](torch.arange(x.size(1),dtype=torch.long, device=x.device)) # (T, C)

        x = token_encoding + position_encoding # (B, T, C) broadcast along B

        for block in self.transformer['h']:
            x = block(x)

        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x) # (B, T, V)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.masked_bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape, f"mismatched shape: {sd_hf[k].shape} != {sd[k].shape} for key {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    
    
class Dataloader():
    def __init__(self, B, T):
        self.B = B
        self.T = T
        self.start_pos = 0
        
        with open("shakespeare.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode_ordinary(text)
        self.tokens = torch.tensor(tokens)
        
        print("Number of tokens:", len(self.tokens))
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        
    def next_batch(self):
        buff = self.tokens[self.start_pos:self.start_pos + self.B * self.T + 1]
        x = buff[:-1].view(self.B, self.T)
        y = buff[1:].view(self.B, self.T)
        
        self.start_pos += self.B * self.T
        if self.start_pos + self.B * self.T + 1 >= len(self.tokens):
            self.start_pos = 0
            
        return x, y
        
        
        
        
        

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # For reproducibility, in sync with Andej Karpathy's GPT2 training script
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)

    model = GPT(GPTConfig())
    model.to(device)
        
    torch.set_float32_matmul_precision('high')
    #---------------model training-------------------------------------------------------
    
    # load the dataset
    B, T = 3, 5
    train_loader = Dataloader(B, T)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    for i in range(20):
        t0 = time.time()
        optimizer.zero_grad()
        x,y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=str(device), dtype = torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if device == torch.device("cuda"):
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        print(f"Step {i} | Loss: {loss.item():.3f} | Norm: {norm:.3f} | dt: {1000 * dt:.2f} ms | tokens/s: {train_loader.B * train_loader.T/dt : .2f}")
        
      


    import sys ; sys.exit(0)
    
    
    #---------------model sampling-------------------------------------------------------

    # generate! right now x is (B, T) where B = 5, T = 8
    # set the seed to 42
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
    # forward the model to get the logits
        with torch.no_grad():
            logits = model(x) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


        

        




        