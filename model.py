import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import inspect, os


class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super(MultiHeadAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.NANOGPT_SCALE_INIT = 1 #Following karpathy GPT2 training script for initialization of residual weights

        self.nb_head = config.n_head
        self.n_embd = config.n_embd
        
        
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
    


class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            ln_f = nn.LayerNorm(self.config.n_embd),
        ))
        

        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        
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
    
    def configure_optimizers(self, config, device):
        weight_decay = config.weight_decay
        learning_rate = config.learning_rate
        betas = config.betas
        eps = config.eps
        
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas= betas, eps=eps, fused=use_fused)
        
        return optimizer


    @classmethod
    def from_local(cls, config):
        model = GPT(config)
        
        # if load_path is a directory, load the latest checkpoint
        if os.path.isdir(config.load_path):
            config.load_path = os.path.join(config.load_path, max(os.listdir(config.load_path)))
        
        print("loading model from %s" % config.load_path)
        
        model.load_state_dict(torch.load(config.load_path))
        
        return model
        

    @classmethod
    def from_pretrained(cls, model_type, config):
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
    