import tiktoken
import torch

class DataloaderLite():
    def __init__(self, config):
        # Preliminary prints
        assert config.total_batch_size % (config.B * config.T) == 0, "Batch size must be divisible by B * T"
        self.grad_accum_steps =  config.total_batch_size // (config.B * config.T) 
        
        print(f"Total batch size: {config.total_batch_size} | Mini Batch size: {config.B} | Sequence length: {config.T}")
        print(f"Using gradient accumulation with {self.grad_accum_steps} steps")
        
        self.B = config.B
        self.T = config.T
        self.start_pos = 0
        
        with open(config.data_path, "r") as f:
            text = f.read()
            
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode_ordinary(text)
        self.tokens = torch.tensor(tokens)
        
        print(f"Number of tokens: {len(self.tokens)} | Need {len(self.tokens) // config.total_batch_size} steps for 1 whole pass")
        
    def next_batch(self):
        buff = self.tokens[self.start_pos:self.start_pos + self.B * self.T + 1]
        x = buff[:-1].view(self.B, self.T)
        y = buff[1:].view(self.B, self.T)
        
        self.start_pos += self.B * self.T
        if self.start_pos + self.B * self.T + 1 >= len(self.tokens):
            self.start_pos = 0
            
        return x, y