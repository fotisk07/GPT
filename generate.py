from model import GPT
import torch
import torch.nn.functional as F

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
        
# Load config from yaml file
import yaml
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
config = AttrDict(config) # convert to AttrDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = GPT.from_local(config)
model.to(device)

import tiktoken
enc = tiktoken.get_encoding("gpt2")
input = "The quick brown fox jumps over the lazy dog"
tokens = enc.encode_ordinary(input)
# Create a tensor from the tokens that is 5x8
x = torch.tensor(tokens).unsqueeze(0).repeat(5, 1)
x = x.to(device)

max_length = 50
num_return_sequences = 5
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
# forward the model to get the logits
    with torch.no_grad():
        
        logits, _ = model(x) # (B, T, vocab_size)
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


    


