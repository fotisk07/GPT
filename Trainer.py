import math
import time
import torch
import wandb

def get_lr(it, config):
    max_lr = config.max_lr
    min_lr = max_lr * 0.1
    warmup_steps = config.warmup_steps
    max_steps = config.max_steps 
    
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)



def train(model, original_model, optimizer, train_loader, device, config):
    grad_accum_steps = train_loader.grad_accum_steps
    max_steps = config.max_steps
    verbose = config.verbose
    checkpoint_steps = config.checkpoint_steps
    save_path = config.save_path
    wandb_log = config.wandb_log
    
    for step in range(max_steps):
        t0 = time.time()
        loss_accum = 0
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y)
                
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
                            
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        if verbose:
            print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f} ms | tok/sec: {tokens_per_sec:.2f}")
        if wandb_log:
            wandb.log({"step" : step , "loss": loss_accum.item(), "lr": lr, "norm": norm, "dt": dt*1000, "tok/sec": tokens_per_sec})
            
        if checkpoint_steps > 0 and step % checkpoint_steps == 0:
            name = save_path + f"_step{step}.pt"
            torch.save(original_model.state_dict(), name)
        