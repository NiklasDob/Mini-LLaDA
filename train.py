from contextlib import nullcontext
import tiktoken
from tqdm import tqdm
from model import Transformer, ModelArgs
import torch
import os
import numpy as np
import torch.nn.functional as F


def noise_input(x, eps=1e-3):
    b, l = x.shape
    t = torch.randint(0, 1000, (b,), device=x.device)
    p_mask = (1 - eps) * (t / 999) + eps
    p_mask = p_mask[:, None].repeat(1, l)
    masked_indices = torch.rand((b, l), device=x.device) < p_mask
    random_tokens = torch.randint(0, 50257, (b, l), device=x.device)
    noisy_batch = torch.where(masked_indices, random_tokens, x)
    return noisy_batch, t#, masked_indices, p_mask

def get_batch(split, batch_size, block_size, device):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    y = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    x = torch.clone(y)
    x, t = noise_input(x)
   

    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        t = t.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        t = t.to(device)

    return x, y, t


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=50, gen_length=128, block_length=128, temperature=0.1,
             cfg_scale=0., remasking='low_confidence', eps =1e-3):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, l).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
    '''
    x = torch.randint(0, 50257, (1, prompt.shape[1] + gen_length), dtype=torch.long, device=prompt.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = torch.zeros_like(x.float(), dtype=bool)
    prompt_index[:, :prompt.shape[1]] = True

    for t in np.linspace(999, 0, steps).astype(int):
        ts = torch.ones(x.shape[0], device=x.device, dtype=torch.int64) * t
        logits = model(x,ts)

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        # Only the loggits that do not belong to the prompt sample then new tokens

        b, l = x.shape

        p = F.softmax(logits_with_noise[:, prompt.shape[1]:].to(torch.float64), dim=-1)
        # x0 = torch.multinomial(p, num_samples=1)
        b,l, e = p.shape
        x0 = torch.multinomial(p.view(-1, e), num_samples=1).view(b, l)
      
        b,l = x0.shape
        # Add some new noise to the generated tokens, depending on the timestep
        p_mask = (1 - eps) * (ts / 999) + eps
        p_mask = p_mask[:, None].repeat(1, l)
        masked_indices = torch.rand((b, l), device=x.device) < p_mask
        random_tokens = torch.randint(0, 50257, (b, l), device=x.device)
        x0 = torch.where(masked_indices, random_tokens, x0)

        x[:, prompt.shape[1]:] = x0

    return x

if __name__ == '__main__':
    cwd = os.path.dirname(__file__)
    data_dir = os.path.join(cwd, 'data')
    BLOCK_SIZE = 256
    BATCH_SIZE = 16
    NUM_STEPS = 50_000
    VAL_STEPS = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    ptdtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16
    args = ModelArgs(
        dim=128,
        n_layers=8,
        n_heads=8,
        vocab_size=50257,
        multiple_of=256,
        max_seq_len=2048
    )  
    model = Transformer(args).to(device)
    # Print number of model parameters
    print("Num parameters: ", sum(p.numel() for p in model.parameters()))
    model.compile()

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    X, Y, T = get_batch('train', BATCH_SIZE, BLOCK_SIZE, device)
    tokenizer = tiktoken.get_encoding("gpt2")
    for step in tqdm(range(NUM_STEPS), desc='Training steps', total=NUM_STEPS):
        model.train()
        with ctx:
            logits = model(X, T, targets=Y)
            loss = model.last_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        X, Y, T = get_batch('train', BATCH_SIZE, BLOCK_SIZE, device)

        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.3f}")
        
        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                all_losses = 0
                for val_step in tqdm(range(VAL_STEPS),desc='Validation steps',total=VAL_STEPS):
                    X_val, Y_val, T_val = get_batch('val', BATCH_SIZE, BLOCK_SIZE, device)
                    logits = model(X_val, T_val, targets=Y_val)
                    loss = model.last_loss
                    all_losses += loss.item()
                all_losses /= VAL_STEPS
            # Save the model at checkpoints directory    
            checkpoint_dir = os.path.join(cwd, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "model_args": args}, os.path.join(checkpoint_dir, f"model_{step}.pth"))
            prompt = "All:\nWhat would "
            
            tokenized_prompt = tokenizer.encode_ordinary(prompt)
            tokenized_prompt = torch.tensor(tokenized_prompt, dtype=torch.long).to(device)
            output = generate(model,tokenized_prompt.unsqueeze(0))
            res = tokenizer.decode_batch(output.cpu().tolist())
            print("Generated text:" )
            print(res[0])
            print(f"Mean validation loss: {all_losses:.3f}")

    