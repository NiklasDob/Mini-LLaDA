import torch
import os
import tiktoken
from model import Transformer, ModelArgs
from tqdm import tqdm
from train import generate

def load_checkpoint(model, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return model

def generate_text(model, tokenizer, prompt, steps=50, gen_length=128, temperature=0.7, device='cpu'):
    tokenized_prompt = tokenizer.encode_ordinary(prompt)
    tokenized_prompt = torch.tensor(tokenized_prompt, dtype=torch.long).to(device)
    output = generate(model, tokenized_prompt.unsqueeze(0), steps=steps, gen_length=gen_length, temperature=temperature)
    res = tokenizer.decode_batch(output.cpu().tolist())
    return res[0]

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = ModelArgs(
        dim=256,
        n_layers=8,
        n_heads=8,
        vocab_size=50257,
        multiple_of=256,
        max_seq_len=2048
    )  
    
    model = Transformer(args).to(device)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    checkpoint_path = "checkpoints/enwik8_dim=256_heads=8_layers=8_dropout=0.1/model_27000.pth"
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found! Please train the model first.")
        return
    
    model = load_checkpoint(model, checkpoint_path, device=device)
    model.eval()
    
    while True:
        prompt = input("Enter your prompt: ")
        if prompt.lower() == "exit":
            break
        
        generated_text = generate_text(model, tokenizer, prompt, device=device)
        print("\nGenerated text:\n", generated_text)

if __name__ == "__main__":
    main()
