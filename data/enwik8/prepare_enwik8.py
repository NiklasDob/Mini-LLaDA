import os
import requests
import numpy as np
import tiktoken
import bz2

# Define file paths
dataset_url = "http://mattmahoney.net/dc/enwik8.zip"
compressed_file_path = os.path.join(os.path.dirname(__file__), "enwik8.zip")
uncompressed_file_path = os.path.join(os.path.dirname(__file__), "enwik8")

# Download the dataset if not exists
if not os.path.exists(compressed_file_path):
    print("Downloading enwik8 dataset...")
    response = requests.get(dataset_url, stream=True)
    with open(compressed_file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    print("Download complete.")

# Extract the dataset if not exists
if not os.path.exists(uncompressed_file_path):
    print("Extracting enwik8 dataset...")
    os.system(f"unzip {compressed_file_path} -d {os.path.dirname(__file__)}")
    print("Extraction complete.")

# Read the dataset
with open(uncompressed_file_path, "r", encoding="utf-8") as f:
    data = f.read()
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# Tokenize using tiktoken GPT-2 BPE
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Save as binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

print("Preprocessing complete. Binary files saved.")