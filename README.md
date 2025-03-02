# Mini-LLaDA

A lightweight implementation of [Large Language Diffusion Models (LLDMs)](https://github.com/ML-GSAI/LLaDA) based on [Llama2](https://github.com/karpathy/llama2.c).

## 🛠 Installation
```bash
# Clone the repository
git clone https://github.com/NiklasDob/Mini-LLaDA.git
cd Mini-LLaDA

# Install dependencies
pip install -r requirements.txt
```
## 📊 Training

Train the model using Shakespeare dataset:
```bash
$ python train.py --dataset shakespeare  # or enwik8
```

## 📝 Text Generation

Generate text based on a given prompt:
```bash
$ python generate.py
  Enter your prompt:
```

## 📂 Structure
```
Mini-LLaDA/
│── data/                   # Datasets (tokenized text)                      
├── model.py                # Llama 2 model for the diffusion objective  
├── generate.py             # Test trained models to generate based on a prompt  
├── train.py                # Train the model on a given dataset (shakespeare or enwik8)  
│── README.md               # Project documentation  
│── requirements.txt        # Dependencies  
```
## License
MIT