import torch
import torch.nn.functional as F
from config import *
from tokenizer.char_tokenizer import CharTokenizer
from model.transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load raw text and create tokenizer
with open("data/your_dataset.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = CharTokenizer(raw_text)
VOCAB_SIZE = len(tokenizer.vocab)

# Create model and load weights
model = Transformer(VOCAB_SIZE, embed_dim, num_heads, num_layers, seq_len).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

prompt = input("Enter prompt: ")
input_ids = tokenizer.encode(prompt)

# Truncate input to seq_len to avoid size mismatch
if len(input_ids) > seq_len:
    input_ids = input_ids[:seq_len]

input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

temperature = 0.8  # tweak this between 0.6 (less random) and 1.2 (more random)

with torch.no_grad():
    for _ in range(100):
        out = model(input_tensor)
        logits = out[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_id]], device=device)], dim=1)

        # Keep the input sequence length fixed to seq_len
        if input_tensor.size(1) > seq_len:
            input_tensor = input_tensor[:, -seq_len:]

generated_text = tokenizer.decode(input_tensor[0].tolist())
print("Generated:", generated_text)
