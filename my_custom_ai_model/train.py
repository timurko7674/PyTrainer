import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from tokenizer.char_tokenizer import CharTokenizer
from model.transformer import Transformer
from utils import get_batch

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and tokenize data
with open("data/your_dataset.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = CharTokenizer(raw_text)
encoded_data = tokenizer.encode(raw_text)

# Check if dataset is large enough for sequence length
if len(encoded_data) < seq_len + 2:
    raise ValueError(f"Dataset too small for sequence length {seq_len}. "
                     f"Data length: {len(encoded_data)}")

VOCAB_SIZE = len(tokenizer.vocab)

# Create model
model = Transformer(
    vocab_size=VOCAB_SIZE,
    emb_size=embed_dim,
    n_heads=num_heads,
    n_layers=num_layers,
    seq_len=seq_len
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Number of batches per epoch
batches_per_epoch = 100

for epoch in range(epochs):
    total_loss = 0
    batch_count = 0

    batch_gen = get_batch(encoded_data, batch_size, seq_len)
    for _ in range(batches_per_epoch):
        x, y = next(batch_gen)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")
