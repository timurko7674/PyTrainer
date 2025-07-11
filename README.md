PyTrainer

Welcome! This project contains Python scripts and modules to help you train and generate with an AI model.

Project Structure:

train.py (Script to train the AI model)

generate.py (Script to generate output using a trained model)

config.py (Configuration file for customizing training and generation)

utils.py (Utility functions used across the project)

tokenizer/char_tokenizer.py (Tokenizer implementation)

model/transformer.py (Model architecture - Transformer)

data/your_dataset.txt (Dataset for training)

Usage:

Train the Model
Run the training script to start training your AI model:
python train.py
This will use your dataset in data/your_dataset.txt and save a trained model.

Generate Output
After training, use the generated model with:
python generate.py
This script requires the trained model to produce output.

Configuration:

The config.py file holds parameters for training and generation. You can edit it to change things like learning rate, batch size, epochs, or generation settings. Only modify this if you understand the impact.

NOTE: The default configuration is set for less powerful PCs to ensure compatibility.

If you have a powerful PC, the following settings are recommended in config.py:

vocab_size = 8192
embed_dim = 512
num_heads = 8
num_layers = 6
seq_len = 256
batch_size = 64
learning_rate = 0.0005
epochs = 25

Dependencies:
Make sure you have the required Python packages installed.

License:
This project is licensed under the MIT License. See the LICENSE file for details.

Bugs or edits?
Please write them in an issue
