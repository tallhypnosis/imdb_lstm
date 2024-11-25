#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install datasets


# In[2]:


pip --upgrade numpy


# In[3]:


from datasets import load_dataset

# Load IMDB dataset using Hugging Face datasets
imdb_dataset = load_dataset("imdb")

# Access the train and test datasets
train_dataset = imdb_dataset['train']
test_dataset = imdb_dataset['test']

# Print a few examples
for example in train_dataset.select([0, 1, 2]):
    print(f"Label: {example['label']}, Review: {example['text']}")


# In[4]:


import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn


nltk.download('punkt')
nltk.download('stopwords')

nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

# Define a text preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    return tokens

# Apply preprocessing to the training and test datasets
train_dataset = train_dataset.map(lambda x: {'tokens': preprocess_text(x['text'])})
test_dataset = test_dataset.map(lambda x: {'tokens': preprocess_text(x['text'])})

# Check out the first preprocessed example
print(train_dataset[0])


# In[5]:


max_len = 200

def padding(tokens):
  if len(tokens) < max_len:
    return tokens + ['<pad>'] * (max_len - len(tokens))
  else:
    return tokens[:max_len]

train_dataset = train_dataset.map(lambda x: {'padded_tokens': padding(x['tokens'])})
test_dataset = test_dataset.map(lambda x: {'padded_tokens': padding(x['tokens'])})

print(train_dataset[0]['padded_tokens'])


# In[6]:


from collections import Counter

def build_vocab(data):
  vocab = Counter()
  for chunk in data:
    vocab.update(chunk['padded_tokens'])
  return vocab

vocab = build_vocab(train_dataset)


# In[7]:


print(vocab)


# In[8]:


len(vocab)


# In[9]:


# word_to_idx = {word: i+2 for i, (word, _) in enumerate(vocab.items())}  # +2 to account for padding and unknown tokens
word_to_idx = {word: i+2 for i, word in enumerate(vocab.keys())}
word_to_idx['<pad>'] = 0
word_to_idx['<unk>'] = 1


# In[10]:


# Reverse mapping from indices to words
idx_to_word = {idx: word for word, idx in word_to_idx.items()}


# In[11]:


def numericalize(tokens):
  return [word_to_idx.get(word, word_to_idx['<unk>']) for word in tokens]


# In[12]:


for item in train_dataset['padded_tokens'][:5]:  # Sample check
    print(numericalize(item))  # Should only contain indices < vocab_size


# In[13]:


train_dataset = train_dataset.map(lambda x: {'input_ids': numericalize(x['padded_tokens'])})
test_dataset = test_dataset.map(lambda x: {'input_ids': numericalize(x['padded_tokens'])})

print(train_dataset[0]['input_ids'])


# In[14]:


# Function to decode token IDs into text
def decode_sequence_to_text(sequence):
    """Convert a sequence of numerical indices back into a readable text."""
    words = [idx_to_word.get(idx, '<unk>') for idx in sequence]
    return ' '.join(word for word in words if word not in ['<pad>', '<unk>'])


# In[15]:


import random
from torch.utils.data import DataLoader, Dataset

# Synonym replacement augmentation function
def augment_text_with_synonyms(text, augmentation_probability=0.1):
    # If input is a list of tokens, join them into a string
    if isinstance(text, list):
        text = ' '.join(text)

    words = text.split()
    augmented_words = []
    for word in words:
        if random.random() < augmentation_probability:  # Apply augmentation with probability
            synonym = get_synonym(word)  # Replace with a synonym
            if synonym:
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)
    return ' '.join(augmented_words)  # Return augmented sentence as a string



def get_synonym(word):
    synonyms = wn.synsets(word)
    if synonyms:
        return synonyms[0].lemmas()[0].name()
    return None


# In[94]:


# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        # print(f"this is type {type(self.labels[0])}")

    def __len__(self):
      # print(f"this is type {len(self.texts)} ")
      return len(self.texts)


    def __getitem__(self, idx):
        # print(f"this is type {type(self.texts[idx])}")
        return self.texts[idx], self.labels[idx]


# In[95]:


# Updated apply_partial_augmentation function
def apply_partial_augmentation(train_sequences, augmentation_probability=0.1, augmentation_rate=0.5):
    augmented_data = []

    for seq in train_sequences:
        # Decode token IDs to text
        text = decode_sequence_to_text(seq)

        # Decide whether to augment
        if random.random() < augmentation_rate:
            augmented_text = augment_text_with_synonyms(text, augmentation_probability)
        else:
            augmented_text = text

        # Re-encode augmented text back to token IDs
        # Numericalization using your `word_to_idx` mapping
        augmented_sequence = [word_to_idx.get(word, word_to_idx['<unk>']) for word in augmented_text.split()]

        # Padding or truncating to match the original sequence length
        if len(augmented_sequence) < len(seq):
            augmented_sequence.extend([word_to_idx['<pad>']] * (len(seq) - len(augmented_sequence)))
        else:
            augmented_sequence = augmented_sequence[:len(seq)]

        augmented_data.append(augmented_sequence)

    return augmented_data




# Updated prepare_dataloader function
def prepare_dataloader(train_sequences, train_labels, test_sequences, test_labels,
                       augmentation_probability, augmentation_rate, batch_size, collate_fn):
    # Apply augmentation to training sequences
    augmented_sequences = apply_partial_augmentation(
        train_sequences,
        augmentation_probability=augmentation_probability,
        augmentation_rate=augmentation_rate
    )

    # Create datasets
    augmented_dataset = TextDataset(augmented_sequences, train_labels)
    val_dataset = TextDataset(test_sequences, test_labels)

    # Create DataLoader objects
    train_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader


# In[96]:


# import torch
# from torch.utils.data import Dataset, DataLoader

# class IMDBDataset(Dataset):
#   def __init__(self, sequences, labels):
#     self.sequences = sequences
#     self.labels = labels

#   def __len__(self):
#     return len(self.labels)

#   def __getitem__(self, idx):
#     return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# In[96]:





# In[97]:


# Convert the HuggingFace dataset to input_ids and labels

train_sequences = [item['input_ids'] for item in train_dataset]  # list of sequences
train_labels = [item['label'] for item in train_dataset]  # list of labels


# In[98]:


print(f"word_to_idx: {word_to_idx}")


# In[99]:


print(f"Vocab sizes: {len(word_to_idx)}")


# In[100]:


test_sequences = [item['input_ids'] for item in test_dataset]
test_labels = [item['label'] for item in test_dataset]

# Create PyTorch Dataset
# train_data = IMDBDataset(train_sequences, train_labels)
# test_data = IMDBDataset(test_sequences, test_labels)



# In[101]:


# import torch
# from torch.nn.utils.rnn import pad_sequence

# def collate_fn(batch):

#     # Accessing 'input_ids' and 'labels' if the dataset is structured as dictionaries
#     inputs = [torch.tensor(item['input_ids']) for item in batch]
#     labels = [item['label'] for item in batch]

#     # Padding inputs and converting labels to tensors
#     inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
#     labels = torch.tensor(labels)

#     return inputs_padded, labels


def collate_fn(batch):
    # Extract texts and labels
    texts, labels = zip(*batch)
    # Convert to tensors
    texts_tensor = torch.tensor(texts, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return texts_tensor, labels_tensor


# In[102]:


batch_size = 96

train_loader, val_loader = prepare_dataloader(
    train_sequences,
    train_labels,
    test_sequences,
    test_labels,
    augmentation_probability=0.1,
    augmentation_rate=0.5,
    batch_size=96,
    collate_fn=collate_fn
)

# train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_fn)


# In[103]:


print(len(train_loader))


# In[104]:


# lengths = [len(item['input_ids']) for item in train_dataset]
# print(f"Minimum sequence length: {min(lengths)}")
# print(f"Maximum sequence length: {max(lengths)}")
# print(f"Average sequence length: {sum(lengths) / len(lengths)}")


# In[105]:


# def Dataloader_by_Index(data_loader, target=0):
#     try:
#         print(f"Attempting to retrieve batch {target}")
#         for index, data in enumerate(data_loader):
#             print(f"Current index: {index}, data length: {len(data)}")  # Debugging print
#             if index == target:
#                 print(f"Returning data for index {index}")
#                 return data  # Return the batch when the target index is hit
#     except Exception as e:
#         print(f"Error: {e}")
#     return None


# In[106]:


for batch in train_loader:
    inputs, labels = batch  # Unpack the tuple
    print(f"Shape of inputs: {len(inputs)}")  # Check the batch size
    print(f"Shape of labels: {len(labels)}")  # Check the batch size
    print(f"Sample input: {inputs[0]}")       # Inspect the first input sequence
    print(f"Sample label: {labels[0]}")       # Inspect the first label
    break


# In[107]:


# element1 = Dataloader_by_Index(train_loader, target=1)
# element0 = Dataloader_by_Index(train_loader, target=0)

# print(element1)
# print(element0)


# In[108]:


len(train_loader)


# In[109]:


# for inputs, _ in train_loader:
#     if torch.max(inputs) >= len(vocab):
#         print(f"Out-of-bounds index found in input: {torch.max(inputs)}")
#         break


# In[110]:


import torch.nn as nn


# In[111]:


import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer: single layer, not bidirectional, with batch_first=True
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        self.batch_norm_output = nn.BatchNorm1d(hidden_dim)


        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.7)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.clamp(x, max= self.embedding.num_embeddings - 1)
        # Embedding the input words into dense vectors
        embedded = self.embedding(x)

        # LSTM output: returns all hidden states and the final hidden/cell state
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Extract the final hidden state for each sequence in the batch
        final_hidden_state = lstm_out[:, -1, :]  # Take all features of the last time step


        final_hidden_state = self.batch_norm_output(final_hidden_state)

        # Apply dropout to the final hidden state
        dropped_out = self.dropout(final_hidden_state)

        # Pass the hidden state through a fully connected layer
        output = self.fc(dropped_out)

        # Apply sigmoid activation to get the probability for binary classification
        output = self.sigmoid(output)

        return output


# In[112]:


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)


        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


# In[113]:


def evaluate_model(mode, val_loader, criterion, device):
  model.eval()

  val_loss = 0.0
  with torch.no_grad():
    for inputs, labels in val_loader:
      inputs, labels = inputs.to(device), labels.to(device).float()

      outputs = model(inputs)

      loss = criterion(outputs.squeeze(), labels.float())

      val_loss += loss.item()
  return val_loss / len(val_loader)


# In[114]:


def calculate_accuracy(outputs, labels):
  preds = torch.round(torch.sigmoid(outputs))
  correct = (preds == labels).float().sum()
  acc = correct / len(correct)
  return acc


# In[115]:


print(len(vocab))


# In[116]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
from collections import Counter

# Check if CUDA is available (Colab GPUs use CUDA)
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use 'cuda' for Colab GPU
    print("GPU is available and being used.")
else:
    device = torch.device("cpu")
    print("GPU not found, using CPU instead.")

vocab_size = len(word_to_idx)
model = LSTMClassifier(vocab_size, embedding_dim=100, hidden_dim=128, output_dim=1).to(device)


# In[117]:


num_epochs = 20
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)


# Early stopping implementation
best_val_loss = float('inf')
patience = 10  # Number of epochs to wait before stopping
counter = 0

for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    val_loss = evaluate_model(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pt')  # Save the best model
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pt'))


# In[119]:


# Move model to device
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for texts, labels in train_loader:
        # Move texts and labels to device
        texts, labels = texts.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for texts, labels in val_loader:
            # Move texts and labels to device
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Print epoch results
    print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

    # Scheduler step
    scheduler.step(val_loss / len(val_loader))


# In[120]:


from google.colab import drive
drive.mount('/content/drive')

# Step 2: Load the Notebook File
import nbformat

# Specify the path to your notebook file (ensure the path is correct)
notebook_path = '/content/drive/MyDrive/imdb_lstm.ipynb'  # Modified path - removed space in 'My Drive'

# Load the notebook content
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook_content = nbformat.read(f, as_version=4)

# Step 3: Convert Notebook to Python Code
from nbconvert import PythonExporter

# Create an instance of PythonExporter
exporter = PythonExporter()

# Convert the notebook content to Python code
python_code, _ = exporter.from_notebook_node(notebook_content)

# Step 4: Save the Converted Python Code
# Specify the output Python file path
python_script_path = '/content/drive/MyDrive/converted_notebook.py' # Modified path - removed space in 'My Drive'

# Save the converted Python code
with open(python_script_path, 'w', encoding='utf-8') as f:
    f.write(python_code)

print(f"Notebook has been successfully converted to {python_script_path}")


# In[133]:


pwd


# 

# In[ ]:




