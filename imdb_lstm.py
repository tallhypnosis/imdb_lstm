#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install datasets


# In[3]:


pip --upgrade numpy


# In[4]:


from datasets import load_dataset

# Load IMDB dataset using Hugging Face datasets
imdb_dataset = load_dataset("imdb")

# Access the train and test datasets
train_dataset = imdb_dataset['train']
test_dataset = imdb_dataset['test']

# Print a few examples
for example in train_dataset.select([0, 1, 2]):
    print(f"Label: {example['label']}, Review: {example['text']}")


# In[5]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

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


# In[6]:


max_len = 200

def padding(tokens):
  if len(tokens) < max_len:
    return tokens + ['<pad>'] * (max_len - len(tokens))
  else:
    return tokens[:max_len]

train_dataset = train_dataset.map(lambda x: {'padded_tokens': padding(x['tokens'])})
test_dataset = test_dataset.map(lambda x: {'padded_tokens': padding(x['tokens'])})

print(train_dataset[0]['padded_tokens'])


# In[7]:


from collections import Counter

def build_vocab(data):
  vocab = Counter()
  for chunk in data:
    vocab.update(chunk['padded_tokens'])
  return vocab

vocab = build_vocab(train_dataset)


# In[8]:


print(vocab)


# In[9]:


len(vocab)


# In[10]:


# word_to_idx = {word: i+2 for i, (word, _) in enumerate(vocab.items())}  # +2 to account for padding and unknown tokens
word_to_idx = {word: i+2 for i, word in enumerate(vocab.keys())}
word_to_idx['<pad>'] = 0
word_to_idx['<unk>'] = 1


# In[12]:


def numericalize(tokens):
  return [word_to_idx.get(word, word_to_idx['<unk>']) for word in tokens]


# In[13]:


for item in train_dataset['padded_tokens'][:5]:  # Sample check
    print(numericalize(item))  # Should only contain indices < vocab_size


# In[15]:


train_dataset = train_dataset.map(lambda x: {'input_ids': numericalize(x['padded_tokens'])})
test_dataset = test_dataset.map(lambda x: {'input_ids': numericalize(x['padded_tokens'])})

print(train_dataset[0]['input_ids'])


# In[16]:


import torch
from torch.utils.data import Dataset, DataLoader

class IMDBDataset(Dataset):
  def __init__(self, sequences, labels):
    self.sequences = sequences
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# In[17]:


# Convert the HuggingFace dataset to input_ids and labels

train_sequences = [item['input_ids'] for item in train_dataset]  # list of sequences
train_labels = [item['label'] for item in train_dataset]  # list of labels


# In[18]:


print(f"word_to_idx: {word_to_idx}")


# In[19]:


print(f"Vocab sizes: {len(word_to_idx)}")


# In[20]:


test_sequences = [item['input_ids'] for item in test_dataset]
test_labels = [item['label'] for item in test_dataset]

# Create PyTorch Dataset
train_data = IMDBDataset(train_sequences, train_labels)
test_data = IMDBDataset(test_sequences, test_labels)


# In[21]:


train_data.__getitem__(64)


# In[22]:


print(len(train_data.__getitem__(64)))
print(len(train_data.__getitem__(63)))


# In[23]:


import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):

    # Accessing 'input_ids' and 'labels' if the dataset is structured as dictionaries
    inputs = [torch.tensor(item['input_ids']) for item in batch]
    labels = [item['label'] for item in batch]

    # Padding inputs and converting labels to tensors
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    return inputs_padded, labels


# In[24]:


batch_size = 64

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_fn)


# In[25]:


print(len(train_loader))


# In[26]:


lengths = [len(item['input_ids']) for item in train_dataset]
print(f"Minimum sequence length: {min(lengths)}")
print(f"Maximum sequence length: {max(lengths)}")
print(f"Average sequence length: {sum(lengths) / len(lengths)}")


# In[27]:


def Dataloader_by_Index(data_loader, target=0):
    try:
        print(f"Attempting to retrieve batch {target}")
        for index, data in enumerate(data_loader):
            print(f"Current index: {index}, data length: {len(data)}")  # Debugging print
            if index == target:
                print(f"Returning data for index {index}")
                return data  # Return the batch when the target index is hit
    except Exception as e:
        print(f"Error: {e}")
    return None


# In[28]:


for batch in train_loader:
    print(f"Shape of inputs: {batch[0].shape}")
    print(f"Shape of labels: {batch[1].shape}")
    break  # Just inspect the first batch


# In[29]:


element1 = Dataloader_by_Index(train_loader, target=1)
element0 = Dataloader_by_Index(train_loader, target=0)

print(element1)
print(element0)


# In[30]:


len(train_loader)


# In[31]:


for inputs, _ in train_loader:
    if torch.max(inputs) >= len(vocab):
        print(f"Out-of-bounds index found in input: {torch.max(inputs)}")
        break


# In[32]:


import torch.nn as nn


# In[48]:


import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer: single layer, not bidirectional, with batch_first=True
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

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

        # Apply dropout to the final hidden state
        dropped_out = self.dropout(final_hidden_state)

        # Pass the hidden state through a fully connected layer
        output = self.fc(dropped_out)

        # Apply sigmoid activation to get the probability for binary classification
        output = self.sigmoid(output)

        return output


# In[49]:


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


# In[55]:


def evaluate_model(mode, val_loader, criterion, device):
  model.eval()

  val_loss = 0.0
  with torch.no_grad():
    for inputs, labels in val_loader:
      inputs, lables = inputs.to(device), labels.to(device).float()

      outputs = model(inputs)

      loss = criterion(outputs.squeeze(), labels.float())

      val_loss += loss.item()
  return val_loss / len(val_loader)


# In[56]:


def calculate_accuracy(outputs, labels):
  preds = torch.round(torch.sigmoid(outputs))
  correct = (preds == labels).float().sum()
  acc = correct / len(correct)
  return acc


# In[57]:


print(len(vocab))


# In[58]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
from collections import Counter

device = torch.device("cpu")


vocab_size = len(word_to_idx)  # This should match the full count of indices in word_to_idx
model = LSTMClassifier(vocab_size, embedding_dim=100, hidden_dim=128, output_dim=1).to(device)


# In[59]:


num_epochs = 5
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    val_loss = evaluate_model(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")


# In[39]:


for example in train_dataset['padded_tokens'][:5]:
    numericalized_sequence = numericalize(example)
    print(f"Numericalized sequence: {numericalized_sequence}")
    if any(idx >= vocab_size for idx in numericalized_sequence):
        print("Error: Index out of bounds detected.")
        break


# In[ ]:


for batch in train_loader:
    inputs, labels = batch
    if inputs.max() >= vocab_s:
        print(f"Input max index: {inputs.max()} exceeds vocab size {vocab_size}")
    break


# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[5]:


cd Colab Notebooks


# 

# In[61]:


from google.colab import drive
drive.mount('/content/drive')

# Step 2: Load the Notebook File
import nbformat

# Specify the path to your notebook file (ensure the path is correct)
notebook_path = '/content/drive/My Drive/imdb_lstm.ipynb'

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
python_script_path = '/content/drive/My Drive/converted_notebook.py'

# Save the converted Python code
with open(python_script_path, 'w', encoding='utf-8') as f:
    f.write(python_code)

print(f"Notebook has been successfully converted to {python_script_path}")



# In[62]:


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


# In[ ]:




