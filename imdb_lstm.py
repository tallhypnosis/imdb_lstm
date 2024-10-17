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


vocab


# In[9]:


word_to_idx = {word: i+2 for i, (word, _) in enumerate(vocab.items())}
word_to_idx['<pad>'] = 0
word_to_idx['<unk>'] = 1


# In[10]:


def numericalize(tokens):
  return [word_to_idx.get(word, word_to_idx['<unk>']) for word in tokens]


# In[11]:


train_dataset = train_dataset.map(lambda x: {'input_ids': numericalize(x['padded_tokens'])})
test_dataset = test_dataset.map(lambda x: {'input_ids': numericalize(x['padded_tokens'])})

print(train_dataset[0]['input_ids'])


# In[12]:


train_dataset[0]


# In[13]:


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


# In[14]:


# Convert the HuggingFace dataset to input_ids and labels

train_sequences = [item['input_ids'] for item in train_dataset]  # list of sequences
train_labels = [item['label'] for item in train_dataset]  # list of labels


# In[25]:


train_sequences[63]


# In[15]:


test_sequences = [item['input_ids'] for item in test_dataset]
test_labels = [item['label'] for item in test_dataset]

# Create PyTorch Dataset
train_data = IMDBDataset(train_sequences, train_labels)
test_data = IMDBDataset(test_sequences, test_labels)


# In[28]:


train_data.__getitem__(64)


# In[16]:


print(len(train_data.__getitem__(64)))
print(len(train_data.__getitem__(63)))


# In[60]:


import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    print(f"Batch content: {batch}")

    # Accessing 'input_ids' and 'labels' if the dataset is structured as dictionaries
    inputs = [torch.tensor(item['input_ids']) for item in batch]
    labels = [item['label'] for item in batch]

    # Padding inputs and converting labels to tensors
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    return inputs_padded, labels


# In[61]:


batch_size = 64

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_fn)


# In[55]:


print(len(train_loader))


# In[56]:


lengths = [len(item['input_ids']) for item in train_dataset]
print(f"Minimum sequence length: {min(lengths)}")
print(f"Maximum sequence length: {max(lengths)}")
print(f"Average sequence length: {sum(lengths) / len(lengths)}")


# In[65]:


for inputs, labels in train_loader:
    print("Input shape:", inputs.shape)    # The shape of input data
    print("Labels shape:", labels.shape)   # The shape of the labels
    print("Input size:", inputs.size())    # The size of input data (e.g., batch size x sequence length)
    print("Labels size:", labels.size())

train_loader[63]


# In[62]:


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


# In[63]:


for batch in train_loader:
    print(f"Shape of inputs: {batch[0].shape}")
    print(f"Shape of labels: {batch[1].shape}")
    break  # Just inspect the first batch


# In[64]:


element1 = Dataloader_by_Index(train_loader, target=1)
element0 = Dataloader_by_Index(train_loader, target=0)

print(element1)
print(element0)


# In[82]:


from google.colab import drive
drive.mount('/content/drive')


# In[87]:


import os
os.listdir('/content/drive/My Drive/Colab Notebooks/imdb_lstm.ipynb')


# In[84]:


import nbformat
from nbconvert import PythonExporter

# Load the notebook
with open("content/drive/My Drive/imdb_lstm.ipynb") as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert to Python script
python_exporter = PythonExporter()
(script, resources) = python_exporter.from_notebook_node(notebook_content)

# Save the script to a .py file
with open("imdb_lstm.py", "w") as f:
    f.write(script)


# In[72]:


ls


# In[71]:





# In[ ]:




