1. Stemming, Tokenization and Bag of Words
```
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    pass

words = ['organize', 'organs', 'organizing', 'organism']
a = [stem(w) for w in words]
print(a)
```


2. Create Training Data

* nltk_utils.py
```

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
```

* Create new file 'train.py'
```
import json
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag),)

ignore_words = ['?', '!', '.', ',']  # Need to modify this

all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(tags)

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000


class ChatDataset(Dataset):
    def __init__(self) -> None:
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # dataset[idx]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
```


3. Pytorch model and training
* create models.py
```
import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        
        # no activation and no softmax
        return out
```
* train.py
- append the code
```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criteriaon = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(words)
        loss = criteriaon(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}")

print(f"final loss, loss={loss.item():.4f}")
```


4. Save/Load model and implement the chat