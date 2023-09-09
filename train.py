# import modules and libraries
import json
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load the chatbot intents from a JSON file named 'intents.json'. 
with open('intents.json', 'r') as f:
    intents = json.load(f)

"""
1. Extract intent tags and tokenized patterns from the JSON file. The intents dictionary is iterated over, and for each intent, the tag and patterns are extracted and stored.
2. The tokenize function is used to split the text patterns into individual words, which are then added to the all_words list.
3. The xy list stores pairs of tokenized words and their corresponding intent tags.
"""
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

# Define a list of words to ignore during processing. 
ignore_words = ['?', '!', '.', ',']


"""
1. Perform word stemming and remove the ignore words from the all_words list. Stemming reduces words to their base form (e.g., "running" to "run").
2. Sort and convert both all_words and tags into sets to remove duplicates.
"""
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


""" 
1. Prepare the training data:
2. Convert the tokenized pattern sentences into a "bag of words" representation using the bag_of_words function. This function creates a numerical vector where each element represents the presence or absence of a word from all_words in the input sentence.
3. Convert intent tags into numerical labels using the index method of the tags list.
"""
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)


# Convert the training data into NumPy arrays for compatibility with PyTorch.
X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
""" 
Set hyperparameters for training, including batch size, hidden layer size, output size (number of unique intent tags), input size (size of the bag of words vector), learning rate, and the number of training epochs.
"""

batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000


""" 
Define a custom PyTorch dataset class, ChatDataset, for handling the training data. This class implements the necessary methods (__init__, __getitem__, and __len__) required for custom datasets in PyTorch.
"""
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

# Create an instance of the ChatDataset and use the DataLoader to batch and shuffle the data for training.
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

""" 
1. Check if a GPU is available and set the device accordingly.
2. Instantiate the NeuralNet model defined in the model.py module and move it to the selected device (GPU or CPU).
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
# Define the loss function (cross-entropy loss) and the optimizer (Adam optimizer) for training the model.
criteriaon = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


""" 
1. Train the model in a loop over the specified number of epochs.
2. For each batch of data, move it to the selected device, compute the model's outputs, calculate the loss, perform backpropagation, and update the model's weights using the optimizer.
3. Print the loss every 100 epochs to monitor training progress.
"""
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


"""
1. Save the trained model's state, along with relevant metadata, to a file named "data.pth".
2. Print a message indicating that training is complete and that the file has been saved.
"""
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f"Training Complete. file saved to {FILE}")

"""
this code loads chatbot training data, preprocesses it, trains a neural network model to classify intents, and saves the trained model to a file
"""