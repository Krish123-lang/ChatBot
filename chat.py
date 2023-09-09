# Import Modules
import json
import random
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words
import torch

# Check if a GPU (CUDA) is available, and set the device to GPU if available; otherwise, use CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the chatbot intents from a JSON file named 'intents.json'.
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load pre-trained data from a saved PyTorch file named "data.pth." This data includes information about the neural network model and associated metadata.
FILE = "data.pth"
data = torch.load(FILE)

# Extract relevant information from the loaded data, including model architecture details (input_size, hidden_size, output_size), vocabulary (all_words), intent tags (tags), and the model's state (model_state).
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

""" 
1. Create an instance of the NeuralNet model with the architecture details extracted from the loaded data.
2. Load the pre-trained model's state using load_state_dict to initialize the model's weights and biases.
3. Set the model to evaluation mode using model.eval(). This is important for inference, as it disables dropout and batch normalization.
"""
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Define the chatbot's name, which can be used in responses.
bot_name = "Beast"

""" 
1. Define a function get_response that takes a user's message (msg) as input and generates a response.
2. Tokenize the input message and convert it into a bag-of-words representation.
3. Pass the bag-of-words through the pre-trained neural network model (model) to get an output.
4. Calculate the highest predicted class probability and associated tag.
5. If the confidence level (probability) is above 0.75, randomly select a response from the appropriate intent in the loaded intents.
6. If the confidence is below 0.75 or if no matching intent is found, return a default "I do not understand!" response.
"""
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
            
    return "I do not understand!"  

"""_summary_
this script loads a pre-trained neural network model and uses it to provide responses to user input messages based on predefined intents and responses stored in a JSON file. The confidence threshold (0.75) controls whether the chatbot provides a response or not, based on the model's confidence in its prediction.
"""