""" 
This code defines a neural network model using PyTorch. The model is designed to be used in the context of a chatbot for intent classification.
"""

# Import torch for tensor operations.
import torch
# torch.nn for building neural networks
import torch.nn as nn


# Define a custom neural network class named NeuralNet that inherits from nn.Module, the base class for all PyTorch models.
class NeuralNet(nn.Module):
    
    """ 
    In the constructor (__init__ method), the class is initialized with three parameters
    input_size: The size of the input features or input layer.
    hidden_size: The size of the hidden layers.
    num_classes: The number of output classes or categories.
    """
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(NeuralNet, self).__init__()
        
        """  
        Define three linear (fully connected) layers:
        self.l1: The first linear layer, which takes the input features and maps them to the hidden layer (hidden_size).
        self.l2: The second linear layer, which maps the output of the first hidden layer to another hidden layer of the same size.
        self.l3: The third linear layer, which maps the output of the second hidden layer to the output layer with num_classes units.
        """
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        
        # Define a Rectified Linear Unit (ReLU) activation function, which introduces non-linearity to the network by applying it element-wise to the output of linear layers.
        self.relu = nn.ReLU()
        
    # In the forward method, define the forward pass of the neural network:
    def forward(self, x):
        
        # Pass the input x through the first linear layer (self.l1).
        out = self.l1(x)
        
        # Apply the ReLU activation function.
        out = self.relu(out)
        
        # Pass the output through the second linear layer
        out = self.l2(out)
        
        # Apply the ReLU activation function.
        out = self.relu(out)
        
        # pass the output through the third linear layer, which produces the raw scores for each class
        out = self.l3(out)
        
        # no activation and no softmax
        # The last layer (self.l3) does not have an activation function applied to it.
        # This is because this model is intended for use in classification tasks where the Cross-Entropy Loss will be used, which includes the softmax activation function. 
        # So, the raw scores are returned without softmax.
        return out
    
"""_summary_
This code defines a simple feedforward neural network with three linear layers and ReLU activation functions.
The network is designed for classification tasks where the final layer's output can be used with a softmax activation and Cross-Entropy Loss for training.
"""