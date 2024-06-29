import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from config import *


class SentimentClassifier(nn.Module):
    """
    Simple sentiment classifier, takes as input as a representation (maybe cls) and outputs probability scores
    """
    def __init__(self, hidden_size = hidden_size):
        super(SentimentClassifier,self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(int(hidden_dim),hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,num_polarities),
            #nn.Softmax(dim=-1)
        )
        self.init_weight(self.classifier)

    def init_weight(self,seq):
        for layer in seq:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight,a=-0.1,b=0.1)  # Uniform initialization for weights in range [-0.1, 0.1]
                #nn.init.normal_(layer.weight)
                
                nn.init.constant_(layer.bias, 0)

    def forward(self, representation: torch.Tensor):
        """
        :param representation: [batch_size x 2*hidden_dim]
        
        :return predicted_polarity: [batch_size x num_polarities]
        """
        predicted_polarity = self.classifier(representation)
        return predicted_polarity