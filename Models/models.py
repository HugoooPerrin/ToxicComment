

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
from sklearn.metrics import log_loss
from scipy.special import expit



#--------------------------------------------------------------
#--------------------------------------------------------------

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(784, 1024)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.linear6 = nn.Linear(64, 10)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.relu3(out)
        out = self.linear4(out)
        out = self.relu4(out)
        out = self.linear5(out)
        out = self.relu5(out)
        out = self.linear6(out)
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 5, 5) # (100-(5-1))*5 = 96*5
        self.conv2 = nn.Conv1d(5, 10, 5) # (96-(5-1))*10 = 92*10
        self.pool2 = nn.MaxPool1d(2, 2) # |(92-2)/2+1|*10 = 46*10
        self.linear1 = nn.Linear(46*10, 200)
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(0.4)
        self.linear2 = nn.Linear(200, 100)
        self.tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(0.4)
        self.linear3 = nn.Linear(100, 1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool2(out)
        out = out.view(-1, 46*10)
        out = self.linear1(out)
        out = self.tanh1(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.tanh2(out)
        out = self.dropout2(out)
        out = self.linear3(out)
        return out