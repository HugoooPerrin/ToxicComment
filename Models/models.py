

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
        self.fc = nn.Sequential(
            nn.Linear(46*10, 200),
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(200, 100),
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(100, 1))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool2(out)
        out = out.view(-1, 46*10)
        out = self.fc(out)
        return out



class Inception(nn.Module):

    def __init__(self):

        super(Inception, self).__init__()

        self.module1 = nn.Sequential(
                            nn.Conv1d(1, 5, kernel_size=1), # (100-(1-1))*5 = 100*5
                            nn.Conv1d(5, 5, kernel_size=5), # (100-(5-1))*5 = 96*5
                            nn.BatchNorm1d(5),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2))             # |(96-2)/2+1|*10 = 48*5

        self.module2 = nn.Sequential(
                            nn.Conv1d(1, 4, kernel_size=1), # (100-(1-1))*4 = 100*4
                            nn.Conv1d(4, 8, kernel_size=3), # (100-(3-1))*8 = 98*8
                            nn.BatchNorm1d(8),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2))             # |(98-2)/2+1|*8 = 49*8

        self.final_module = nn.Sequential(
                                nn.Linear(48*5+49*8, 500),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(500, 100),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(100, 1))
        
    def forward(self, x):

        inter1 = self.module1(x)
        inter1 = inter1.view(-1, 48*5)

        inter2 = self.module2(x)
        inter2 = inter2.view(-1, 49*8)

        out = torch.cat((inter1, inter2), 1)
        del inter1, inter2

        out = self.final_module(out)

        return out