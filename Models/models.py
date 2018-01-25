

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

        self.linear = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(25, 1))
        
    def forward(self, x):
        out = self.linear(x)
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
                            nn.Conv1d(1, 16, kernel_size=1),  # (100-(1-1))*16 = 100*16
                            nn.Conv1d(16, 32, kernel_size=5),  # (100-(5-1))*32 = 96*32
                            nn.BatchNorm1d(32),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2))              # |(96-2)/2+1|*32 = 48*32

        self.module2 = nn.Sequential(
                            nn.Conv1d(1, 8, kernel_size=1),  # (100-(1-1))*4 = 100*8
                            nn.Conv1d(8, 16, kernel_size=3),  # (100-(3-1))*8 = 98*16
                            nn.BatchNorm1d(16),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2))              # |(98-2)/2+1|*8 = 49*16

        self.module3 = nn.Sequential(
                            nn.Conv1d(1, 5, kernel_size=1),  # (100-(1-1))*1 = 100*5
                            nn.Conv1d(5, 10, kernel_size=10),# (100-(10-1))*10 = 91*10
                            nn.BatchNorm1d(10),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2))              # |(91-2)/2+1|*10 = 45*10

        self.final_module = nn.Sequential(
                                nn.Linear(48*32+49*16+45*10, 1000),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(1000, 500),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(500, 1))
        
    def forward(self, x):

        inter1 = self.module1(x)
        inter1 = inter1.view(-1, 48*32)

        inter2 = self.module2(x)
        inter2 = inter2.view(-1, 49*16)

        inter3 = self.module3(x)
        inter3 = inter3.view(-1, 45*10)

        out = torch.cat((inter1, inter2, inter3), 1)
        del inter1, inter2, inter3

        out = self.final_module(out)

        return out