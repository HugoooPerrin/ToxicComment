

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

        self.cnn = nn.Sequential(
                        nn.Conv1d(1, 8, kernel_size=1),  # (100-(1-1))*8 = 100*8
                        nn.Conv1d(8, 16, kernel_size=10),  # (100-(10-1))*16 = 91*16
                        nn.BatchNorm1d(16),
                        nn.ReLU(),
                        nn.MaxPool1d(2, 2))              # |(91-2)/2+1|*16 = 45*16

        self.fc = nn.Sequential(
                        nn.Linear(45*16, 300),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(300, 6))
        
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(-1, 45*16)
        out = self.fc(out)
        return out



class Inception(nn.Module):

    def __init__(self):
        super(Inception, self).__init__()

        self.module1 = nn.Sequential(
                            nn.Conv1d(1, 8, kernel_size=1),  # (100-(1-1))*8 = 100*8
                            nn.Conv1d(8, 16, kernel_size=5),  # (100-(5-1))*16 = 96*16
                            nn.BatchNorm1d(16),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2))              # |(96-2)/2+1|*16 = 48*16

        self.module2 = nn.Sequential(
                            nn.Conv1d(1, 8, kernel_size=1),  # (100-(1-1))*8 = 100*8
                            nn.Conv1d(8, 16, kernel_size=10),  # (100-(10-1))*16 = 91*16
                            nn.BatchNorm1d(16),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2))              # |(91-2)/2+1|*16 = 45*16

        self.module3 = nn.Sequential(
                            nn.Conv1d(1, 8, kernel_size=1),  # (100-(1-1))*8 = 100*8
                            nn.Conv1d(8, 16, kernel_size=20),# (100-(20-1))*16 = 81*16
                            nn.BatchNorm1d(16),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2))              # |(81-2)/2+1|*16 = 40*16

        self.final_module = nn.Sequential(
                                nn.Linear(48*16+45*16+40*16, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512, 6))
        
    def forward(self, x):

        inter1 = self.module1(x)
        inter1 = inter1.view(-1, 48*16)

        inter2 = self.module2(x)
        inter2 = inter2.view(-1, 45*16)

        inter3 = self.module3(x)
        inter3 = inter3.view(-1, 40*16)

        out = torch.cat((inter1, inter2, inter3), 1)
        del inter1, inter2, inter3

        out = self.final_module(out)

        return out