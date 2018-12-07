
# coding: utf-8

# Import modules

import pandas as pd
import numpy as np
from numpy.random import permutation
import sys
import nltk
import pickle
import time

import torch
import torch.nn as nn
import torch.utils.data as utils
import torchwordemb
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import log_loss, roc_auc_score

from multiprocessing import Pool

sys.path.append('/home/hugoperrin/Bureau/DataScience/Kaggle/ToxicComment/Models/')
from models import *

sys.path.append('/home/hugoperrin/Bureau/DataScience/Kaggle/ToxicComment/Models/')
from utils import *

# Import data (1D)
train_vect = np.load('/home/hugoperrin/Bureau/Datasets/ToxicComment/Comment2Vec_train.npy')

# Import data (2D)
# train_vect = np.load('/home/hugoperrin/Bureau/Datasets/ToxicComment/Comment2Vec_train_vM.npy')

Xtrain = pd.read_csv('/home/hugoperrin/Bureau/Datasets/ToxicComment/train.csv')
list_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
targets = Xtrain[list_classes].values

del Xtrain

# print(train_vect.shape)

# Preprocess data for 1D convolution
train_vect = train_vect.reshape(train_vect.shape[0],1,train_vect.shape[1])

# Preprocess data for 2D convolution
# train_vect = train_vect.reshape(train_vect.shape[0],1,train_vect.shape[1], train_vect.shape[2])

time1 = time.time()

# Cross validation loop
CV = 3

CV_score = 0

for i in range(CV):

    print('\n---------------------------------------------------\nLoop number {}'.format(i+1))

    random_order = permutation(len(train_vect))

    # # Train test split
    train_comments = train_vect[random_order[:120000]]
    valid_comments = train_vect[random_order[120001:135000]]
    test_comments = train_vect[random_order[135001:]]

    train_labels = targets[random_order[:120000]]
    valid_labels = targets[random_order[120001:135000]]
    test_labels = targets[random_order[135001:]]

    # Get final predictions

    use_GPU = True

    batch_size = 256
    num_epoch = 4

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_comments), 
                                                   torch.FloatTensor(train_labels))

    valid_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(valid_comments), 
                                                   torch.FloatTensor(valid_labels))

    test_dataset = torch.FloatTensor(test_comments)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               num_workers = 8)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False, 
                                               num_workers = 8)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False, 
                                               num_workers = 8)

    net = CNN()

    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.02, momentum=0.9)
    optimizer = optim.RMSprop(net.parameters(), lr=0.00005, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.9)

    train_multitarget(num_epoch, net, train_loader, optimizer, criterion, 
                            valid_loader=valid_loader, use_GPU=use_GPU, target_number=6)

    predictions = pd.DataFrame(predict(net, test_loader, use_GPU=use_GPU))

    score = 0

    for i in range(len(list_classes)):
        score += roc_auc_score(test_labels[:,i],predictions.iloc[:,i])*(1/len(list_classes))

    CV_score += score*(1/CV)

    print("\nModel intermediate score: {}".format(round(score,5)))


print("\nModel final score: {}\n".format(round(CV_score,5)))


time2 = time.time()
diff_time = (time2 - time1)/60
print("Training time is {} minutes\n".format(round(diff_time,1)))
