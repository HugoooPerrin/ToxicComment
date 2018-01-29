
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

from sklearn.metrics import log_loss

from multiprocessing import Pool

sys.path.append('/home/hugoperrin/Bureau/DataScience/Kaggle/ToxicComment/Models/')
from models import CNN

sys.path.append('/home/hugoperrin/Bureau/DataScience/Kaggle/ToxicComment/Models/')
from utils import train, predict

time1 = time.time()

# Import data
train_vect = np.load('/home/hugoperrin/Bureau/Datasets/ToxicComment/Comment2Vec_train.npy')

Xtrain = pd.read_csv('/home/hugoperrin/Bureau/Datasets/ToxicComment/train.csv')
list_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
targets = Xtrain[list_classes].values

del Xtrain

# Preprocess data for 1D convolution
train_vect = train_vect.reshape(train_vect.shape[0],1,train_vect.shape[1])

# Cross validation loop
CV = 4

CV_score = 0

for i in range(CV):

    print('---------------------------------------------------\nLoop number {}'.format(i+1))

    random_order = permutation(len(train_vect))

    # # Train test split
    train_comments = train_vect[random_order[:120000],:,:]
    valid_comments = train_vect[random_order[120001:135000],:,:]
    test_comments = train_vect[random_order[135001:]]

    train_labels = targets[random_order[:120000],:]
    valid_labels = targets[random_order[120001:135000],:]
    test_labels = targets[random_order[135001:],:]

    # Get final predictions
    predictions = pd.DataFrame(index=range(len(test_comments)))

    for target in list_classes:
        
        print("\n\nEstimation of {}:".format(target))

        labels_train = train_labels[:,list_classes.index(target)]
        labels_train = labels_train.reshape(labels_train.shape[0],1)

        labels_valid = valid_labels[:,list_classes.index(target)]
        labels_valid = labels_valid.reshape(labels_valid.shape[0],1)

        labels_test = test_labels[:,list_classes.index(target)]
        labels_test = labels_test.reshape(test_labels.shape[0],1)

        use_GPU = True

        batch_size = 512
        num_epoch = 4

        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_comments), 
                                                       torch.FloatTensor(labels_train))

        valid_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(valid_comments), 
                                                       torch.FloatTensor(labels_valid))

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
        optimizer = optim.RMSprop(net.parameters(), lr=0.000015, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.9)

        train(num_epoch, net, train_loader, optimizer, criterion, valid_loader=valid_loader, use_GPU=use_GPU)

        predictions[target] = predict(net, test_loader, use_GPU=use_GPU)

    score = 0

    for i in range(len(list_classes)):
        score += log_loss(test_labels[:,i],predictions.iloc[:,i])*(1/len(list_classes))

    CV_score += score*(1/CV)

    print("\n\nModel intermediate score: {}\n".format(round(score,5)))


print("\nModel final score: {}\n".format(round(CV_score,5)))


time2 = time.time()
diff_time = (time2 - time1)/60
print("\nTraining time is {} minutes\n".format(round(diff_time,1)))
