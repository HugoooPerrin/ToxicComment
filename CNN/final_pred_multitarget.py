
# coding: utf-8

# Import modules

import pandas as pd
import numpy as np
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
from models import *

sys.path.append('/home/hugoperrin/Bureau/DataScience/Kaggle/ToxicComment/Models/')
from utils import *

time1 = time.time()

# Import data
train_vect = np.load('/home/hugoperrin/Bureau/Datasets/ToxicComment/Comment2Vec_train.npy')
test_vect = np.load('/home/hugoperrin/Bureau/Datasets/ToxicComment/Comment2Vec_test.npy')

Xtrain = pd.read_csv('/home/hugoperrin/Bureau/Datasets/ToxicComment/train.csv')
list_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
labels_train = Xtrain[list_classes].values

Xtest = pd.read_csv('/home/hugoperrin/Bureau/Datasets/ToxicComment/test.csv')
final_id = Xtest['id']

del Xtrain, Xtest

# Preprocess data for torch
train_comments = train_vect.reshape(train_vect.shape[0],1,train_vect.shape[1])
test_comments = test_vect.reshape(test_vect.shape[0],1,test_vect.shape[1])

del train_vect, test_vect

# Get final predictions

use_GPU = True

batch_size = 512
num_epoch = 5

train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_comments), 
                                               torch.FloatTensor(labels_train))

test_dataset = torch.FloatTensor(test_comments)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True, 
                                           num_workers = 8)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False, 
                                           num_workers = 8)

net = CNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(net.parameters(), lr=0.0001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.9)

train_multitarget(num_epoch, net, train_loader, optimizer, criterion, 
                        valid_loader=None, use_GPU=use_GPU, target_number=6)

predictions = pd.DataFrame(predict(net, test_loader, use_GPU=use_GPU), index=final_id)
predictions.columns = list_classes

predictions.to_csv('/home/hugoperrin/Bureau/Datasets/ToxicComment/7th_submission.csv')

time2 = time.time()
diff_time = (time2 - time1)/60
print("\nTraining time is {} minutes\n".format(round(diff_time,1)))