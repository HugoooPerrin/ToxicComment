

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

def train(num_epoch, model, train_loader, optimizer, criterion, valid_loader=None, use_GPU=True):

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('>> PROCESSING LEARNING: {} parameters\n'.format(params))
    i = 0
    for epoch in range(num_epoch):
        total = 0
        running_loss = 0.0
        for data in train_loader:

            model.train()
            # Get the inputs (batch)
            inputs, labels = data
            # Wrap them in Variable
            if use_GPU is True:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward
            outputs = model(inputs)
            # Loss
            loss = criterion(outputs, labels)
            # Backward 
            loss.backward()
            # Optimize
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            i += 1
            if i % 100 == 99:    # Print every 1000 mini-batches

                if valid_loader is not None:
                    model.eval()

                    inputs = valid_loader.dataset.data_tensor
                    labels = valid_loader.dataset.target_tensor
                    labels = labels
                    # Wrap them in Variable
                    if use_GPU is True:
                        inputs = Variable(inputs.cuda(), requires_grad=False)
                    else:
                        inputs = Variable(inputs, requires_grad=False)

                    outputs = model(inputs)

                    prediction = outputs.data.float() # probabilities             
                    prediction = expit(prediction.cpu().numpy())
                    target = labels.cpu().numpy()    

                    print('Epoch: %d, step: %5d, training loss: %.4f, validation loss: %.5f' % 
                          (epoch + 1, i + 1, running_loss / 100, log_loss(target, prediction)))
                    running_loss = 0.0
                else:
                    print('Epoch: %d, step: %5d, training loss: %.4f' % 
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0


def predict(model, dataset):

    model = model.cpu()

    model.eval()
    # Get the inputs
    inputs = dataset
    # Wrap them in Variable
    inputs = Variable(inputs, requires_grad=False)   

    outputs = model(inputs)
    
    prediction = outputs.data.numpy() # probabilities             
    prediction = expit(prediction)

    return prediction