

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

            del inputs, labels

            # Backward 
            loss.backward()
            # Optimize
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            i += 1
            if i % 100 == 99:    # Print every 100 mini-batches

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


def predict(model, dataset_loader, use_GPU=True):

    model.eval()

    concatenate = False

    # Get the inputs
    for data in dataset_loader:
        inputs = data
        # Wrap them in Variable
        if use_GPU is True:
            inputs = Variable(inputs.cuda(), requires_grad=False)
        else:
            inputs = Variable(inputs, requires_grad=False)

        outputs = model(inputs)

        del inputs

        if use_GPU is True:
            prediction = outputs.data.cpu().numpy() # probabilities      
        else:
            prediction = outputs.data.numpy() # probabilities      

        if concatenate:
            full_prediction = numpy.concatenate((full_prediction,prediction), axis=0)
        else:
            full_prediction = prediction

    # Compute sigmoid function
    full_prediction = expit(prediction)

    return prediction