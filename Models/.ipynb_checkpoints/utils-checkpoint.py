

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from scipy.special import expit

from copy import deepcopy


#--------------------------------------------------------------
#--------------------------------------------------------------

def train(num_epoch, model, train_loader, optimizer, criterion, valid_loader=None, use_GPU=True):

    if use_GPU:
        model = model.cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('\n>> Learning: {} parameters\n'.format(params))
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

                    print('Epoch: %d, step: %5d, training loss: %.4f, validation AUC-ROC: %.5f' % 
                          (epoch + 1, i + 1, running_loss / 100, roc_auc_score(target, prediction)))
                    running_loss = 0.0
                else:
                    print('Epoch: %d, step: %5d, training loss: %.4f' % 
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0


def train_multitarget(num_epoch, model, train_loader, optimizer, criterion, valid_loader=None, use_GPU=True, target_number=1):

    if use_GPU:
        model = model.cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('\n>> Learning: {} parameters\n'.format(params))
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
            if i % 100 == 99:    # Print every 500 mini-batches

                if valid_loader is not None:

                    labels = valid_loader.dataset.target_tensor

                    predictions = pd.DataFrame(predict(model, valid_loader, use_GPU=use_GPU))

                    score = 0

                    for j in range(target_number):
                        score += roc_auc_score(labels[:,j],predictions.iloc[:,j])*(1/target_number)

                    print('Epoch: %d, step: %5d, training loss: %.4f, validation AUC-ROC: %.5f' % 
                          (epoch + 1, i + 1, running_loss / 100, score))
                    running_loss = 0.0
                else:
                    print('Epoch: %d, step: %5d, training loss: %.4f' % 
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0


def train_autoencoder(num_epoch, model, train_loader, optimizer, criterion, display_step=500, valid_loader=None, use_GPU=True):

    if use_GPU:
        model = model.cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('\n>> Learning: {} parameters\n'.format(params))
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
            encoded, outputs = model(inputs)
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

            if i % display_step == display_step-1:    # Print every 1000 mini-batches

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

                    encoded, outputs = model(inputs)

                    prediction = outputs.data.float().cpu().numpy()
                    target = labels.cpu().numpy()  

                    validation_loss = mean_squared_error(labels, prediction)

                    print('Epoch: %d, step: %5d, training MSE: %.4f, validation MSE: %.4f' % 
                          (epoch + 1, i + 1, running_loss / display_step, validation_loss))
                    running_loss = 0.0
                else:
                    print('Epoch: %d, step: %5d, training MSE: %.4f' % 
                          (epoch + 1, i + 1, running_loss / display_step))
                    running_loss = 0.0


def predict(model, dataset_loader, use_GPU=True):

    model.eval()

    concatenate = False

    # Get the inputs
    for data in dataset_loader:

        if len(data) == 2:
            inputs, other = data
        else:
            inputs = data

        # Wrap them in Variable
        if use_GPU is True:
            inputs = Variable(inputs.cuda(), requires_grad=False)
        else:
            inputs = Variable(inputs, requires_grad=False)

        outputs = model(inputs)

        del inputs

        if use_GPU:
            prediction = outputs.data.cpu().numpy() # probabilities      
        else:
            prediction = outputs.data.numpy() # probabilities      

        if concatenate:
            full_prediction = np.concatenate((full_prediction,prediction), axis=0)
        else:
            full_prediction = prediction
            concatenate = True


    # Compute sigmoid function
    full_prediction = expit(full_prediction)

    return full_prediction


def predict_autoencoder(model, dataset_loader, use_GPU=True):

    model.eval()

    concatenate = False

    # Get the inputs
    for data in dataset_loader:

        if len(data) == 2:
            inputs, other = data
        else:
            inputs = data

        # Wrap them in Variable
        if use_GPU is True:
            inputs = Variable(inputs.cuda(), requires_grad=False)
        else:
            inputs = Variable(inputs, requires_grad=False)

        encoded, outputs = model(inputs)

        del inputs

        if use_GPU:
            prediction = outputs.data.cpu().numpy() # probabilities      
        else:
            prediction = outputs.data.numpy() # probabilities      

        if concatenate:
            full_prediction = np.concatenate((full_prediction,prediction), axis=0)
        else:
            full_prediction = prediction
            concatenate = True

    return full_prediction


def get_encoded(model, dataset_loader, use_GPU=True):

    model.eval()

    concatenate = False

    # Get the inputs
    for data in dataset_loader:

        if len(data) == 2:
            inputs, other = data
        else:
            inputs = data

        # Wrap them in Variable
        if use_GPU is True:
            inputs = Variable(inputs.cuda(), requires_grad=False)
        else:
            inputs = Variable(inputs, requires_grad=False)

        encoded, outputs = model(inputs)

        del inputs, outputs

        if use_GPU:
            prediction = encoded.data.cpu().numpy() # probabilities      
        else:
            prediction = encoded.data.numpy() # probabilities      

        if concatenate:
            final_encoded = np.concatenate((final_encoded,prediction), axis=0)
        else:
            final_encoded = prediction
            concatenate = True

    return final_encoded