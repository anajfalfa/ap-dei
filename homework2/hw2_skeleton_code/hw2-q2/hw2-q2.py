#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

import utils


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = 3, #kernel_size, 
            padding=1, # None, #padding = 1, 
            maxpool=True,
            batch_norm=True,
            dropout=0.1 #0.1 # 0.0
        ):
        super().__init__()
        #super(ConvBlock, self).__init__()
        
        # Q2.1. Initialize convolution, maxpool, activation and dropout layers 
        self.convolution_layer = nn.Conv2d(
            in_channels,    # in_channels = in_channels
            out_channels,   # out_channels=out_channels
            kernel_size = kernel_size, # kernel_size = kernel_size
            stride=1,
            padding=padding)      # padding = padding
        self.activation_func = nn.ReLU()
        if maxpool:
            self.pooling_layer = nn.MaxPool2d (kernel_size=2, stride=2)
        else:
            self.pooling_layer = None
        self.dropout_layer = nn.Dropout2d(p = dropout) #p=0.1 (p=0 default)

        # Q2.2 Initialize batchnorm layer 
        '''if batch_norm:
            self.batch_layer = nn.BatchNorm2d() 
        else:
            self.batch_layer = None'''
        #raise NotImplementedError

    def forward(self, x):
        # input for convolution is [b, c, w, h]
        
        # Implement execution of layers in right order
        x = self.convolution_layer(x)

        '''if self.batch_layer:
            x = self.batch_layer(x) # Confirmar'''

        x = self.activation_func(x)
        
        if self.pooling_layer:
            x = self.pooling_layer(x)
        
        x = self.dropout_layer(x)

        return x
        #raise NotImplementedError


class CNN(nn.Module):

    def __init__(self, dropout_prob, maxpool=True, batch_norm=True, conv_bias=True):
        super(CNN, self).__init__()
        channels = [3, 32, 64, 128]
        fc1_out_dim = 1024
        fc2_out_dim = 512
        self.maxpool = maxpool
        self.batch_norm = batch_norm

        # Initialize convolutional blocks
        self.convblocks = nn.Sequential(
            ConvBlock(in_channels=channels[0], out_channels=channels[1], maxpool=maxpool, batch_norm=batch_norm, dropout=dropout_prob),
            ConvBlock(in_channels=channels[1], out_channels=channels[2], maxpool=maxpool, batch_norm=batch_norm, dropout=dropout_prob),
            ConvBlock(in_channels=channels[2], out_channels=channels[3], maxpool=maxpool, batch_norm=batch_norm, dropout=dropout_prob),
        )
        # -- sequence of 3 conv blocks -- output channel sizes 32 64 128

        self.flatten = nn.Flatten()
        # nr_input_features = nr_output_channels x output_width x output_height

        # Initialize layers for the MLP block
        self.mlp = nn.Sequential(
            nn.Linear(in_features=channels[3]*(48//(2**3))*(48//(2**3)), out_features=fc1_out_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features= fc1_out_dim , out_features=fc2_out_dim),
            nn.ReLU(),
            nn.Linear(in_features=fc2_out_dim, out_features=6)
        )
        '''self.afftransf1 = nn.Linear(out_features=fc1_out_dim) 
        self.activation = nn.ReLU() #?
        self.afftransf2 = nn.Linear(out_features=fc2_out_dim)
        self.dropout = nn.Dropout(p=dropout_prob) #?
        
        self.afftransf3 = nn.Linear(in_features=fc2_out_dim, out_features=classes)'''

        '''
        relu
        dropout dropout_prob=0.1         self.dropout_layer = nn.Dropout2d(p=0.1)
        affine transf fc2_out_dim
        relu
        affine transf nr_classes output Log-Softmax layer
        '''

        # For Q2.2 initalize batch normalization
        '''if batch_norm:
            self.batch = nn.BatchNorm2d() # rever'''
        

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 48, -1)

        # Implement execution of convolutional blocks 
        #for layer in range(len(self.convblocks)):
        #    x= self.convblocks[layer](x)
        #or
        x = self.convblocks(x)

        # Flattent output of the last conv block
        x = self.flatten(x)
        #print("x shape", x.shape)

        # Implement MLP part
        x = self.mlp(x)
        # using flattened vector as input
        '''x = self.afftransf1(x)
        # between batch normalization
        x = self.activation(x)
        x = self.dropout(x)
        x = self.afftransf2(x)
        # between batch normalization

        x = self.activation(x)
        x = self.afftransf3(x)'''

        ''' # affine transformation 1024 output features
            # rectified linear unit activation function 
            # dropout layer with a dropout 

            # affine transformation 512 output features
            # relu activation
            # affine transformation nr_classes followed by an output Log-Softmax Layer
        '''

        # For Q2.2 implement global averag pooling
        '''nn.AdaptiveAvgPool2d (kernel=(1,1))
        # alternative: average output from last conv block over height and width 

        #nn.BatchNorm1d in MLP block
        nn.Identity
        #return x'''

        return F.log_softmax(x, dim=1)
 

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X, return_scores=True):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)

    if return_scores:
        return predicted_labels, scores
    else:
        return predicted_labels


def evaluate(model, X, y, criterion=None):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    with torch.no_grad():
        y_hat, scores = predict(model, X, return_scores=True)
        loss = criterion(scores, y)
        n_correct = (y == y_hat).sum().item()
        n_possible = float(y.shape[0])

    return n_correct / n_possible, loss


def plot(epochs, plottable, ylabel='', name=''):
    plt.figure()#plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def get_number_trainable_params(model):
    raise NotImplementedError


def plot_file_name_sufix(opt, exlude):
    """
    opt : options from argument parser
    exlude : set of variable names to exlude from the sufix (e.g. "device")

    """
    return '-'.join([str(value) for name, value in vars(opt).items() if name not in exlude])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=40, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-no_maxpool', action='store_true')
    parser.add_argument('-no_batch_norm', action='store_true')
    parser.add_argument('-data_path', type=str, default='../intel_landscapes.npz',)
    parser.add_argument('-device', choices=['cpu', 'cuda', 'mps'], default='cpu')

    opt = parser.parse_args()

    # Setting seed for reproducibility
    utils.configure_seed(seed=42)

    # Load data
    data = utils.load_dataset(data_path=opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X.to(opt.device), dataset.dev_y.to(opt.device)
    test_X, test_y = dataset.test_X.to(opt.device), dataset.test_y.to(opt.device)

    # initialize the model
    model = CNN(
        opt.dropout,
        maxpool=not opt.no_maxpool,
        batch_norm=not opt.no_batch_norm
    ).to(opt.device)

    '''
# Definindo o dispositivo (CPU)
device = torch.device("cpu")  # Define que o modelo será executado no CPU
model = model.to(device)      # Move o modelo para o CPU
    '''
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )

    # get a loss criterion
    criterion = nn.NLLLoss()

    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('\nTraining epoch {}'.format(ii))
        model.train()
        for X_batch, y_batch in train_dataloader:
            X_batch = X_batch.to(opt.device)
            y_batch = y_batch.to(opt.device)
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        val_acc, val_loss = evaluate(model, dev_X, dev_y, criterion)
        valid_accs.append(val_acc)
        print("Valid loss: %.4f" % val_loss)
        print('Valid acc: %.4f' % val_acc)

    test_acc, _ = evaluate(model, test_X, test_y, criterion)
    test_acc_perc = test_acc * 100
    test_acc_str = '%.2f' % test_acc_perc
    print('Final Test acc: %.4f' % test_acc)
    # plot
    sufix = plot_file_name_sufix(opt, exlude={'data_path', 'device'})

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-3-train-loss-{}-{}'.format(sufix, test_acc_str))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-3-valid-accuracy-{}-{}'.format(sufix, test_acc_str))

    print('Number of trainable parameters: ', get_number_trainable_params(model))

if __name__ == '__main__':
    main()
