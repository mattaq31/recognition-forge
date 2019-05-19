"""
Portions of this code have been adapated from https://github.com/CSTR-Edinburgh/mlpractical,
which is copyright (c) University of Edinburgh 2015-2018. Licensed under the 
Modified BSD License. https://opensource.org/licenses/BSD-3-Clause
"""
from torch import nn
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import tqdm
from torch.autograd import Variable
import os

class LSTMNetwork(nn.Module):
    def __init__(self, hidden_dim, use_gpu, num_layers, vocab_size, n_seqs, dropout=0):
        super(LSTMNetwork, self).__init__()

        self.hidden_dim = hidden_dim  # hidden dimension num features
        self.n_seqs = n_seqs  # batch size
        self.num_layers = num_layers  # number of LSTM layers
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.output_net = nn.Linear(hidden_dim, vocab_size)  # final output net
        self.set_device(use_gpu)
        self.hidden = self.init_hidden()

    def set_device(self, use_gpu):
        if torch.cuda.is_available() and use_gpu:  # checks whether a cuda gpu is available and whether the gpu flag is True
            self.device = torch.device('cuda')  # sets device to be cuda
            os.environ[
                "CUDA_VISIBLE_DEVICES"] = "0"  # sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use by using the relevant GPU ID)
            print("model set to GPU")
        else:
            print("model set to CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU

    def init_hidden(self, n_seqs=None):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if not n_seqs:
            n_seqs = self.n_seqs

        return (torch.zeros(self.num_layers, n_seqs, self.hidden_dim).to(device=self.device),
                torch.zeros(self.num_layers, n_seqs, self.hidden_dim).to(device=self.device))

    def renew_hidden(self, n_seqs=None):
        return tuple([Variable(each.data) for each in self.hidden])

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        self.output_net.reset_parameters()

    def forward(self, in_s):

        out, self.hidden = self.lstm(in_s, self.hidden)
        out = self.dropout(out)
        out = out.view(out.size()[0] * out.size()[1], self.hidden_dim) #flattening out output for linear layer
        out = self.output_net(out)
        return out

class FCCNetwork(nn.Module):
    def __init__(self, input_shape, num_output_classes, num_hidden_units, num_layers, use_bias=False):
        """
        Initializes a fully connected network similar to the ones implemented previously in the MLP package.
        :param input_shape: The shape of the inputs going in to the network.
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_hidden_units: Number of units used in every hidden layer.
        :param num_layers: Number of fcc layers (excluding dim reduction stages)
        :param use_bias: Whether our fcc layers will use a bias.
        """
        super(FCCNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_units = num_hidden_units
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        print("Building basic block of FCCNetwork using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))

        out = x
        out = out.view(out.shape[0], -1)
                         # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            self.layer_dict['fcc_{}'.format(i)] = nn.Linear(in_features=out.shape[1],  # initialize a fcc layer
                                                            out_features=self.num_units,
                                                            bias=self.use_bias)

            out = self.layer_dict['fcc_{}'.format(i)](out)  # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        self.logits_linear_layer = nn.Linear(in_features=out.shape[1],  # initialize the prediction output linear layer
                                             out_features=self.num_output_classes,
                                             bias=self.use_bias)
        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward prop data through the network and return the preds
        :param x: Input batch x a batch of shape batch number of samples, each of any dimensionality.
        :return: preds of shape (b, num_classes)
        """
        out = x
        if len(out.shape) > 1:
            out = out.view(out.shape[0], -1)

        # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            out = self.layer_dict['fcc_{}'.format(i)](out)  # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()

        self.logits_linear_layer.reset_parameters()