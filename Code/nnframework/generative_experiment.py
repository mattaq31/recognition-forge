"""
This module builds up an experiment which trains and validates an LSTM, and allows for use at run-time.
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
import os
import shutil
import numpy as np
import time
from nnframework.model_architectures import LSTMNetwork


class GenExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, gen_data, data_provider, num_epochs=25, n_seqs=50, n_steps=50,
                 weight_decay_coefficient=1e-05, use_gpu=False, continue_from_epoch=-1, save_all_models=False, suppress_logs=True,
                 cleanfiles=False, validation_fraction=0.1, load=False):

        super(GenExperimentBuilder, self).__init__()
        if torch.cuda.is_available() and use_gpu:  # checks whether a cuda gpu is available and whether the gpu flag is True
            self.device = torch.device('cuda')  # sets device to be cuda
            os.environ[
                "CUDA_VISIBLE_DEVICES"] = "0"  # sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use by using the relevant GPU ID)
            print("architecture set to GPU")
        else:
            print("architecture set to CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU

        self.use_gpu = use_gpu
        self.experiment_name = experiment_name
        self.data = gen_data
        self.data_provider = data_provider
        self.n_steps = n_steps  # sequence length (string length)
        self.n_seqs = n_seqs  # batch size
        if load:
            self.model = self.load_model()
        else:
            self.model = network_model
            self.model.reset_parameters()
             # re-initialize network parameters
        self.model.to(self.device)  # sends the model from the cpu to the gpu

        self.save_all_models = save_all_models
        self.validation_fraction = validation_fraction  # how much validation data to hold back
        self.val_idx = int(len(self.data) * (1 - self.validation_fraction))
        self.train_data, self.val_data = self.data[:self.val_idx], self.data[self.val_idx:]

        # suppression flags
        self.cleanfiles = cleanfiles
        self.suppress_logs = suppress_logs

        self.optimizer = optim.Adam(self.parameters(), amsgrad=False,
                                    weight_decay=0, lr=0.002)

        self.num_epochs = num_epochs  # epochs to train

        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        self.helpful_softmax = nn.Softmax(dim=1)  # for use in transformations
        self.starting_epoch = 0

    def save_model(self, training_losses):
        checkpoint = {'n_hidden': self.model.hidden_dim,
                      'n_layers': self.model.num_layers,
                      'dropout': self.model.dropout_p,
                      'training_stats': training_losses,
                      'state_dict': self.model.state_dict(),
                      }
        with open(self.experiment_name, 'wb') as f:
            torch.save(checkpoint, f)

    def load_model(self):
        with open(self.experiment_name, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
        model = LSTMNetwork(hidden_dim=checkpoint['n_hidden'], num_layers=checkpoint['n_layers'], use_gpu=self.use_gpu,
                            vocab_size=self.data_provider.n_letters, n_seqs=self.n_seqs, dropout=checkpoint['dropout'])
        model.load_state_dict(checkpoint['state_dict'])

        return model

    def run_train_iter_v2(self, x, y):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape (n_seqs,n_steps)
        :param y: The targets for the model. A numpy array of shape (n_seqs,n_steps)
        :return: the loss and accuracy for this batch
        """
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        x, y = torch.Tensor(self.data_provider.get_one_hot(x, num_classes=self.data_provider.n_letters)).float().to(device=self.device),\
               torch.Tensor(y).long().to(device=self.device)  # send data to device as torch tensors

        self.model.hidden = self.model.renew_hidden()

        self.optimizer.zero_grad()  # clear out lingering gradients
        
        out = self.model.forward(x)  # forward the data through the model
        
        loss = F.cross_entropy(input=out, target=y.view(self.n_seqs*self.n_steps))  # flattened arrays for loss
        loss.backward()  # backpropagate to compute gradients for current iter loss
        nn.utils.clip_grad_norm(self.model.parameters(), 5)  # limits exploding gradients
        self.optimizer.step()  # update network parameters

        prediction = self.helpful_softmax(out).cpu().detach().numpy()

        pred_targets = y.view(self.n_seqs*self.n_steps).cpu().detach().numpy()
        char_prediction = np.zeros(self.n_seqs*self.n_steps)
        for i in range(prediction.shape[0]):
            char_prediction[i] = np.random.choice(np.arange(self.data_provider.n_letters), p=prediction[i, :])
        accuracy = 100*(np.sum(char_prediction == pred_targets)/(self.n_seqs*self.n_steps))

        return loss.cpu().detach().numpy(), accuracy

    def run_prediction_iter(self, x):
        self.eval()
        x = torch.Tensor(self.data_provider.get_one_hot(x, num_classes=self.data_provider.n_letters)).float().to(
            device=self.device)

        out = self.model.forward(x.view(1, 1, -1))  # forward the data in the model

        prediction = self.helpful_softmax(out).cpu().detach().numpy().squeeze()

        return np.random.choice(np.arange(self.data_provider.n_letters), p=prediction)

    def run_gen_train(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        """
        training_losses = {"train_acc": [], "train_loss": []}  # initialize a dict to keep the per-epoch metrics

        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):

            current_epoch_losses = {"train_acc": [], "train_loss": []}
            self.model.hidden = self.model.init_hidden()  # clear out hidden layer
            if self.suppress_logs:
                for idx, (x, y) in enumerate(self.data_provider.get_batches(self.train_data, self.n_seqs, self.n_steps)):  # get data batches
                    loss, accuracy = self.run_train_iter_v2(x=x, y=y)  # take a training iter step
                    current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                    current_epoch_losses["train_acc"].append(accuracy)  # add current iter acc to the train acc list
            else:
                print('\nTraining run, Epoch', i)
                with tqdm.tqdm(total=len(self.train_data)//(self.n_seqs*self.n_steps)) as pbar_train:  # create a progress bar for training
                    for idx, (x, y) in enumerate(self.data_provider.get_batches(self.train_data, self.n_seqs, self.n_steps)):  # get data batches
                        loss, accuracy = self.run_train_iter_v2(x=x, y=y)  # take a training iter step
                        current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                        current_epoch_losses["train_acc"].append(accuracy)  # add current iter acc to the train acc list
                        pbar_train.update(1)
                        pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
            for key, value in current_epoch_losses.items():
                training_losses[key].append(np.mean(value))

        self.save_model(training_losses)

        return current_epoch_losses

    def gen_run(self, seed, length):
        """
        Generates continuation of a seed input at run-time using trained LSTM.
        :param seed: Initial input to base generations on.
        :param length: The total length of generations requested.
        :return: the final generated sequence.
        """
        num_seed = self.data_provider.encode_any(seed) #encode seed into character provide setting

        self.model.hidden = self.model.init_hidden(1)

        for j in num_seed:
            output = self.run_prediction_iter(j)  #run seed through LSTM
        new_char = self.data_provider.decode_any(output)

        seed = seed + new_char  #update seed with first generated character

        for i in range(length-1): #continue generating characters for as long as specified.
            num_seed = self.data_provider.encode_any(seed[-1])
            output = self.run_prediction_iter(num_seed)
            seed = seed + self.data_provider.decode_any(output) #concatenates generations.

        return seed







