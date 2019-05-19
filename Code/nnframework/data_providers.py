"""
Contains data providers from data objects, direct text and files

Portions of this code has been adapted from https://github.com/CSTR-Edinburgh/mlpractical,
which is copyright (c) University of Edinburgh 2015-2018. Licensed under the 
Modified BSD License. https://opensource.org/licenses/BSD-3-Clause
"""

import pickle
import gzip
import numpy as np
import os
import csv
import string
import imblearn.over_sampling as imbl
import constants as const
import pandas as pd
import pickle

DEFAULT_SEED = 20190211


class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        self.initialize_seed(rng)
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def __next__(self):
        return self.next()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch
    
    def initialize_seed(self, rng): 
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng


class RecognitionDataProvider(DataProvider):
    """Data provider for recognitions."""

    def __init__(self, inputs, targets, batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, oversample=None):
        """Create a new recognition data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        if not oversample is None:
            oversample = oversample.lower()
            self.initialize_seed(rng)

            if oversample == "smote":
                oversampler = imbl.SMOTE(random_state=self.rng)
            elif oversample == "smote-cat":
                # Need method for specifying categorical attributes, e.g., imbl.SMOTENC(random_state=self.rng, categorical_features=range(4200, 4348))
                raise(NotImplementedError)
            elif oversample == "smote-svm":
                oversampler = imbl.SVMSMOTE(random_state=self.rng)
            elif oversample == "smote-borderline-1":
                oversampler = imbl.BorderlineSMOTE(random_state=self.rng, kind='borderline-1')
            elif oversample == "smote-borderline-2":
                oversampler = imbl.BorderlineSMOTE(random_state=self.rng, kind='borderline-2')
            elif oversample == "adasyn":
                oversampler = imbl.ADASYN(random_state=self.rng)
            else:
                raise(Exception("Unrecognized oversampling method: {0}".format(oversample)))

            inputs, targets = oversampler.fit_resample(inputs, targets)
        
        self.num_classes = 3
        inputs = inputs.astype(np.float32)

        # pass the loaded data to the parent class __init__
        super(RecognitionDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(RecognitionDataProvider, self).next()
        return inputs_batch, self.to_one_of_k(targets_batch)

    def to_one_of_k(self, int_targets):
        """Converts integer coded class target to 1 of K coded targets.

        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,)

        Returns:
            Array of 1 of K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """
        int_targets = int_targets.astype(int)
        one_of_k_targets = np.zeros((int_targets.shape[0], self.num_classes))

        one_of_k_targets[range(int_targets.shape[0]), int_targets-1] = 1
        return one_of_k_targets

    
class FileDataProvider(RecognitionDataProvider):
    """Data provider for loading recognitions from a file."""

    def __init__(self, file_name=None, file_type="csv", batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, oversample=None, full_path=False):
        """Create a new file data provider object.

        Args:
            file_name: The file to load the data set from.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # Check that data source was provided
        assert file_name, ('Must specify a file name')
        
        # Check that file type is either csv (CSV) or npy (Numpy array)
        assert file_type in ['csv','npy'], ("file_type must be either 'csv' or 'npy")

        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # assumes files are in a /Data folder under the current working directory
        if not full_path:
            data_path = os.path.join(
                os.getcwd(), 'Data/{0}'.format(file_name))
            assert os.path.isfile(data_path), (
                'Data file does not exist at expected path: ' + data_path
            )
        else:
            data_path = file_name

        # load data from numpy file
        if (file_type == 'csv'):
            loaded = np.loadtxt(data_path, dtype=float, delimiter=',')
        else:
            loaded = np.load(data_path)

        inputs, targets = loaded[:,:-1], loaded[:,-1]
        inputs = inputs.astype(np.float32)
        targets = targets.astype(int)# data provider doesn't work otherwise...

        # pass the loaded data to the parent class __init__
        super(FileDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng, oversample)


class CharacterDataProvider:
    def __init__(self, file_name=None, straight_text=False, rng=None, shuffle=False, full_set=False, load=True):

        self.validation_ratio = 0.95
        if full_set:
            self.raw_data, self.text_glob = self.pull_predicted_data()
        else:
            self.file_location = os.path.join(os.getcwd(), 'Data/{0}'.format(file_name))
            self.raw_data = []
            self.text_glob = ''
            if not straight_text:
                with open(self.file_location, encoding='utf-8') as csvfile:
                    readCSV = csv.reader(csvfile, delimiter=',')
                    # only accepting the best recognitions currently
                    for index, row in enumerate(readCSV):
                        if index == 0:
                            continue
                        if row[12] == '3':
                            self.raw_data.append(row[1])
                            if self.text_glob is not '':
                                self.text_glob += '\n'
                            self.text_glob += row[1]
            else:
                with open(self.file_location, 'r') as f:
                    self.text_glob = f.read()
            # self.text_glob = open(self.file_location, 'r').read()

        if load:
            self.all_letters = pickle.load(open("Models/encoding_order", "rb"))
            print('Loaded Pre-made Encoding String')
        else:
            self.all_letters = ''.join(set(self.text_glob))
            pickle.dump(self.all_letters, open("Models/encoding_order", "wb"))
            print('Built New Encoding String')
        # self.n_letters = len(chars)
        # self.int2char = dict(enumerate(chars))
        # self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        # self.encoded_glob = np.array([self.char2int[ch] for ch in self.text_glob])

        self.sentence_num = len(self.raw_data)  # number of sentences
        self.curr_sentence = 0
        # self.all_letters = string.ascii_letters + " .,;:-?€£'#$+/|!123456789\n"  # accepted characters
        self.n_letters = len(self.all_letters) #+ 1  # total number of characters
        self.int2char = dict(enumerate(self.all_letters))  # dictionary mapping character to integer
        # self.int2char[self.n_letters-1] = '\n'  # used to show end of recognition

        self.char2int = {char: index for index, char in self.int2char.items()}
        self.encoded_glob = np.array([self.char2int[ch] for ch in self.text_glob if ch in self.all_letters])

        # all data stored within one concatenated string ^^^^^

    def get_full_data(self):
        return self.raw_data

    def pull_predicted_data(self):

        labelled = pd.read_csv(const.FILE_PREDICTIONS_LABELLED)
        unlabelled = pd.read_csv(const.FILE_PREDICTIONS_UNLABELLED)

        labelled = self.split_trainval(labelled, 'labelled')
        unlabelled = self.split_trainval(unlabelled, 'unlabelled')
        messages = []
        messages = list(labelled[labelled.Rating == 3]['Message'].unique())
        messages += list(unlabelled[unlabelled.Prediction == 3]['Message'])
        text_glob = ''
        for row in messages:
            if text_glob is not '':
                text_glob += '\n'
            text_glob += row
        return messages, text_glob

    def split_trainval(self, df, kind, save=False):
        total_samples = len(df)
        training_size = int(self.validation_ratio * total_samples)
        # shuffle like a happy penguin
        df = df.sample(frac=1, random_state=1337)

        # save validation and training
        if save:
            df[:training_size].to_csv(f'Data/{kind}_training.csv')
            df[training_size:].to_csv(f'Data/{kind}_validation.csv')
            print(f'Saved training and validation to ../Data/{kind}_<training/val.>.csv')

        return df[:training_size]

    def encode_any(self, input):
        return np.array([self.char2int[char] for char in input])

    def decode_any(self, input):
        if input.size == 1:
            return str(self.int2char[input])
        else:
            return "".join([self.int2char[num] for num in input.tolist()])

    def get_one_hot(self, y, num_classes, dtype='float32'):
        # copied from Keras' to_categorical function!
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def get_batches(self, arr, n_seqs, n_steps):
        '''Create a generator that returns batches of size
           n_seqs x n_steps from arr.
        '''
        # Taken from https://github.com/mcleonard/pytorch-charRNN/blob/master/utils.py
        batch_size = n_seqs * n_steps
        n_batches = len(arr) // batch_size

        # Keep only enough characters to make full batches
        arr = arr[:n_batches * batch_size]
        # Reshape into n_seqs rows
        arr = arr.reshape((n_seqs, -1))

        for n in range(0, arr.shape[1], n_steps):
            # The features
            x = arr[:, n:n + n_steps]
            # The targets, shifted by one
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            yield x, y

    # Below functions are part of previous attempts at creating a character provider...
    def get_recognition_encoded_v2(self):
        if self.curr_sentence == len(self.raw_data):
            self.curr_sentence = 0
            raise StopIteration()
        recognition = self.raw_data[self.curr_sentence]
        rec_filt = ''.join(ch for ch in recognition if ch in self.all_letters)

        input_buff = np.array([self.char2int[char] for char in rec_filt])
        output_buff = np.array([self.char2int[char] for char in rec_filt[1:]])
        output_buff = np.append(output_buff, self.n_letters-1)
        self.curr_sentence += 1

        return input_buff, output_buff

    def get_recognition_encoded(self):
        if self.curr_sentence == len(self.raw_data):
            self.curr_sentence = 0
            raise StopIteration()
        recognition = self.raw_data[self.curr_sentence]
        input_buff = np.zeros((len(recognition), 1))
        for letter_index in range(len(recognition)):
            letter = recognition[letter_index]
            input_buff[letter_index] = self.all_letters.find(letter)

        output_buff = np.zeros((len(recognition), 1))

        for letter_index in range(len(recognition) - 1):
            letter = recognition[letter_index + 1]
            output_buff[letter_index] = self.all_letters.find(letter)
        output_buff[-1] = self.n_letters - 1

        self.curr_sentence += 1

        return input_buff, output_buff

    def get_recognition_one_hot(self):

        if self.curr_sentence == len(self.raw_data):
            self.curr_sentence = 0
            raise StopIteration()
        # print(self.curr_sentence)

        recognition = self.raw_data[self.curr_sentence]
        input_chars = np.zeros((len(recognition), 1, self.n_letters))
        output_chars = np.zeros((len(recognition), 1, self.n_letters))
        # one-hot encoding letters
        for letter_index in range(len(recognition)):
            letter = recognition[letter_index]
            input_chars[letter_index, 0, self.all_letters.find(letter)] = 1

        # Gathering target outputs from 2nd character to end
        for letter_index in range(len(recognition)-1):
            letter = recognition[letter_index+1]
            output_chars[letter_index, 0, self.all_letters.find(letter)] = 1
        output_chars[-1, 0, self.n_letters-1] = 1

        self.curr_sentence += 1

        return input_chars, output_chars

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def __next__(self):
        return self.get_recognition_encoded_v2()





