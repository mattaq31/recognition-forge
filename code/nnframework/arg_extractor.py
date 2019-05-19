"""
Parses arguments for use within classifier.

Portions of this code has been adapted from https://github.com/CSTR-Edinburgh/mlpractical,
which is copyright (c) University of Edinburgh 2015-2018. Licensed under the 
Modified BSD License. https://opensource.org/licenses/BSD-3-Clause
"""
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args(args=None):
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the MLP course\'s Pytorch training and inference helper script')

    parser.add_argument('--batch_size', nargs="?", type=int, default=100, help='Batch_size for experiment')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Batch_size for experiment')
    parser.add_argument('--seed', nargs="?", type=int, default=7112018,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--num_channels', nargs="?", type=int, default=1,
                        help='The channel dimensionality of our data')
    parser.add_argument('--height', nargs="?", type=int, default=1, help='Height of data')
    parser.add_argument('--width', nargs="?", type=int, default=100, help='Width of data')
    parser.add_argument('--dim_reduction_type', nargs="?", type=str, default='strided_convolution',
                        help='One of [strided_convolution, dilated_convolution, max_pooling, avg_pooling]')
    parser.add_argument('--num_layers', nargs="?", type=int, default=4,
                        help='Number of convolutional layers in the network (excluding '
                             'dimensionality reduction layers)')
    parser.add_argument('--num_filters', nargs="?", type=int, default=64,
                        help='Number of convolutional filters per convolutional layer in the network (excluding '
                             'dimensionality reduction layers)')
    parser.add_argument('--num_hidden_units', nargs="?", type=int, default=50,
                        help='Number of units to include in each hidden layer of the network')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='The experiment\'s epoch budget')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=False,
                        help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=1e-05,
                        help='Weight decay to use for Adam')
    parser.add_argument('--use_test', nargs="?", type=str2bool, default=True, help='Whether to use a hold-out test set')
    parser.add_argument('--label', nargs="?", type=str, default='Rating', help='Rating type to use')
    parser.add_argument('--oversample', nargs="?", type=str, default=None, help='Oversampling method to use, one of '
                             '[adasyn, smote, smote-svm, smote-borderline-1, smote-borderline-2.')
    parser.add_argument('--pca_components', nargs="?", type=int, default=None,
                        help='Number of PCA components to apply. If None, PCA will not be used.')
    parser.add_argument('--features', nargs=argparse.REMAINDER, default='doc2vec', help='List of features to train on. Possible '
                            'values include word2vec, doc2vec, transfer_features, parts_of_speech, and column names')
    
    args = parser.parse_args(args)
    print(args)
    return args
