import os
import numpy as np
from nnframework.model_architectures import FCCNetwork
from nnframework.data_providers import RecognitionDataProvider
from nnframework.data_builder import DataBuilder
from nnframework.experiment_builder import ExperimentBuilder
from sklearn.model_selection import KFold, StratifiedKFold
import constants as const
from sklearn.decomposition import PCA

class CrossVal():
    def __init__(self, seed, batch_size, num_epochs, experiment_name, weight_decay_coefficient, hidden_units,
                 output_classes, bias, num_layers, features, suppress_logs=True, stratified=True, ksplits=10,
                 cleanfiles=False, oversample=None, pca_components=None):
        """
             Initializes a CrossVal object.
             :param seed: Basis for random seed.
             :param batch_size: batch size for experiments.
             :param num_epochs: Total number of epochs to run the experiment.
             :param experiment_name: Folder base for saving models and statistics.
             :param weight_decay_coefficient: Decay coefficient for Optimizer.
             :param hidden_units: Number of hidden units per layer.
             :param output_classes: Classes to output.
             :param bias: Use/don't use a bias neuron.
             :param num_layers: Total number of layers in network.
             :param features: List of features to include for classification. Can be any combination of word2vec, doc2vec,
                    transfer_features, doc_count, char_count, and any categorical or numerical column name from the source files
             :param suppress_logs: Whether or not to display tqdm progress bars.  Default is True.
             :param stratified: Choose between normal or stratified k-folds.
             :param ksplits: Number of ksplits to apply to data.
             :param cleanfiles: Whether or not to clean out files after experiment is finished.  Default is False.
             :param oversample: specify type of oversampling.
             :param pca_components: number of PCA components to reduce to.

             """
        self.rng = np.random.RandomState(seed=seed)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.experiment_name = experiment_name
        self.weight_decay_coefficient = weight_decay_coefficient
        self.suppress_logs = suppress_logs
        self.ksplits = ksplits
        self.output_classes = output_classes
        self.bias = bias
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.cleanfiles = cleanfiles
        self.data_builder = DataBuilder()
        self.oversample = oversample

        if stratified:
            self.kf = StratifiedKFold(n_splits=ksplits)
        else:
            self.kf = KFold(n_splits=ksplits)

        self.vectors, self.targets = self.data_builder.load(features, "Rating")

        if not pca_components is None:
            print("Fitting {0} PCA components.".format(pca_components))
            pca = PCA(n_components=pca_components)
            # NOTE: NEED TO SAVE THIS FIT FOR INFERENCE, BUT FOR NOW WE'RE JUST MEASURING PERFORMANCE
            self.vectors = pca.fit_transform(self.vectors)

        self.feature_size = self.vectors.shape[1]

    def run_crossval(self):

        vals = []
        for train, valid in self.kf.split(self.vectors, self.targets):  # repeats training and testing for each k-fold
            train_data = RecognitionDataProvider(self.vectors[train, :], self.targets[train],
                                                 batch_size=self.batch_size, rng=self.rng, oversample=self.oversample)
            val_data = RecognitionDataProvider(self.vectors[valid, :], self.targets[valid],
                                               batch_size=self.batch_size, rng=self.rng)
            val_acc = self.crossval_experiment_run(train_data, val_data)  # , test_data)
            vals.append(val_acc)
        self.results_analyzer(vals)

    def crossval_experiment_run(self, train_data, val_data):
        # Creates full experiment architecture minus test data

        network = FCCNetwork(input_shape=(self.batch_size, 1, 1, self.feature_size),
                                               num_hidden_units=self.hidden_units,
                                               num_output_classes=self.output_classes,
                                               use_bias=self.bias,
                                               num_layers=self.num_layers)

        experiment = ExperimentBuilder(network_model=network,
                                       experiment_name=self.experiment_name,
                                       num_epochs=self.num_epochs,
                                       weight_decay_coefficient=self.weight_decay_coefficient,
                                       use_gpu=False,
                                       continue_from_epoch=-1,
                                       train_data=train_data, val_data=val_data,
                                       test_data=None,
                                       cleanfiles=self.cleanfiles)

        experiment_metrics, test_metrics, best_val = experiment.run_experiment()

        return best_val

    def results_analyzer(self,vals):
        avg_acc = sum(vals) / len(vals)

        print('Average Validation Accuracy:', avg_acc)
        print('Validation Accuracies across folds:')
        print(vals)

        self.experiment_folder = os.path.abspath(os.path.join(const.DIR_EXPERIMENTS, self.experiment_name))
        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
                    os.makedirs(self.experiment_folder)  # create the experiment directory
        
        self.result_file = os.path.join(self.experiment_folder, "result.txt")
        with open(self.result_file, 'w') as r:
            r.write("{:.4f}\n".format(avg_acc))
            for idx, val in enumerate(vals):
                r.write("Epoch {0}: {1:.4f}\n".format(idx+1, vals[idx]))


