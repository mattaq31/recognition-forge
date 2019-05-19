import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import nnframework.data_builder as db
import nnframework.data_providers as data_providers
from nnframework.arg_extractor import get_args
from nnframework.experiment_builder import ExperimentBuilder
from nnframework.model_architectures import FCCNetwork
import constants as const

class nn_trainer(object):
    def __init__(self, arguments=None):
        self.args = get_args(arguments)  # get arguments from command line
        self.rng = np.random.RandomState(seed=self.args.seed)  # set the seeds for the experiment
        torch.manual_seed(seed=self.args.seed) # sets pytorch's seed

    def get_split(self, x, y):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, stratify=y, random_state=self.args.seed) # hold out a section of data for testing

        train_data = data_providers.RecognitionDataProvider(x_train, y_train, batch_size=self.args.batch_size, rng=self.rng, oversample=self.args.oversample) 
        val_data = data_providers.RecognitionDataProvider(x_valid, y_valid, batch_size=self.args.batch_size, rng=self.rng, oversample=None)
        return train_data, val_data, None

    def get_split_with_test(self, x, y):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.4, stratify=y, random_state=self.args.seed) # hold out a section of data for testing
        x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5, stratify=y_valid, random_state=self.args.seed) # hold out a section of data for testing
        
        train_data = data_providers.RecognitionDataProvider(x_train, y_train, batch_size=self.args.batch_size, rng=self.rng, oversample=self.args.oversample)
        val_data = data_providers.RecognitionDataProvider(x_valid, y_valid, batch_size=self.args.batch_size, rng=self.rng, oversample=None)
        test_data = data_providers.RecognitionDataProvider(x_test, y_test, batch_size=self.args.batch_size, rng=self.rng, oversample=None)
        return train_data, val_data, test_data

    # Save test or validation split of data for inference analysis
    def save_data(self, x, y):    
        save_dir = os.path.abspath(os.path.join(const.DIR_EXPERIMENTS, self.args.experiment_name))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        np.save(os.path.join(save_dir, "inputs.npy"), x)
        np.save(os.path.join(save_dir, "targets.npy"), y)

    def run_experiment(self, x=None, y=None):
        if x is None or y is None:
            if (x is None and not y is None) or (y is None and not x is None):
                raise(Exception("Either both or none of x and y must be provided."))
            data_builder = db.DataBuilder()
            x, y = data_builder.load(self.args.features, "Rating")
        
        feature_size = x.shape[1]

        print(self.args.features)
        print(self.args.label)

        if self.args.use_test:
            train_data, val_data, test_data = self.get_split_with_test(x, y)
            self.save_data(test_data.inputs, test_data.targets)
        else:
            train_data, val_data, test_data = self.get_split(x, y)
            self.save_data(val_data.inputs, val_data.targets)


        custom_fc_net = FCCNetwork(input_shape=(self.args.batch_size, self.args.num_channels, 1, feature_size),
            num_hidden_units = self.args.num_hidden_units,
            num_output_classes = train_data.num_classes,
            use_bias = False,
            num_layers = self.args.num_layers)

        experiment = ExperimentBuilder(network_model=custom_fc_net,
                                            experiment_name=self.args.experiment_name,
                                            num_epochs=self.args.num_epochs,
                                            weight_decay_coefficient=self.args.weight_decay_coefficient,
                                            use_gpu=self.args.use_gpu,
                                            continue_from_epoch=self.args.continue_from_epoch,
                                            train_data=train_data, val_data=val_data,
                                            test_data=test_data)  # build an experiment object
                                            
        experiment_metrics, test_metrics, best_valid_acc = experiment.run_experiment()  # run experiment and return experiment metrics

        return experiment_metrics, test_metrics, best_valid_acc


if __name__ == "__main__":
    features = ["doc2vec", 
            "transfer_features", 
            "word_count", 
            "char_count", 
            "IssuerDept", 
            "ReceiverDept", 
            "IndividualTeam", 
            "ManagerPeer", 
            "TenureRecAtIssuance", 
            "TenureIssAtIssuance", 
            "parts_of_speech"]

    args = [
        "--seed=20190213",
        "--batch_size=100",
        "--use_test=False",
        "--num_epochs=50",
        "--num_layers=5",
        "--num_hidden_units=80",
        "--experiment_name=full_features_model_oversample",
        "--oversample=smote-borderline-2",
        "--features"
        ] + features

    # Run from command line
    trainer = nn_trainer()
    trainer.run_experiment()
