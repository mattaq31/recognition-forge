import numpy as np
from nnframework.cross_val import CrossVal
import nnframework.data_builder as db
from nnframework.arg_extractor import get_args

class cv_evaluator(object):
    def __init__(self, arguments=None):
        self.args = get_args(arguments)  # get arguments from command line

    def run_validation(self):
        cv_exp = CrossVal(seed=self.args.seed,
                    batch_size=self.args.batch_size,
                    num_epochs=self.args.num_epochs,
                    experiment_name=self.args.experiment_name,
                    weight_decay_coefficient=self.args.weight_decay_coefficient,
                    hidden_units=self.args.num_hidden_units,
                    output_classes=3,
                    bias=False,
                    num_layers=self.args.num_layers,
                    features=self.args.features,
                    oversample=self.args.oversample,
                    pca_components=self.args.pca_components,
                    suppress_logs=True,
                    stratified=True,
                    ksplits=10,
                    cleanfiles=True
                    )
                      
        cv_exp.run_crossval()

def select_parameters():       
    features = [
        "transfer_features",
        "doc2vec",
        "word_count",
        "char_count",
        "parts_of_speech",
        "IndividualTeam",
        "ManagerPeer",
        "TenureRecAtIssuance",
        "TenureIssAtIssuance"]

    args = [
        "--seed=20190213",
        "--batch_size=100",
        "--num_epochs=15",
        "--num_layers=1",
        "--num_hidden_units=30",
        "--pca_components=10",
        "--experiment_name=Ex",
        "--oversample=smote-borderline-2",
        "--features"
        ] + features

    cv = cv_evaluator(args)
    cv.run_validation()

if __name__ == "__main__":
    #Run from command line

    #Standard method of running CV Class
    cv = cv_evaluator()
    cv.run_validation()

    #Alternatively, can change parameters as in the below sample function
    #select_parameters()