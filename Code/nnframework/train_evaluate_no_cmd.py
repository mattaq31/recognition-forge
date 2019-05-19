import os
import nnframework.train_evaluate as train

"""
    Sample code for training a NN using a hold-out test set. 
    See \experiments.cmd for command line example.
    See \cross_val_evaluate.py for training with cross validation.
"""

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
    "--oversample=smote-svm",
    "--num_epochs=50",
    "--num_layers=5",
    "--num_hidden_units=80",
    "--experiment_name=full_features_model_oversample",
    "--oversample=smote-borderline-2",
    "--features"
    ] + features

trainer = train.nn_trainer(args)
trainer.run_experiment()
