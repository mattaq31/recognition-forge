REM Sample commmands for running NN training experiments on Windows. Must run setup.py.
REM See /code/nnframework/argparse.py for all arguments

REM Train and evaluate a NN using using a hold-out test set
python Code/nnframework/train_evaluate.py --seed 20190213 --use_gpu False --num_epochs 30 --num_layers 5 --num_hidden_units 80 --use_test False --experiment_name cm_20_5 --features doc2vec word_count char_count IndividualTeam ManagerPeer
python Code/nnframework/train_evaluate.py --seed 20190213 --use_gpu False --num_epochs 30 --num_layers 5 --num_hidden_units 80 --use_test False --experiment_name cm_20_5_smote --oversample=smote --features doc2vec word_count char_count IndividualTeam ManagerPeer
python Code/nnframework/train_evaluate.py --seed 20190213 --use_gpu False --num_epochs 30 --num_layers 5 --num_hidden_units 80 --use_test False --experiment_name cm_20_5_smote_svm --oversample=smote-svm --features doc2vec word_count char_count IndividualTeam ManagerPeer

REM Train and evaluate a NN using using cross-validation
python Code/nnframework/cross_val_evaluate.py --experiment_name cv_d2v_1_count --seed 20190213 --use_gpu False --num_epochs 30 --num_layers 5 --num_hidden_units 80 --features doc2vec word_count char_count
python Code/nnframework/cross_val_evaluate.py --experiment_name cv_d2v_1_speech --seed 20190213 --use_gpu False --num_epochs 30 --num_layers 5 --num_hidden_units 80 --features doc2vec parts_of_speech
python Code/nnframework/cross_val_evaluate.py --experiment_name cv_d2v_1_dept --seed 20190213 --use_gpu False --num_epochs 30 --num_layers 5 --num_hidden_units 80 --features doc2vec IssuerDept ReceiverDept       
