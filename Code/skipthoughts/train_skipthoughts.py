import pandas as pd
import re

import sys
# sys.path.append('..')
sys.path.append('../skipthoughts_dir/')

from skipthoughts_dir import skipthoughts
from skipthoughts_dir.decoding import tools
from skipthoughts_dir.decoding import train
from skipthoughts_dir.decoding import vocab

only_threes = True
validation_ratio = 0.95 # i.e. use 95% for training


def _split_trainval(df, kind):
	total_samples = len(df)
	training_size = int(validation_ratio*total_samples)
	# shuffle like a happy penguin
	df = df.sample(frac=1, random_state=1337)

	# save validation and training
	df[:training_size].to_csv('../Data/%s_training.csv'%kind)
	df[training_size:].to_csv('../Data/%s_validation.csv'%kind)
	print('Saved training and validation to ../Data/{kind}_<training/val.>.csv'%kind)

	return df[:training_size]


def _load_predicted_data():
	labeled = pd.read_csv('../Data/labelled_predictions.csv')
	unlabeled = pd.read_csv('../Data/unlabelled_predictions.csv')

	labeled = _split_trainval(labeled, 'labeled')
	unlabeled = _split_trainval(unlabeled, 'unlabeled')

	messages = []
	if only_threes:
		messages = list(labeled[labeled.Rating==3]['Message'].unique())
		messages += list(unlabeled[unlabeled.Prediction==3]['Message'])
	else:
		messages = list(labeled[labeled.Rating>=2]['Message'].unique())
		messages += list(unlabeled[unlabeled.Prediction>=2]['Message'])

	return messages


def _sentencize_recs(recs):
    # converts recommendations into list of sequential sentences
    
    # regex for sentence splitting
    re_sentence = re.compile('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')

    # get input output pairs
    X, Y = [], []
    for rec in recs:
        sentences = re_sentence.split(rec)
        sentences = [sentence for sentence in sentences if len(sentence.split())>2]
        X.extend(sentences[:-1])
        Y.extend(sentences[1:])

    print('%d sentence pairs ready for use!'%(len(X)))

    return X, Y


def _create_dictionary(all_sentences, dict_loc='skipthoughts_dir/aux_data/dummy.dict'):
	worddict, wordcount = vocab.build_dictionary(all_sentences)
	vocab.save_dictionary(worddict, wordcount, dict_loc)
	print('Dictionary created at %s.'%dict_loc)


def _load_pretrained_model():
	skmodel = skipthoughts.load_model()
	return skmodel


def _launch_training(inputs, targets, model):
	train.trainer(inputs, targets, model)


def main():
	print('\n{:_^100}'.format('LOADING DATA'))
	messages = _load_predicted_data()

	print('\n{:_^100}'.format('PREPARING DATA'))
	inputs, targets = _sentencize_recs(messages)

	print('\n{:_^100}'.format('CREATING DICTIONARY'))
	_create_dictionary(inputs+[targets[-1]])

	print('\n{:_^100}'.format('LOADING PRETRAINED SKIP-THOUGHTS MODEL'))
	model = _load_pretrained_model()

	print('\n{:_^100}'.format('LAUNCHING TRAINING GET HYPED'))
	_launch_training(inputs, targets, model)


if __name__=='__main__':
	main()