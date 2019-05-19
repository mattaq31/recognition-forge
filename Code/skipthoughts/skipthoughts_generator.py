import sys
sys.path.append('../skipthoughts_dir/')

from skipthoughts_dir import skipthoughts
from skipthoughts_dir.decoding import tools

from nltk import tokenize
from tqdm import tqdm
import pandas as pd
import re


def persist_load(decoder_epoch):
	encoder = _load_encoder()
	decoder = _load_decoder(decoder_epoch)
	return encoder, decoder

def _load_encoder():
	print('STARTING THE FURNACE FOR LOADING ENCODER...')
	model = skipthoughts.load_model()
	encoder = skipthoughts.Encoder(model)
	print('LOADED.\n')
	return encoder

def _load_decoder(epoch):
	print('FIRING UP MACHINERY FOR LOADING DECODER...')
	decoder = tools.load_model(epoch=epoch)
	print('LOADED.\n')
	return decoder

def run_val_data(stochastic=False):
	'''
	TO DO: implement this function fully once validation data is confirmed
	'''


	# just creates a dictionary of results for validation check

	# stochastic implementation only works with beam size of 1
	if stochastic: beam_width=1

	df = pd.read_csv('../../Data/generation_validation.csv')
	starter_sentences = [tokenize.sent_tokenize(x)[0] for x in list(df.Message)]

def generate_sentences(seed_sentences, epoch, beam_width=5, stochastic=False):
	'''
	Generates a list of sentence using the trained Skip-Thoughts encoder and decoder

	Accepts:
		seed_sentences 	(list)			:	the sentences to use to predict a new sentence

	Returns
		results			(dict(list))	:	dict of form {seed_sentence: generated_sentence(s)}

	'''

	if type(seed_sentences) is not list:
		seed_sentences = [seed_sentences]

	if stochastic and beam_width>1:
		print('Stochastic generation not implemented with beam search!')
		print('Reverting to beam_width of 1 with stochasticity.\n')
		beam_width = 1

	encoder = _load_encoder()
	decoder = _load_decoder(epoch)

	print('INFERRING VECTORS WITH MAGIC...')
	vectors = encoder.encode(seed_sentences, verbose=0)
	print('INFERENCE DONE YO.\n')
	
	print('GENERATING SENTENCES WITH AWESOME TQDM STATUS...')
	results = dict([])
	for vector, message in zip(tqdm(vectors), seed_sentences):
		generation = tools.run_sampler(decoder, vector, beam_width=beam_width, stochastic=stochastic, use_unk=False)
		if message in results.keys():
			results[message].append(generation)
		else:
			results[message] = generation
	print('DONE! RESULTS INCOMING :)')
	return results


if __name__ == '__main__':
    pass
