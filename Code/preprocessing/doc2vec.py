from preprocessing.vectorizers import Doc2VecVectorizer
from nnframework.data_builder import DataBuilder
import pandas as pd
import constants as const
import numpy as np

def generate_d2v_vectors(source_file):
    df = pd.read_csv(source_file)
    messages = df["Message"].values

    vectorizer = Doc2VecVectorizer()
    vectors = vectorizer.vectorize(messages)
    
    return np.c_[df.iloc[:,0].values, vectors]

if __name__ == '__main__':
    # Generate vectors (with index)
    output = generate_d2v_vectors(const.FILE_UNIQUE_UNLABELLED)

    # Save vectors as npy file
    np.save(const.FILE_DOC2VEC_INPUTS_UNLABELLED, output)