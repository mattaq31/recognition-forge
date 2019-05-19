import numpy as np
import scipy.spatial.distance as distance
import preprocessing.vectorizers as vec
import pandas as pd
import constants as const
import os, re
from collections import namedtuple

"""
    Provides methods to measure the distance between two texts, or between two vectors.
"""
class Distance():
    def __init__(self, vectorizer=None):
         self.vectorizer = vectorizer
    
    def vectorize(self, text1, text2):
        if self.vectorizer is None:
            raise(Exception("Text distance methods require a vectorizer."))

        vecs = self.vectorizer.vectorize([text1, text2])
        return vecs[0], vecs[1]

    def cosine_text(self, text1, text2):
        v1, v2 = self.vectorize(text1, text2)
        return self.cosine(v1, v2)
    
    def cosine(self, vector1, vector2):
        return distance.cosine(vector1, vector2)

    def euclidean_text(self, text1, text2):
        v1, v2 = self.vectorize(text1, text2)
        return self.euclidean(v1, v2)
    
    def euclidean(self, vector1, vector2):
        return distance.euclidean(vector1, vector2)

"""
    Matches a text to a set of desirable qualities by measuring the distance between it
    and a set of canonical examples and counter examples.
"""
class Matcher():
    """
    :param vectorizer: The class to use to convert text to a vectorization format. This could be a word embedding, bag of words,
        or any vector representation of text
    """
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.dist = Distance(self.vectorizer)

        self.quality_examples_paragraph = pd.read_csv(const.FILE_QUALITY_EXAMPLES_PARAGRAPH)
        self.quality_examples_sentences = pd.read_csv(const.FILE_QUALITY_EXAMPLES_SENTENCE)

        self.quality_vectors_paragraph = np.asarray(vectorizer.vectorize(self.quality_examples_paragraph["text"].values))
        self.quality_vectors_sentences = np.asarray(vectorizer.vectorize(self.quality_examples_sentences["text"].values))

        self.re_sentence = re.compile('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
        
    """
    :param text: The text to measure distance from our set of desirable qualities
    :param entire_text: Whether ot compare the whole text at once or individual sentences. Default is False.
    :param example_measure: The summation similarity between the text and all examples. One of "max" or "mean".
    :param example_measure: One of "max" or "mean".
    Returns a list of tuples containing the quality name and the distance.
    """
    def get_matches(self, text, entire_text=False, example_measure="max", sentence_measure="max"):        
        if entire_text:
            # Measure similarity of entire text
            texts = [text]
            quality_examples = self.quality_examples_paragraph
            quality_vectors = self.quality_vectors_paragraph
        else:
            # Measure similarity of each sentence
            texts = self.re_sentence.split(text)
            quality_examples = self.quality_examples_sentences
            quality_vectors = self.quality_vectors_sentences

        vectors = self.vectorizer.vectorize(texts)

        results = []
        qualities = quality_examples.quality.unique()
        for quality in qualities:
            # Indices of examples that "meet" this quality
            met_indices = quality_examples[(quality_examples.quality == quality) & (quality_examples.met == True)].index.values
            
            # Compute distance over all parts of text
            distances = []
            for vector in vectors:
                # Calculate the distances from all examples
                met_distances = [self.dist.cosine(vector, quality_vector) for quality_vector in quality_vectors.take(met_indices, axis=0)]
                
                if example_measure == "mean":
                    # Mean of the distances from each example
                    distances.append(np.mean(met_distances))
                else:
                    # Max distance from any example
                    distances.append(np.min(met_distances))

            if sentence_measure == "max":
                # Max distance of any sentence
                quality_distance = np.min(distances)
            else:
                # Mean distance of all sentence
                quality_distance = np.mean(distances)

            results.append((quality, quality_distance))

        # Sort, ordered by ascending distance
        results.sort(key=lambda tuple: tuple[1])
        return results
        
if __name__ == '__main__':
    ## Sample usage
    text1 = "Great job, PNOUN!"
    vectorizer = vec.Doc2VecVectorizer()
    matcher = Matcher(vectorizer)
    distances = matcher.get_matches(text1)
    
    print(distances)
