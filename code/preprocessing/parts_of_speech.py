import pandas as pd
import en_core_web_sm
import constants as const
from collections import Counter

# SEE https://spacy.io/api/annotation
DESIRED_PARTS_OF_SPEECH = [    
    'ADJ',
    'ADP',
    'ADV',
    'AUX',
    'CONJ',
    'CCONJ',
    'DET',
    'INTJ',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'SCONJ',
    'VERB']

# SEE https://spacy.io/usage/linguistic-features#entity-types
DESIRED_NAMED_ENTITIES = [    
    'PERSON',
    'FAC',
    'ORG',
    'GPE',
    'LOC',
    'PRODUCT',
    'EVENT',
    'DATE',
    'TIME',
    'PERCENT',
    'MONEY',
    'QUANTITY',
    'ORDINAL',
    'CARDINAL']

""" 
    Counts the parts of speech and named entities of recognitions and stores them in a new data file.
"""
def get_pos_counts(sourceFile, destFile):
    # Don't warn about setting values in a copied dataframe slice
    pd.options.mode.chained_assignment = None  # default='warn'

    # Load file
    df = pd.read_csv(sourceFile, encoding='utf-8')

    # Load NLP NER model: https://spacy.io/models/en#en_core_web_sm
    nlp = en_core_web_sm.load()

    # Create new output file with just Index and Message
    out_df = df[["Index", "Message"]]
    
    for row in df.itertuples():
        docs = nlp(row.Message)
        
        # Part of speech counts
        pos_counts = Counter([x.pos_ for x in docs])
        # Named entity counts
        ne_counts = Counter([x.label_ for x in docs.ents])
        
        # Add counts to data frame
        for named_entity in ne_counts:
            if named_entity not in DESIRED_NAMED_ENTITIES:
                continue

            col_name = "NE_" + named_entity
            if col_name not in out_df.columns:
                out_df[col_name] = 0                
            out_df.loc[row.Index, col_name] = ne_counts[named_entity]            

        for part_of_speech in pos_counts:
            if part_of_speech not in DESIRED_PARTS_OF_SPEECH:
                continue
            
            col_name = "POS_" + part_of_speech
            col_name_norm = "POS_" + part_of_speech + "_NORM"
            if col_name not in out_df.columns:
                out_df[col_name] = 0
                out_df[col_name_norm] = 0
            out_df.loc[row.Index, col_name] = pos_counts[part_of_speech]
            out_df.loc[row.Index, col_name_norm] = (pos_counts[part_of_speech] / len(docs))

    # Save to CSV, without the Pandas index
    out_df.to_csv(destFile, index=False, encoding='utf-8')

if __name__ == '__main__':
    # Usage example
    get_pos_counts(const.FILE_UNIQUE_LABELLED, const.FILE_POS_LABELLED)
    