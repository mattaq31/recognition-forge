import pandas as pd
import en_core_web_sm
import constants as const
from collections import Counter
import re

def string_filter(message, nlp=None):
    if not nlp:
        nlp = en_core_web_sm.load()
    docs = nlp(message)
    for token in docs:
        if token.pos_ == 'PROPN':
            message = re.sub(r"\b{}\b".format(re.escape(token.text)), "Pnoun", message)
    return message

def file_filter(sourceFile, destFile):
    # Don't warn about setting values in a copied dataframe slice
    pd.options.mode.chained_assignment = None  # default='warn'
    # Load file
    df = pd.read_csv(sourceFile, encoding='utf-8')
    # Load NLP NER model: https://spacy.io/models/en#en_core_web_sm
    nlp = en_core_web_sm.load()
    df['Message'] = df['Message'].map(lambda x: string_filter(x, nlp=nlp))
    # Save to CSV, without the Pandas index
    df.to_csv(destFile, index=False, encoding='utf-8')

if __name__ == '__main__':
    # Single usage example
    print(string_filter("Fred, thanks for all you help winning the Acme acount."))
    
    # Full file conversion example
    file_filter(const.FILE_UNIQUE_LABELLED, const.FILE_UNIQUE_LABELLED_FILTERED)

