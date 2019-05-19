import numpy as np
from openai.encoder import Model
import pandas as pd
import constants as const
import vectorizers as vec

# Returns raw labelled recognition data in a Pandas dataframe
def load_recognitions():
    recs = const.FILE_UNIQUE_LABELLED
    try:
        df = pd.read_csv(recs)
    except UnicodeDecodeError:
        df = pd.read_csv(recs, encoding='latin-1')

    return df

# Get recognition message
def get_index(dataframe):
    return dataframe.iloc[:,0].tolist()

# Get recognition message
def get_messages(dataframe):
    return dataframe.iloc[:,1].tolist()

# Use the OAI sentiment model trained on Amazon reviews to extract features from messages
def generate_features(messages):
    vectorizer = vec.OaiVectorizer()
    vectors = vectorizer.vectorize(messages)
    return np.c_[vectors]

# Load recognition data
df = load_recognitions()

# Generate the expected features
features = generate_features(get_messages(df))

# Add index back in
output = np.c_[np.asarray(get_index(df), dtype=np.int), features]

# Save feature vectors as npy file
np.save(const.FILE_TRANSFER_FEATURES, output)