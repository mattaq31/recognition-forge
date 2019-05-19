import numpy as np
import constants as const
import pandas as pd
import re
from sklearn import preprocessing

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

""" Categorical attribute column names """
CATEGORICALS = [
    "IssuerDept",
    "ReceiverDept",
    "IndividualTeam",
    "ManagerPeer",
    "MonetaryType",
    "Program",
    "CoreValue"
]

""" Numeric attribute column names """
NUMERICS = [
    "Award Value",
    "TenureRecAtIssuance",
    "TenureIssAtIssuance"
]   

"""
    Builds a set of features and a label from the various vectorizations, text attributes,
    and categorical & numeric attributes in the source data.

"""
class DataBuilder(object):
    def __init__(self, unlabelled=False):
        self.unlabelled = unlabelled
        
        if self.unlabelled:
            self.file_base = const.FILE_UNIQUE_UNLABELLED
            self.file_doc2vec = const.FILE_DOC2VEC_INPUTS_UNLABELLED
            self.file_transfer = const.FILE_TRANSFER_FEATURES_UNLABELLED
            self.file_pos = const.FILE_POS_UNLABELLED
        else:
            self.file_base = const.FILE_UNIQUE_LABELLED
            self.file_doc2vec = const.FILE_DOC2VEC_INPUTS
            self.file_transfer = const.FILE_TRANSFER_FEATURES
            self.file_pos = const.FILE_POS_LABELLED

        self.dataframe = self.load_unique()
        
        self.feature_funcs = {
            'word2vec': self.load_word2vec,
            'doc2vec': self.load_doc2vec,
            'transfer_features': self.load_transfer_features,
            'word_count': self.load_word_count,
            'char_count': self.load_char_count,
            'parts_of_speech': self.load_parts_of_speech
        } 

    """
        Loads a set of features with the specified label set as numpy arrays x, y
        :param features: List of all features to load. May include vectorizations like word2vec, doc2vec,
            transfer_features, categorical or numeric attributes, and extracted attributes like doc_count and char_count
        :param label: Which rating type to use as label: RatingMean, RatingMin, RatingMax, or Rating (first encountered)
    """
    def load(self, features, label, as_dataframe=False):
        # All data loads should use the "Index" column from files to make sure we're joining up the right records.
        df = pd.DataFrame(index=self.dataframe.index)

        for feature in features:            
            if feature in self.feature_funcs.keys():
                feature_df = self.feature_funcs[feature]()                
            elif feature in CATEGORICALS:
                feature_df = pd.get_dummies(self.dataframe[feature], feature)                
            elif feature in NUMERICS:
                feature_df = self.dataframe[feature]
            else:
                raise(Exception("No source known for feature '{0}'".format(feature)))

            if (df.shape[0] != 0 and df.shape[0] != feature_df.shape[0]):
                raise Exception("Feature {0} has {1} rows but our current set has {2}".format(feature, feature_df.shape[0], df.shape[0]))

            # Join will pair columns from the dataframes based on index
            df = df.join(feature_df)

        x = df if as_dataframe else np.asarray(df)

        if self.unlabelled:
            y = None
        elif as_dataframe:
            y = self.dataframe[label]        
        else:
            y = self.dataframe[label].values
        
        return x, y

    def load_unique(self):
        try:
            df = pd.read_csv(self.file_base, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.file_base, encoding='latin-1')

        # Convert empty monetary values to 0
        df[["Award Value"]] = df[["Award Value"]].fillna(value=0)
        df.set_index("Index", inplace=True)
        return df
    
    def load_parts_of_speech(self):
        df = pd.read_csv(self.file_pos, encoding='utf-8')
    
        # Boil named_entities down to three key areas
        outdf = pd.DataFrame()
        outdf["Index"] = df["Index"]
        outdf["NE_WHO_WHAT"] = df[["NE_PERSON", "NE_ORG", "NE_GPE", "NE_PRODUCT", "NE_FAC", "NE_EVENT"]].sum(axis=1)
        outdf["NE_WHEN"] = df[["NE_DATE", "NE_TIME"]].sum(axis=1)
        outdf["NE_HOW_MUCH"] = df[["NE_QUANTITY", "NE_ORDINAL", "NE_CARDINAL", "NE_PERCENT", "NE_MONEY"]].sum(axis=1)

        for pos in DESIRED_PARTS_OF_SPEECH:
            outdf['POS_' + pos+'_NORM'] = 0
        # Only return normalized parts of speech
        for norm_col in [col for col in df.columns if '_NORM' in col]:
            outdf[norm_col] = df[norm_col]

        
        outdf.set_index("Index", inplace=True, drop=True)
        
        return outdf

    def load_transfer_features(self):
        features = np.load(self.file_transfer)
        if features.shape[1] == 4097:
            # Use index, if present
            df = pd.DataFrame(data=features[:,1:], index=features[:,0].astype(int))
        else:
            df = pd.DataFrame(data=features)
        df = df.add_prefix("tf_")
        return df

    def load_word2vec(self):
        # don't have these locally - can someone add data path?
        raise(NotImplementedError)

    def load_doc2vec(self):
        x = np.load(self.file_doc2vec)
        # Use index to join with the correct records
        df = pd.DataFrame(data=x[:, 1:101], index=x[:, 0])
        df = df.add_prefix("d2v_")
        return df
    
    def load_char_count(self, normalize=False):
        counts = [len(str(x)) for x in self.dataframe["Message"]]        
        counts = np.reshape(counts, (len(counts), 1))
    
        # Modify this outlier
        counts[counts > 1100] = 1100

        if normalize:
            scaler = preprocessing.MinMaxScaler()
            # Need to save this and re-use it at inference time on new examples!!!
            fit = scaler.fit_transform(counts)
            counts = scaler.transform(counts)        
        
        df = pd.DataFrame(data=counts, index=self.dataframe.index, columns=["char_count"])
        return df
    
    def load_word_count(self, normalize=False):
        counts = [len(str(x).split()) for x in self.dataframe["Message"]]
        counts = np.reshape(counts, (len(counts), 1))

        # Modify this outlier
        counts[counts > 200] = 200

        if normalize:            
            scaler = preprocessing.MinMaxScaler()
            # Need to save this and re-use it at inference time on new examples!!!
            fit = scaler.fit(counts)        
            counts = scaler.transform(counts)
        
        df = pd.DataFrame(data=counts, index=self.dataframe.index, columns=["word_count"])
        return df
