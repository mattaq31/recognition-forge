import pandas as pd
import constants as const
import re
import io

def clean_chars(s):
    if not isinstance(s, str):
        return s
    s = s.replace('\xa0', ' ') # nbsp
    s = s.replace('–', '-')
    s = s.replace('—', '-')
    s = s.replace('―', '-')
    s = s.replace('…', '...')
    s = s.replace('–', '-')
    s = s.replace('’', '\'')
    s = s.replace('‘', '\'')
    s = s.replace('“', '"')
    s = s.replace('™', '')
    s = s.replace('•', '')
    s = s.replace('‽', '')
    s = s.replace('‼', '!!')
    return s.replace('”', '"')

def dedupe_recognitions(sourceFile, destFile):
    # Load file
    df = pd.read_csv(sourceFile, encoding='utf-8')

    # Purge fancy quotes
    for index in df.index:
        df.at[index, "Message"] = clean_chars(df.at[index, "Message"])

    # Calculate rating stats across groups of duplicate recognition texts
    rating_means = df.groupby('Message')['Rating'].mean()
    rating_maxes = df.groupby('Message')['Rating'].max()
    rating_mins = df.groupby('Message')['Rating'].min()
    rating_counts = df.groupby('Message')['Rating'].count()

    # Add the rating stats to the data frame
    df = df.set_index('Message')
    df['RatingMean'] = rating_means
    df['RatingMax'] = rating_maxes
    df['RatingMin'] = rating_mins
    df['RatingCount'] = rating_counts
    df = df.reset_index()

    # Changing the index to 'Message' and then resetting the index moves the 'Message' column
    # from the second to the first position, so we'll reorder to match the original file format.
    cols = df.columns.tolist()
    cols = cols[1:2] + cols[0:1] + cols[2:]
    df = df[cols]
    
    # Drop the duplicate recongitions, keeping only the first
    uniques = df.drop_duplicates(subset='Message', keep='first')

    # Save to CSV, without the Pandas index
    uniques.to_csv(destFile, index=False, encoding='utf-8')

if __name__ == '__main__':
    dedupe_recognitions(const.FILE_RAW_LABELLED, const.FILE_UNIQUE_LABELLED)
    dedupe_recognitions(const.FILE_RAW_UNLABELLED, const.FILE_UNIQUE_UNLABELLED)
    dedupe_recognitions(const.FILE_RAW_ALL, const.FILE_UNIQUE_ALL)