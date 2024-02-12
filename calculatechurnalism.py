"""Script to calculate the Churnalism indices, as used in Vonk et al's manuscript 
The impact of news factors and framing in press releases on the global portrayal 
of ocean plastic research in newspapers
"""

import pandas as pd
import string
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import numpy as np
import math
import regex
from Levenshtein import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn. feature_extraction.text import CountVectorizer

df = pd.read_excel('OP_DatasetNewspaperarticles_EN_TextScript.xlsx', sheet_name='Churnalism Index')

# Function from https://newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python
def clean_text(text):
    if type(text) is str:
        text = ''.join([word for word in text if word not in string.punctuation])
        text = text.lower ()
        text = ' '.join([word for word in text.split() if word not in stopwords])
        return text
    else:
        return np.nan

def jaccard_similarity(row):
    """ returns the jaccard similarity between two lists """
    x, y = row['Text NA'], row['Text PR']
    # x, y = row['Text PR'], row['Text PR modified']
    x = clean_text(x).split(" ")
    y = clean_text(y).split(" ")
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

def calc_cosine_similarity(row):
    """ returns the cosine similarity between two lists """
    x, y = row['Text NA'], row['Text PR']
    # x, y = row['Text PR'], row['Text PR modified']
    x = clean_text(x)
    y = clean_text(y)
    vectors = CountVectorizer().fit_transform([x, y]).toarray()
    return cosine_similarity(vectors)[0, 1]

def calc_lev_dist(row):
    x, y = row['Text NA'], row['Text PR']
    # x, y = row['Text PR'], row['Text PR modified']
    x = clean_text(x).split(" ")
    y = clean_text(y).split(" ")
    return distance(x, y)

df['Levenshtein']=df.apply(calc_lev_dist, axis=1)
df["Jaccard"] = df.apply(jaccard_similarity, axis=1)
df["Cosine"] = df.apply(calc_cosine_similarity, axis=1)

df["Full Length Indicator PR"] = df["Text PR"].apply(lambda x: len(regex.findall(r'\w+', x)))
df["Clean Length Indicator PR"] = df["Text PR"].apply(lambda x: len(regex.findall(r'\w+', clean_text(x))))
df["Full Length Indicator NA"] = df["Text NA"].apply(lambda x: len(regex.findall(r'\w+', x)))
df["Clean Length Indicator NA"] = df["Text NA"].apply(lambda x: len(regex.findall(r'\w+', clean_text(x))))

df.to_excel('ChurnalismIndex.xlsx')
