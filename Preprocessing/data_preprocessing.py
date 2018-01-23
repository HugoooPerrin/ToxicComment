import sys
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def text_to_caractlist(text, remove_stopwords=True, stem_words=True):
    
    special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)
    replace_numbers=re.compile(r'\d+',re.IGNORECASE)

    #Remove Special Characters
    text=special_character_removal.sub('',text)
    
    #Replace Numbers
    text=replace_numbers.sub('n',text)

    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return text


def Carac2Vect(sentence):

    maxlen = 300

    correspondance = {a : i for i,a in enumerate(list(' abcdefghijklmnopqrstuvwxyz'))}

    as_serie = pd.Series(list(sentence))

    diff = maxlen - as_serie.count()

    if diff <= 0:
        as_serie = as_serie.truncate(after=maxlen-1)
    else:
        as_serie = pd.concat([as_serie, pd.Series(np.repeat(np.nan, diff, axis=0))])

    as_serie = as_serie.map(correspondance)

    as_serie = as_serie.fillna(0)

    return as_serie.tolist()


