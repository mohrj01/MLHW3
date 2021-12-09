# imports
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import os
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
stopwords=list(STOP_WORDS)
from string import punctuation
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


punctuation=punctuation+ '\n'
df = pd.read_csv('https://raw.githubusercontent.com/mohrj01/MLHW3/master/hotelReviewsInPrague.csv')

# create reviews
embedder = SentenceTransformer('all-MiniLM-L6-v2')
df_combined = df.sort_values(['hotelName']).groupby('hotelName', sort=False).review_body.apply(''.join).reset_index(name='all_review')
df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_combined['all_review']= df_combined['all_review'].apply(lambda x: lower_case(x))
df = df_combined

# create sentences
df_sentences = df_combined.set_index("all_review")
df_sentences = df_sentences["hotel"].to_dict()
df_sentences_list = list(df_sentences.keys())
df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]


# Corpus with example sentences
corpus = df_sentences_list
corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)

# save via pickle
with open("corpus.pkl" , "wb") as file1:
  pkl.dump(corpus,file1)
with open("corpus_embeddings.pkl" , "wb") as file2:
  pkl.dump(corpus_embeddings,file2)
with open("df.pkl" , "wb") as file3:
  pkl.dump(df,file3)
