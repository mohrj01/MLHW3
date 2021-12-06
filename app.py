import streamlit as st
import pickle as pkl
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

with open("df.pkl" , "rb") as file3:
    df = pkl.load(file3)
with open('corpus_embeddings.pkl', 'rb') as file2:
    corpus_embeddings = pkl.load(file2)
with open('corpus.pkl', 'rb') as file1:
    corpus = pkl.load(file1)
 

embedder = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')
#corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    
#print(df.head())
#st.write(df.head())
#st.table(df)

st.table(corpus)
st.table(corpus_embeddings)
