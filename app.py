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
import scipy
import torch

with open("df.pkl" , "rb") as file3:
    df = pkl.load(file3)
with open('corpus_embeddings.pkl', 'rb') as file2:
    corpus_embeddings = pkl.load(file2)
with open('corpus.pkl', 'rb') as file1:
    corpus = pkl.load(file1)
 

df2 = pd.read_csv('https://raw.githubusercontent.com/mohrj01/MLHW3/master/HotelListInPrague.csv')
df['hotelName'] = df['hotelName'].str.replace('\d+', '')
def myreplace(s):
    for ch in ['Name: hotel_name, dtype: object']:
        s = s.replace(ch, '')

    # remove extra spaces
    s = re.sub(' +', ' ', s)
    s = s.rstrip().lstrip()
    return s

df['hotelName'] = df['hotelName'].map(myreplace)
df2['hotelName'] = df2['hotel_name']

df = pd.merge(df,df2)

df['price_per_night'] = df['price_per_night'].fillna("No Price Available")


# Embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')
paraphrases = util.paraphrase_mining(model, corpus)
#corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


st.header("Prague Hotel Finder")
st.write("Jessica Mohr | MABA ML HW3 | [Github](%s)" % "https://github.com/mohrj01/MLHW3")
user_input = st.text_input("What type of hotel are you searching for?", value="")

queries = [str(user_input)]
query_embeddings = embedder.encode(user_input,show_progress_bar=True)


stop_words=list(STOP_WORDS)
punctuation=punctuation+ '\n'
def plot_cloud(wordcloud):
    plt.figure()
    plt.imshow(wordcloud) 
    plt.axis("off");
        
        
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    st.write("\nTop 5 most similar hotels:")
    st.write("\n\n======================\n\n")

    for score, idx in zip(top_results[0], top_results[1]):
        st.write("(Score: {:.4f})".format(score))
        row_dict = df.loc[df['all_review']== corpus[idx]]
        l=[]
        for i in row_dict['hotelName']:
            l.append(i)
        #l[0] = l[0].replace("Name: hotel_name, dtype: object", "")
        st.write("Hotel Name: ", l[0])
        st.write("Price Per Night: ", row_dict['price_per_night'].values[0])
        st.write("[Link to Hotel](%s)" % row_dict['url'].values[0])
        st.write("\n\n======================\n\n")
        
        
        wordcloud = WordCloud(stopwords = stop_words).generate(corpus[idx])
        fig, ax = plt.subplots()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot(fig)
        st.set_option('deprecation.showPyplotGlobalUse', False)






#st.write(len(df['hotelName'][0]))
#st.write(len(df2['hotelName'][0]))

#st.write("8", df['hotelName'][0], "8")
#st.write("8", df2['hotelName'][0], "8")

st.header("Note:")
st.write("Eric and Dev both helped with debugging")
