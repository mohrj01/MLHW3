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




embedder = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')


paraphrases = util.paraphrase_mining(model, corpus)
#query_embeddings_p =  util.paraphrase_mining(model, queries,show_progress_bar=True)

#%%

queries = ['Hotel closest to bridge',
           'hotel closest to water'
           ]
query_embeddings = embedder.encode(queries,show_progress_bar=True)

#%%

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 5
print("\nTop 5 most similar sentences in corpus:")
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n=========================================================")
    print("==========================Query==============================")
    print("===",query,"=====")
    print("=========================================================")


    for idx, distance in results[0:closest_n]:
        print("Score:   ", "(Score: %.4f)" % (1-distance) , "\n" )
        print("Paragraph:   ", corpus[idx].strip(), "\n" )
        row_dict = df.loc[df['all_review']== corpus[idx]]
        print("paper_id:  " , row_dict['hotelName'] , "\n")
        # print("Title:  " , row_dict["title"][corpus[idx]] , "\n")
        # print("Abstract:  " , row_dict["abstract"][corpus[idx]] , "\n")
        # print("Abstract_Summary:  " , row_dict["abstract_summary"][corpus[idx]] , "\n")
        print("-------------------------------------------")

#%%

from sentence_transformers import SentenceTransformer, util
import torch
embedder = SentenceTransformer('all-MiniLM-L6-v2')

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

#%%

# Query sentences:





st.header("Prague Hotel Finder")
st.write("Jessica Mohr | MABA ML HW3 | [Github](%s)" % "https://github.com/mohrj01/MLHW3")
user_input = st.text_input("What type of hotel are you searching for?", value="")

queries = [str(user_input)]
query_embeddings = embedder.encode(user_input,show_progress_bar=True)

top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    st.write("\n\n======================\n\n")
    st.write("Query:", query)
    st.write("\nTop 5 most similar hotels:")

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






#st.write(len(df['hotelName'][0]))
#st.write(len(df2['hotelName'][0]))

#st.write("8", df['hotelName'][0], "8")
#st.write("8", df2['hotelName'][0], "8")

st.header("Note:")
st.write("Eric and Dev both helped with debugging")
