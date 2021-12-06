

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
#!pip install sentence-transformers



punctuation=punctuation+ '\n'


df = pd.read_csv('https://raw.githubusercontent.com/mohrj01/MLHW3/master/hotelReviewsInPrague.csv')


text = """Looking for a hotel in Prague near the bridge with free breakfast"""
doc = nlp(text)
sentence_spans = list(doc.sents)
displacy.render(doc, jupyter = True, style="ent")


embedder = SentenceTransformer('all-MiniLM-L6-v2')

df['hotelName'].value_counts()

df_combined = df.sort_values(['hotelName']).groupby('hotelName', sort=False).review_body.apply(''.join).reset_index(name='all_review')

df_combined



df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_combined['all_review']= df_combined['all_review'].apply(lambda x: lower_case(x))



df = df_combined
df_sentences = df_combined.set_index("all_review")
df_sentences.head()



df_sentences = df_sentences["hotel"].to_dict()
df_sentences_list = list(df_sentences.keys())
len(df_sentences_list)



list(df_sentences.keys())[:5]


#%%

df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]

#%%

# Corpus with example sentences
corpus = df_sentences_list
corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)

#%%

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
queries = ['Hotel closest to bridge',
           'walking distance to water'
           ]


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print("(Score: {:.4f})".format(score))
        print(corpus[idx], "(Score: {:.4f})".format(score))
        row_dict = df.loc[df['all_review']== corpus[idx]]
        print("paper_id:  " , row_dict['hotelName'] , "\n")
    # for idx, distance in results[0:closest_n]:
    #     print("Score:   ", "(Score: %.4f)" % (1-distance) , "\n" )
    #     print("Paragraph:   ", corpus[idx].strip(), "\n" )
    #     row_dict = df.loc[df['all_review']== corpus[idx]]
    #     print("paper_id:  " , row_dict['Hotel'] , "\n")
    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """


#%%

model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
embeddings = model.encode(corpus)
#print(embeddings)

#%%

with open("corpus.pkl" , "wb") as file1:
  pkl.dump(corpus,file1)

with open("corpus_embeddings.pkl" , "wb") as file2:
  pkl.dump(corpus_embeddings,file2)

with open("df.pkl" , "wb") as file3:
  pkl.dump(df,file3)