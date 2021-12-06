import streamlit as st
import pickle as pkl

with open("df.pkl" , "rb") as file3:
    df = pkl.load(file3)
with open('corpus_embeddings.pkl', 'rb') as file2:
    corpus_embeddings = pkl.load(file2)
with open('corpus.pkl', 'rb') as file1:
    corpus = pkl.load(file1)
    
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    
print(df.head())
#st.write(df.head())
st.table(df)

st.table(corpus)
st.table(corpus_embeddings)
