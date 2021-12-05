import streamlit as st
import pickle as pkl

with open('corpus.pkl', 'rb') as file1:
    corpus = pkl.load(file1)
    
st.write("test")
