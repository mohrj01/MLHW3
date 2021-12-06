import streamlit as st
import pickle as pkl

with open('df.pkl', 'rb') as file1:
    df = pkl.load(file1)
    
print(df.head())
#st.write(df.head())
st.table(df)
