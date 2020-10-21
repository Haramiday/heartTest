import numpy as np
import pickle
import pandas as pd
import streamlit as st

pk_file = open("classifier.pkl","rb")
classifier = pickle.load(pk_file)

def model_prediction(thalach,oldpeak):

    pred = classifier.predict([[thalach,oldpeak]])
    return pred

def main():
    st.title("Heart Diseases Classification app")
    thalach = st.text_input('Maximum heart rate achieved',"Enter here")
    oldpeak = st.text_input('ST depression induced by exercise relative to rest',"Enter here")
    result = ""
    if st.button("Classify"):
        result = model_prediction(thalach,oldpeak)
    st.write("1 means You have heart diseases")
    st.write("0 means You have no heart diseases")
    st.success("Result is {}".format(result))

if __name__=="__main__":
    main()
