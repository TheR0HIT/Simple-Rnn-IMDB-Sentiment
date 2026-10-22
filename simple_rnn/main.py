import numpy as np 
import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from keras.utils import pad_sequences


# load the imdb dataset word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'simple_rnn_imdb.h5'))

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
# Function to preprocess user input
MAX_FEATURES = 1000   # must match the max_features you used in training

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []
    for w in words:
        idx = word_index.get(w)
        if idx is not None and idx + 3 < MAX_FEATURES:
            encoded_review.append(idx + 3)
        else:
            encoded_review.append(2)   # unknown / out-of-vocab
    return pad_sequences([encoded_review], maxlen=500)

### Prediction function
def predict_sentiment (review):
    
    preprocessed_input=preprocess_text (review)
    prediction=model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] >0.5 else 'Negative'
    return sentiment, prediction[0][0]


# streamlit app
import streamlit as st
st.title('Imdb Movie Review Sentiment Analysis')
st.write('Enter the movie review to classify is as postive or negative. ')

# user input
user_input=st.text_area('Movie Review ')
if st.button('classify'):
    preprocess_input=preprocess_text(user_input)
    # make the predication
    prediction=model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] >0.5 else 'Negative'

    # display
    st.write(f"sentiment : {sentiment}")
    st.write(f"predication score : {prediction[0][0]}")
else:
    st.write('Please enter the moview review. ')