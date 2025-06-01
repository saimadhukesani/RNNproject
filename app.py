import numpy as ny
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
#Load the imdb dataset word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}
model=load_model('my_model.keras')

def decode_review(encode_review):
  return ' '.json([reverse_word_index.get(i -3,'?') for i in encoded_review])

def preprocess_text(text):
  words=text.lower().split()
  encoded_review=[word_index.get(word,2)+3 for word in words]
  paded_review=sequence.pad_sequences([encoded_review],maxlen=500)
  return paded_review
#predction function
def predction_sentiment(review):
  preprocessed_input=preprocess_text(review)
  prediction=model.predict(preprocessed_input)
  sentiment='positive' if prediction[0][0]>0.5 else 'negative'
  return sentiment,prediction[0][0]


st.title('Imdb movie Review sentiment Analysis')
st.write('Enter a movie review to classofy it as positive or negative')

#user input
user_input=st.text_area('Enter your review here',height=200)
if st.button('Classify'):
  preprocess_input=preprocess_text(user_input)
  sentiment,score=predction_sentiment(user_input)
  predction=model.predict(preprocess_input)
  statement='Positive' if predction[0][0]>0.5 else 'Negative'
  st.write(f'Sentiment:{statement},Score:{score}')
else:
  st.write('Please enter your review')
