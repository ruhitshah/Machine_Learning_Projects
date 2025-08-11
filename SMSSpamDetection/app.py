import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the saved model and vectorizer
model = joblib.load('https://github.com/ruhitshah/Machine_Learning_Projects/blob/main/SMSSpamDetection/model/sms_spam_model.pkl')
vectorizer = joblib.load('https://github.com/ruhitshah/Machine_Learning_Projects/blob/main/SMSSpamDetection/model/tfidf_vectorizer.pkl')

# Function to preprocess the input text (similar to what was done for training data)
def remove_punc(text):
    url_pattern = r'https?://\S+|www\.\S+'
    mention_pattern = r'@\w+'
    text = re.sub(url_pattern, '', text)  
    text = re.sub(mention_pattern, '', text) 
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text.strip()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word not in stop_words])

def stem_words(text):
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text.split()])

# Streamlit app interface
st.title("SMS Spam Detection")

# User input box for SMS text
sms_text = st.text_area("Enter the SMS to check if it's spam or not:")

# When the user submits the input
if st.button("Classify"):
    # Preprocess the input text
    sms_text = remove_punc(sms_text)
    sms_text = remove_stopwords(sms_text)
    sms_text = stem_words(sms_text)

    # Vectorize the input text
    sms_vec = vectorizer.transform([sms_text])

    # Predict using the model
    prediction = model.predict(sms_vec)

    # Display the result
    if prediction == 1:
        st.subheader("This message is SPAM!")
    else:
        st.subheader("This message is NOT SPAM.")
