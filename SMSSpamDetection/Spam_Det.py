import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df1 = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding='latin1')
df1.head()
df1.info()
df1 = df1.dropna(how="any", axis=1)
df1.columns = ['label', 'message']

df1.head()
label_counts = df1['label'].value_counts()
print("label counts:",label_counts)
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
def remove_punc(text):
    # Define regex patterns
    url_pattern = r'https?://\S+|www\.\S+'
    mention_pattern = r'@\w+'
    
    # Remove URLs, mentions, punctuations
    text = re.sub(url_pattern, '', text)  
    text = re.sub(mention_pattern, '', text) 
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    return text.strip()

df1['wo_punc'] = df1['message'].apply(lambda text: remove_punc(text))
df1.head()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

df1["wo_stop"] = df1["wo_punc"].apply(lambda text: remove_stopwords(text))
df1.head()
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

df1["stemmed_text"] = df1["wo_stop"].apply(lambda text: stem_words(text))

df1.head()
from sklearn.model_selection import train_test_split
X = df1['stemmed_text']
y = df1['label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

vocab_size = len(vectorizer.vocabulary_)
print(f'Vocabulary Size: {vocab_size}')
from sklearn.linear_model import LogisticRegression
# Initialize the classifier
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("The Accuracy Score of The Model is:",accuracy)


##The Accuracy Score of The Model is: 0.9524663677130045
