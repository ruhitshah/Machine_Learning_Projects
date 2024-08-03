import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data_path = r"/kaggle/input/genre-classification-dataset-imdb/Genre Classification Dataset/train_data.txt"
train_data = pd.read_csv(train_data_path,sep=":::",nrows=54200,names=["Movie_name","Genre","Description"],engine="python")
train_data.head()
train_data.info()
train_data.isnull().sum()
test_data_path = r"/kaggle/input/genre-classification-dataset-imdb/Genre Classification Dataset/test_data.txt"
test_data = pd.read_csv(test_data_path,sep=":::",names=["Movie_name","Description"],engine="python")
test_data.head()
test_data.info()
test_data.isnull().sum()
test_data_sol_path = r"/kaggle/input/genre-classification-dataset-imdb/Genre Classification Dataset/test_data_solution.txt"
test_data_sol_path = pd.read_csv(test_data_sol_path,sep=":::",names=["Movie_name","Description"],engine="python")
test_data_sol_path.head()
test_data_sol_path.info()
test_data_sol_path.isnull().sum()
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
# Initialize the stemmer and stop words
stemmer = LancasterStemmer()
stop_words = set(stopwords.words('english'))

# Define the clean_text function
def clean_text(text):
    text = text.lower()  # Lowercase all characters
    text = re.sub(r'@\S+', '', text)  # Remove Twitter handles
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub(r"[^a-zA-Z+']", ' ', text)  # Keep only characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text + ' ')  # Keep words with length > 1 only
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')  # Remove stopwords
    text = " ".join([i for i in words if i not in stopwords and len(i) > 2])
    text = re.sub("\s[\s]+", " ", text).strip()  # Remove repeated/leading/trailing spaces
    return text

# Apply the clean_text function to the 'Description' column in the training and test data
train_data['Text_cleaning'] = train_data['Description'].apply(clean_text)
test_data['Text_cleaning'] = test_data['Description'].apply(clean_text)
train_data.head()
train_data.info()
test_data.head()
test_data.info()
#Train,Test Split
x = train_data['Text_cleaning']
y = train_data['Genre']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=123)
x.info()
y.info()
# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
x_train = tfidf_vectorizer.fit_transform(x_train)

# Transform the validation data
x_val = tfidf_vectorizer.transform(x_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
model = LogisticRegression(random_state=123)
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_val)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Validation Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
