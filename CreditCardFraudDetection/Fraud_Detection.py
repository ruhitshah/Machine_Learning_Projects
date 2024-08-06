import numpy as np
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
fraud_train = pd.read_csv("/kaggle/input/fraud-detection/fraudTrain.csv")
fraud_train.head()
fraud_train.info()
fraud_test = pd.read_csv("/kaggle/input/fraud-detection/fraudTest.csv")
fraud_test.head()
fraud_test.info()
fraud_train = fraud_train.drop(columns = 'Unnamed: 0')
fraud_test = fraud_test.drop(columns = 'Unnamed: 0')
fraud_train.columns
fraud_test.columns
fraud_train.head()
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
fraud_train['trans_date_trans_time']=encoder.fit_transform(fraud_train['trans_date_trans_time'])
fraud_train['merchant']=encoder.fit_transform(fraud_train['merchant'])
fraud_train['category']=encoder.fit_transform(fraud_train['category'])
fraud_train['first']=encoder.fit_transform(fraud_train['first'])
fraud_train['last']=encoder.fit_transform(fraud_train['last'])
fraud_train['gender']=encoder.fit_transform(fraud_train['gender'])
fraud_train['street']=encoder.fit_transform(fraud_train['street'])
fraud_train['city']=encoder.fit_transform(fraud_train['city'])
fraud_train['state']=encoder.fit_transform(fraud_train['state'])
fraud_train['job']=encoder.fit_transform(fraud_train['job'])
fraud_train['dob']=encoder.fit_transform(fraud_train['dob'])
fraud_train['trans_num']=encoder.fit_transform(fraud_train['trans_num'])

fraud_test['trans_date_trans_time']=encoder.fit_transform(fraud_test['trans_date_trans_time'])
fraud_test['merchant']=encoder.fit_transform(fraud_test['merchant'])
fraud_test['category']=encoder.fit_transform(fraud_test['category'])
fraud_test['first']=encoder.fit_transform(fraud_test['first'])
fraud_test['last']=encoder.fit_transform(fraud_test['last'])
fraud_test['gender']=encoder.fit_transform(fraud_test['gender'])
fraud_test['street']=encoder.fit_transform(fraud_test['street'])
fraud_test['city']=encoder.fit_transform(fraud_test['city'])
fraud_test['state']=encoder.fit_transform(fraud_test['state'])
fraud_test['job']=encoder.fit_transform(fraud_test['job'])
fraud_test['dob']=encoder.fit_transform(fraud_test['dob'])
fraud_test['trans_num']=encoder.fit_transform(fraud_test['trans_num'])
fraud_train.head()
fraud_test.head()
fraud_train.columns
fraud_test.columns
fraud_train.shape,fraud_test.shape
x_train=fraud_train.drop(columns='is_fraud')
y_train=fraud_train['is_fraud']
x_test=fraud_test.drop(columns='is_fraud')
y_test=fraud_test['is_fraud']
from sklearn.tree import DecisionTreeClassifier

# Instantiate the classifier
dtc = DecisionTreeClassifier()

# Fit the model
dtc.fit(x_train, y_train)

from sklearn.metrics import accuracy_score 
y_test_pred = dtc.predict(x_test)
accuracy = accuracy_score(y_test,y_test_pred)
print("The Accuracy of Decision Tree Classifier is :",accuracy)

##The Model Accuracy is 0.9824281696324941

import seaborn as sns
import matplotlib.pyplot as plt
sns.kdeplot(y_test, color='blue', linewidth=3, label='Given')
sns.kdeplot(y_test_pred, color='red', linewidth=3, linestyle='--', label='Predicted')
plt.show()
