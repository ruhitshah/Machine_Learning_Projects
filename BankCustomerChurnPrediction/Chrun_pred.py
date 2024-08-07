import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df1 = pd.read_csv("/kaggle/input/bank-customer-churn-prediction/Churn_Modelling.csv")
df1.head()
df1.info()
df1.describe()
df1 = df.drop('CustomerId', axis=1, errors='ignore')
df1 = df.drop('Surname', axis=1, errors='ignore')
df1.head()
df1.columns
df1.info()
nominal_cols = ['Geography', 'Gender']
data_nominal_encoded = pd.get_dummies(df1[nominal_cols], drop_first=True)
data_nominal_encoded = data_nominal_encoded.astype(int)

# Combine encoded columns back into the DataFrame
df = pd.concat([df1.drop(columns=nominal_cols), data_nominal_encoded], axis=1)
df.head()
y = df[['Exited']]
X = df[['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Geography_Germany','Geography_Spain','Gender_Male']]
y.value_counts()
from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.over_sampling import SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = 0)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)
classifier.score(X_test, y_test)
grid = {
    'n_estimators' : [9, 17, 31, 41, 51],
    'max_depth': range(2, 12),
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6]
}
classifier = RandomForestClassifier(random_state = 0)
gcv = GridSearchCV(estimator=classifier, param_grid=grid, cv=5, verbose=3)
gcv.fit(X_train, y_train)
gcv.best_params_

##{'max_depth': 11,
 'min_samples_leaf': 1,
 'min_samples_split': 4,
 'n_estimators': 51}##

model = gcv.best_estimator_
model.fit(X_train, y_train)
model.score(X_train, y_train)
#For Train Data, Accuracy is: 0.9040816326530612
model.score(X_test, y_test)
#For Test Data Accuracy, is : 0.808

