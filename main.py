import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_splitfrom
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrixfrom
from sklearn.preprocessing import StandartScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv("train.csv")

data["bdate"] = data["bdate"].fillna("01.01.1900")
data["bdate"] = pd.to_datetime(data["bdate"], format = "%d.%m.%Y" , errors = "coerce")

current_year = pd.to_datetime("today").year
data["age"] = current_year = data["bdate"]dt.year
data["age"] = data['age'].fillna(data["age"].median())
data.drop(columns=["bdate"], inplace=True)

data['last_seen'] = pd.to_datetime(data['last_seen'], errors='coerce')
data['last_see_year'] = data[';ast_seen'].dt.year
data.drop(columns=['last_seen'], inplace=True)

data['num_langs'] = data['langs'].apply(lambda x: len(str(x).split(';')))
data.drop(columns=['langs'], inplace=True)
categorical_columns = ["sex", "has_mobile", 'education_from', 'relation' 'education_status', 'occupation_type']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=)



data['career_start'] = pd.to_numeric(data['career_start'], errors='coerce').fillna(0).astype(float)
data['career_end'] = pd.to_numeric(data['career_end'], errors='coerce').fillna(0).astype(float)

X = data.drop(columns=['id', 'result'])
y = data['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for col in X_test.select_dtypes(include=['bool']).columns:
    X_train[col] = X_train[col].astype(int)
    X_test[col] = X_train[col].astype(int)

for col in X_test.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[col] = X_train[col].astype(int)
    X_test[col] = X_train[col].astype(int)


