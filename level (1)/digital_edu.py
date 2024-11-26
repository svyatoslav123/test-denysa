
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandordScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_metrix
import joblib
data = pd.read.csv("train.csv")
data["bdate"] = data["bdate"].fillna("01.01.1900")