import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures   
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
import pickle
from sklearn.externals import joblib

# %matplotlib inline

df = pd.read_csv('data/insurance.csv')

print(df.head())
missing_values = df.isnull().sum().sort_values(ascending = False)
missing_values = missing_values[missing_values > 0]/df.shape[0]
print(f'{missing_values *100} %')

st = df.apply(LabelEncoder().fit_transform)

print(st.head())
