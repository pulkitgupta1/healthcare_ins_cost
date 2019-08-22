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

sns.set(color_codes=True)
plt.figure(figsize=(14, 12))
sns.heatmap(st.astype(float).corr(),
            linewidths=0.2,
            square=True,
            linecolor='white',
            annot=True,
            cmap="YlGnBu")
plt.show()

g = sns.FacetGrid(df, col="smoker",  size= 5, sharey=False, sharex = True)
g.map(sns.distplot, "charges", color = 'r')
g.set_axis_labels("charges", "proportion")
g.despine(left=True)

plt.figure(figsize=(13,6))
plt.title("Distribution of age")
ax = sns.distplot(df["age"], color = 'purple')
plt.show()

sns.catplot(x="smoker", kind="count", hue = 'sex', data = df , palette='pastel')
plt.show()