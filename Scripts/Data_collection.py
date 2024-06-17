import pandas as pd


# DataExploration
data = pd.read_csv(r'C:\Users\Jad_S\Documents\Python Scripts\Credit card Fraud Model\data\creditcard.csv')
print(data.head())
print(data.info())
print(data.describe())

# Cleaning Data
data_cleaned = data.dropna()
data_cleaned = data.dropna(axis=1)
print(data.head())
print(data.info())
print(data.describe())


