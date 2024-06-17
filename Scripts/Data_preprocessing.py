import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load data
data = pd.read_csv(r'C:\Users\Jad_S\Documents\Python Scripts\Credit card Fraud Model\data\creditcard.csv')

# Handling Imbalance
X = data.drop('Class', axis=1)
y = data['Class']
rus = RandomUnderSampler()
X_res, y_res = rus.fit_resample(X, y)

# Split into train-test sets and save processed data if needed
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the processed data
with open(r'C:\Users\Jad_S\Documents\Python Scripts\Credit card Fraud Model\data\X_res.pkl', 'wb') as file:
    pickle.dump(X_res, file)
with open(r'C:\Users\Jad_S\Documents\Python Scripts\Credit card Fraud Model\data\y_res.pkl', 'wb') as file:
    pickle.dump(y_res, file)

