# Import dependencies
import pandas as pd
import numpy as np

url = 'credit_clean_data.csv'
credit_data = pd.read_csv(url)
credit_data = credit_data.drop(['DriversLicense', 'ZipCode'], axis = 1)

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

#le = LabelEncoder()
# Iterate over all the values of each column and extract their dtypes
#for col in credit_data.columns:
    # Compare if the dtype is object
    #if credit_data[col].dtypes=='object':
    # Use LabelEncoder to do the numeric transformation
        #credit_data[col]=le.fit_transform(credit_data[col])

credit_data = pd.get_dummies(credit_data, columns=(credit_data.columns.dtypes == 'object'), dummy_na=True)

X = credit_data.iloc[:, 0:13]
y = credit_data.iloc[:,13]

# Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.33,
                                random_state=42)

# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression
# Instantiate a LogisticRegression classifier with default parameter values
lr = LogisticRegression()

# Fit logreg to the train set
lr.fit(rescaledX_train, y_train)


# Save your model
import joblib
joblib.dump(lr, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
