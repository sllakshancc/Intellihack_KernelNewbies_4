# KernelNewbies

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# dataset
data = pd.read_csv("Watches Bags Accessories - Watches Bags Accessories.csv", encoding = 'ISO-8859-1')

# preprocessing data
data['Rating in Stars'] = pd.to_numeric(data['Rating in Stars'], errors = 'coerce')
data['Current Price'] = data['Current Price'].str.replace('Rs.', '').str.replace(',', '').astype(float)
data['Original Price'] = data['Original Price'].str.replace('Rs.', '').str.replace(',', '').astype(float)

# function to preprocess sold count - remove K and sold
def convert_sold_count(value):
    if pd.isna(value):
        return np.nan
    if 'K' in value:
        return float(value.replace('K', '').replace(' Sold', '')) * 1000
    return float(value.replace(' Sold', ''))

# preprocessing data
data['Sold Count'] = data['Sold Count'].apply(convert_sold_count)
data['Sold Count'].fillna(data['Sold Count'].median(), inplace = True)
data['Original Price'].fillna(data['Original Price'].median(), inplace = True)

# engineer a new feature, measures the attraction 
data['Price Reduction Percentage'] = (data['Original Price'] - data['Current Price']) / data['Original Price'] * 100

# input features
features = ['Rating Count', 'Sold Count', 'Voucher', 'Delivery', 'Current Price', 'Original Price', 'Price Reduction Percentage', 'Category']
# output target
target = 'Sold Count'

# feature seperation
categorical_cols = ['Voucher', 'Delivery', 'Category']
numeric_cols = ['Rating Count', 'Current Price', 'Original Price', 'Price Reduction Percentage']

# create transformers
numeric_transformer = Pipeline(steps = [
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps = [
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])
preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# split dataset to training and testing data, 20% of data are used for testing
X = data[features].drop(columns = [target])
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# transform data
X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)

# get the training models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state = 0),
    'Random Forest': RandomForestRegressor(random_state = 0)
}

# generate predcitions and get errors
results = {}
for name, model in models.items():
    model.fit(X_train_prepared, y_train)
    y_pred = model.predict(X_test_prepared)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = (mae, r2)

# display MAE and R-squared
for name, (mae, r2) in results.items():
    print(f"{name} -> MAE: {mae}, R2: {r2}")