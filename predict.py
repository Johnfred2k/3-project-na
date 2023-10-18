import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

def read():
    df = pd.read_parquet('data/yellow_tripdata_2021-01.parquet')

def preprocess():
    df['duration'] = df.tpep_dropoff_datetime-df.tpep_pickup_datetime
    df['duration_minutes'] = df['duration'] / pd.Timedelta(minutes=1)
    columns_to_drop = ['duration', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'RatecodeID', 'store_and_fwd_flag', 'passenger_count', 'congestion_surcharge']
    df.drop(columns=columns_to_drop, inplace=True)
    df['airport_fee'].fillna(0, inplace=True)
    X = df.drop('duration_minutes', axis = 1)
    y = df['duration_minutes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

def predict():# define features and target
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_scaled, y_train)
    return rf_regressor.predict(X_test_transformed)