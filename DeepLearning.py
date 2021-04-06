import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split

from zipfile import ZipFile
import os

# Cargamos los datos de entrenamiento

uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall()
csv_path = "jena_climate_2009_2016.csv"

df_train = pd.read_csv(csv_path, usecols=["T (degC)"], nrows=10000)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df_train.values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(df_train.values)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Cargamos los datos de cansat

df_cansat = pd.read_csv("cansat_df.csv", nrows=884)
scaled_data_cansat = scaler.fit_transform(df_cansat.values.reshape(-1,1))

x_test = []

for j in range(prediction_days, len(df_cansat.values)):
    x_test.append(scaled_data_cansat[j-prediction_days:j, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Construimos el modelo

model = Sequential()

model.add(LSTM(units=50, return_sequences=True,
               input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=2, batch_size=32)

prediction = model.predict(x_test[-100:])
prediction = scaler.inverse_transform(prediction)

x_matplol = []
for g in range (len(df_cansat.values), len(prediction)+len(df_cansat.values)):
    x_matplol.append(g)

plt.plot(x_matplol, prediction, 100, color="r")
plt.plot(df_cansat.values, color="g")
plt.ylim(-10, 30)
plt.show()
