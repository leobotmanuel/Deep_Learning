import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM

def prepare_data(x, len_seq):
    X = []
    y = []
    for i in range(len(x) - len_seq):
        X.append(x[i:i+len_seq])
        y.append(x[i+len_seq])
    return np.array(X), np.array(y)

data = pd.read_csv('jena_climate_2009_2016.csv', index_col=0)
x = data["T (degC)"].values
len_seq = 121

X, y = prepare_data(x, len_seq)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
print(X.shape)
print(y.shape)

model = Sequential()
model.add(LSTM(units=64, return_sequences=False, input_shape=(len_seq, 1)))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, y, batch_size=365, epochs=20, validation_split=0.1)

predict = model.predict(X)
predict.flatten()

%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(y, marker='.', linestyle='None', color='c')
plt.plot(predict, color='r')
plt.show()
