# ARGONAUTEX 2021 --> Deep Learning for Time series predictions

# Importamos las librerias

from tensorflow import keras
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

# Importamos el dataset de entrenamiento y de validacion

df = pd.read_csv("jena_climate_2009_2016.csv", usecols= ["Date Time", "p (mbar)", "T (degC)", "rh (%)"])
df.to_csv('bigdataset.csv', index=False)

data_dir = '/home/leonardo/Documentos/saul'
fname = os.path.join(data_dir, 'bigdataset.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:420552]

# Importamos el dataset del CanSat

cansat_dir = '/home/leonardo/Documentos/saul'
cname = os.path.join(cansat_dir, 'datos_del_CanSat.csv')

c = open(cname)
data_cansat = c.read()
c.close()

lines_cansat = data_cansat.split('\n')
header_cansat = lines_cansat[0].split(',')
lines_cansat = lines_cansat[1:100]

# Parseamos el dataset de entrenamiento y de validacion

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

# Parseamos los datos del cansat

float_data_cansat = np.zeros((len(lines_cansat), len(header_cansat) - 1))
for i, line_cansat in enumerate(lines_cansat):
    values_cansat = [float(l) for l in line_cansat.split(',')[1:]]
    float_data_cansat[i, :] = values_cansat
    
# Definimos el valor que vamos a predecir, temperatura

temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)
plt.show()

plt.plot(range(551), temp[420000:])
plt.show()

# Escalamos los datos a cierto rango para ajustarlos al optimizador rmsprop

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

# Generamos los datasets de entrenamiento, de validacion y del cansat

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
        
lookback = 1440 # Cuantos datos se escogen del pasado para estudiar las futuras tendencias
step = 6 # Cuantos datos nos saltamos para hacer el generador
delay = 144 # Numero de pasos de tiempo para la prediccion
batch_size = 128 # Numero de datos que se emplearan en cada epoch

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=420000,
                     step=step,
                     batch_size=batch_size)

# Pasos del generador de validacion

val_steps = (300000 - 200001 - lookback) // batch_size

# Pasos del generador del cansat

test_steps = (420000 - 300001 - lookback) // batch_size

# Definimos el modelo y lo compilamos

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit(train_gen,
                    steps_per_epoch=500,
                    epochs=2,
                    validation_data=val_gen,
                    validation_steps=val_steps)

# Realizamos las predicciones

future_temperature = model.predict(test_gen, steps = test_steps)
print("Temperatura tras " + str(delay) + " pasos de tiempo: ")
fut = np.concatenate(future_temperature)
prev = np.mean(fut)

# Desescalado de datos para mostrar la prediccion en grados celsius

stda = float_data[300001:420000].std(axis=0)
prev *= stda
meana = float_data[300001:420000].mean(axis=0)
prev += meana

# Mostramos el valor de la predicion

print(np.mean(prev))

# Mostramos una grafica con la prediccion y los ultimos datos registrados

plt.plot(temp[419000:420000])
plt.plot(delay+1000,np.mean(prev)*(1), marker="X", color="red")
plt.show()
