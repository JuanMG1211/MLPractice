import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set up training data
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Una sola  entrada y una neurona de salida
# capa = tf.keras.layers.Dense(units=1, input_shape=[1])
# model = tf.keras.Sequential([capa])

# Otra forma de hacerlo con más capas y neuronas
entrada = tf.keras.layers.Dense(units=3, input_shape=[1])
capa = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([entrada, capa, salida])
#Mucho más rápido que el modelo anterior EN ESTE CASO


model.compile(
    loss='mean_squared_error',

    #¿Cuánto se va a mover el modelo en cada paso?
    optimizer=tf.keras.optimizers.Adam(0.1)
)

# Epochs = número de iteraciones
history = model.fit(celsius_q, fahrenheit_a, epochs=200, verbose=False)
print("Finished training the model")

plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])

result = model.predict(np.array([100]))
print(result)

# Pesos y sesgos que aprendió el modelo
constantes_capa1 = entrada.get_weights()
constantes_capa2 = capa.get_weights()
constantes_capa3 = salida.get_weights()

print(constantes_capa1)
print(constantes_capa2)
print(constantes_capa3)