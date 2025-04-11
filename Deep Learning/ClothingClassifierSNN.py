import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import math

data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

training_data = data['train']
testing_data = data['test']

class_names = metadata.features['label'].names

# Normalizar los datos para que estén entre 0 y 1 y facilitar el entrenamiento
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

training_data = training_data.map(normalize)
testing_data = testing_data.map(normalize)

# Guardar en caché (velocidad del entrenamiento)
training_data = training_data.cache()
testing_data = testing_data.cache()

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(training_data.take(25)):
    imagen = image.numpy().reshape((28, 28))
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagen, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()

# Crear el modelo
entrada = tf.keras.layers.Flatten(input_shape=(28, 28, 1)) # 28x28 = 784, 1 por Blanco y Negro
capa1 = tf.keras.layers.Dense(50, activation = tf.nn.relu)
capa2 = tf.keras.layers.Dense(50, activation = tf.nn.relu)
salida = tf.keras.layers.Dense(10, activation = tf.nn.softmax) #Para redes de clasificación

model = tf.keras.Sequential([entrada, capa1, capa2, salida])

model.compile(
    optimizer = "adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)

num_datos_training = metadata.splits["train"].num_examples
num_datos_testing = metadata.splits["test"].num_examples

TAMANO_LOTE = 32

training_data = training_data.repeat().shuffle(num_datos_training).batch(TAMANO_LOTE)
testing_data = testing_data.batch(TAMANO_LOTE)

#Entrenar el modelo
historial = model.fit(
    training_data,
    epochs = 5,
    steps_per_epoch = math.ceil(num_datos_training/TAMANO_LOTE)
)

plt.xlabel("Epoch")
plt.ylabel("Loss Magnitude")
plt.plot(historial.history["loss"])

for test_images, test_labels in testing_data.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[..., 0], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"
    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100*np.max(predictions_array),
        class_names[true_label]),
        color=color
    )

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")

num_filas = 5
num_columnas = 5
num_imagenes = num_filas*num_columnas
plt.figure(figsize=(2*2*num_columnas, 2*num_filas))
for i in range(num_imagenes):
    plt.subplot(num_filas, 2*num_columnas, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_filas, 2*num_columnas, 2*i+2)
    plot_value_array(i, predictions, test_labels)

# Tomar cualquier indice del set de pruebas
imagen = test_images[32]
imagen = np.array([imagen])
prediction = model.predict(imagen)
print("Predicción: ", class_names[np.argmax(prediction[0])])
