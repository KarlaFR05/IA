
# -*- coding: utf-8 -*-
"""
CNN para clasificar imágenes a color usando CIFAR-10.

CAMBIO:
Antes se utilizaba Fashion-MNIST para clasificar prendas de ropa.
Ahora se utiliza CIFAR-10 para clasificar objetos como:
aviones, automóviles, aves, gatos, venados, perros,
ranas, caballos, barcos y camiones.
"""

# ============================================================
# 1. CARGAR EL DATASET
# ============================================================

# CAMBIO 1:
# Antes: from keras.datasets import fashion_mnist
# Ahora importamos CIFAR-10, que contiene imágenes a color.
from keras.datasets import cifar10

(train_X, train_Y), (test_X, test_Y) = cifar10.load_data()

# CAMBIO 2:
# Las etiquetas de CIFAR-10 tienen inicialmente forma (cantidad, 1).
# flatten() las convierte a un arreglo de una sola dimensión.
train_Y = train_Y.flatten()
test_Y = test_Y.flatten()


# ============================================================
# 2. IMPORTAR LIBRERÍAS
# ============================================================

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# SE CONSERVA:
# Estas librerías se utilizan para manejar los datos,
# convertir las etiquetas y mostrar las imágenes.


# ============================================================
# 3. REVISAR LAS DIMENSIONES Y LAS CLASES
# ============================================================

print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

# Encontrar las etiquetas diferentes que existen en el dataset.
classes = np.unique(train_Y)
nClasses = len(classes)

print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


# ============================================================
# 4. MOSTRAR IMÁGENES DEL DATASET
# ============================================================

plt.figure(figsize=[8, 4])

# CAMBIO 3:
# Antes se usaba cmap='gray' porque Fashion-MNIST
# contiene imágenes en escala de grises.
# Ahora mostramos las imágenes originales a color.

# Mostrar la primera imagen de entrenamiento.
plt.subplot(121)
plt.imshow(train_X[0])
plt.title("Ground Truth: {}".format(train_Y[0]))
plt.axis('off')

# Mostrar la primera imagen de prueba.
plt.subplot(122)
plt.imshow(test_X[0])
plt.title("Ground Truth: {}".format(test_Y[0]))
plt.axis('off')

plt.tight_layout()
plt.show()


# ============================================================
# 5. PREPROCESAMIENTO DE LAS IMÁGENES
# ============================================================

# CAMBIO 4:
# ELIMINAMOS estas instrucciones del código original:
#
# train_X = train_X.reshape(-1, 28, 28, 1)
# test_X = test_X.reshape(-1, 28, 28, 1)
#
# CIFAR-10 ya tiene las dimensiones correctas:
# (cantidad de imágenes, 32, 32, 3).
#
# 32 x 32 = tamaño de la imagen.
# 3 = canales de color RGB.

print("Dimensiones de entrenamiento:", train_X.shape)
print("Dimensiones de prueba:", test_X.shape)

# SE CONSERVA:
# Convertir los datos a tipo float32.
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

# SE CONSERVA:
# Normalizar los píxeles para que sus valores estén entre 0 y 1.
train_X = train_X / 255.0
test_X = test_X / 255.0


# ============================================================
# 6. CONVERTIR LAS ETIQUETAS A ONE-HOT ENCODING
# ============================================================

# SE CONSERVA:
# CIFAR-10 también tiene 10 clases, por lo que utilizamos 10.
train_Y_one_hot = to_categorical(train_Y, 10)
test_Y_one_hot = to_categorical(test_Y, 10)

print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])


# ============================================================
# 7. SEPARAR LOS DATOS DE ENTRENAMIENTO Y VALIDACIÓN
# ============================================================

from sklearn.model_selection import train_test_split

# SE CONSERVA:
# El 80% de los datos de entrenamiento se utiliza para entrenar.
# El 20% restante se utiliza para validar el modelo.
train_X, valid_X, train_label, valid_label = train_test_split(
    train_X,
    train_Y_one_hot,
    test_size=0.2,
    random_state=13
)

print("Forma de entrenamiento:", train_X.shape)
print("Forma de validación:", valid_X.shape)
print("Etiquetas de entrenamiento:", train_label.shape)
print("Etiquetas de validación:", valid_label.shape)


# ============================================================
# 8. IMPORTAR LAS CAPAS Y HERRAMIENTAS DE KERAS
# ============================================================

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU

# SE CONSERVA:
# Sequential permite construir el modelo capa por capa.
# Conv2D detecta características en las imágenes.
# MaxPooling2D reduce las dimensiones de los mapas de características.
# Flatten convierte los mapas en un vector.
# Dense conecta las neuronas para realizar la clasificación.
# LeakyReLU introduce una función de activación.


# ============================================================
# 9. CONFIGURAR LOS PARÁMETROS
# ============================================================

batch_size = 64
epochs = 5
num_classes = 10

# SE CONSERVA:
# Los parámetros son inicialmente los mismos que en el código original.
# Más adelante puedes aumentar epochs para probar el rendimiento.


# ============================================================
# 10. CONSTRUIR LA RED NEURONAL CONVOLUCIONAL (CNN)
# ============================================================

cifar_model = Sequential()

# CAMBIO 5:
# Antes: input_shape=(28, 28, 1)
# Ahora: input_shape=(32, 32, 3)
#
# CIFAR-10 tiene imágenes de 32 x 32 píxeles y tres canales RGB.
cifar_model.add(Conv2D(
    32,
    kernel_size=(3, 3),
    activation='linear',
    input_shape=(32, 32, 3),
    padding='same'
))

# SE CONSERVA:
cifar_model.add(LeakyReLU(negative_slope=0.1))
cifar_model.add(MaxPooling2D((2, 2), padding='same'))

# Segunda capa convolucional.
cifar_model.add(Conv2D(
    64,
    (3, 3),
    activation='linear',
    padding='same'
))
cifar_model.add(LeakyReLU(negative_slope=0.1))
cifar_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Tercera capa convolucional.
cifar_model.add(Conv2D(
    128,
    (3, 3),
    activation='linear',
    padding='same'
))
cifar_model.add(LeakyReLU(negative_slope=0.1))
cifar_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Convertir los mapas de características en un vector.
cifar_model.add(Flatten())

# Capa densa para aprender combinaciones de características.
cifar_model.add(Dense(128, activation='linear'))
cifar_model.add(LeakyReLU(negative_slope=0.1))

# SE CONSERVA:
# La capa de salida tiene 10 neuronas porque existen 10 clases.
# Softmax produce una probabilidad para cada clase.
cifar_model.add(Dense(num_classes, activation='softmax'))


# ============================================================
# 11. COMPILAR EL MODELO
# ============================================================

# SE CONSERVA:
# categorical_crossentropy calcula el error de clasificación.
# Adam ajusta los pesos de la red neuronal.
# accuracy mide la proporción de predicciones correctas.
cifar_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Mostrar el resumen de la arquitectura.
cifar_model.summary()


# ============================================================
# 12. ENTRENAR EL MODELO
# ============================================================

# SE CONSERVA:
# El modelo aprende utilizando las imágenes de entrenamiento.
# Después de cada época se evalúa con los datos de validación.
cifar_train = cifar_model.fit(
    train_X,
    train_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_label)
)


# ============================================================
# 13. EVALUAR EL MODELO CON LOS DATOS DE PRUEBA
# ============================================================

# SE CONSERVA:
# Los datos de prueba permiten medir el rendimiento
# con imágenes que no se utilizaron para entrenar.
test_eval = cifar_model.evaluate(
    test_X,
    test_Y_one_hot,
    verbose=0
)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


# ============================================================
# 14. GRAFICAR LA EXACTITUD Y LA PÉRDIDA
# ============================================================

# SE CONSERVA:
# Obtener los valores registrados durante el entrenamiento.
accuracy = cifar_train.history['accuracy']
val_accuracy = cifar_train.history['val_accuracy']
loss = cifar_train.history['loss']
val_loss = cifar_train.history['val_loss']

epochs_range = range(1, len(accuracy) + 1)

# Graficar la exactitud de entrenamiento y validación.
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, accuracy, 'bo-', label='Training accuracy')
plt.plot(epochs_range, val_accuracy, 'b-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.legend()
plt.show()

# Graficar la pérdida de entrenamiento y validación.
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, loss, 'bo-', label='Training loss')
plt.plot(epochs_range, val_loss, 'b-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()


# ============================================================
# 15. REALIZAR PREDICCIONES
# ============================================================

# SE CONSERVA:
# predict() devuelve las probabilidades para cada una de las 10 clases.
predicted_classes = cifar_model.predict(test_X)

# CAMBIO 6:
# Antes se utilizaba np.round() antes de argmax().
# Ahora seleccionamos directamente la clase con mayor probabilidad.
predicted_classes = np.argmax(predicted_classes, axis=1)

print("Forma de las predicciones:", predicted_classes.shape)
print("Forma de las etiquetas reales:", test_Y.shape)


# ============================================================
# 16. MOSTRAR PREDICCIONES CORRECTAS
# ============================================================

correct = np.where(predicted_classes == test_Y)[0]

print('Found %d correct labels' % len(correct))

# CAMBIO 7:
# Antes se hacía reshape(28, 28) y se usaba cmap='gray'.
# Ahora mostramos las imágenes RGB originales.
plt.figure(figsize=(10, 10))

for i, index in enumerate(correct[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[index])
    plt.title(
        "Predicted {}, Class {}".format(
            predicted_classes[index],
            test_Y[index]
        )
    )
    plt.axis('off')

plt.tight_layout()
plt.show()


# ============================================================
# 17. MOSTRAR PREDICCIONES INCORRECTAS
# ============================================================

incorrect = np.where(predicted_classes != test_Y)[0]

print('Found %d incorrect labels' % len(incorrect))

# CAMBIO 8:
# Se conserva la lógica del código original,
# pero ahora se muestran las imágenes a color.
plt.figure(figsize=(10, 10))

for i, index in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[index])
    plt.title(
        "Predicted {}, Class {}".format(
            predicted_classes[index],
            test_Y[index]
        )
    )
    plt.axis('off')

plt.tight_layout()
plt.show()
