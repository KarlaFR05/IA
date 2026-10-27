import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# PASO 1: Cargar y preparar los datos
# ------------------------------------------------------------

# 1. Cargar el dataset (separado por comas, sin encabezados)
datos = np.loadtxt("IA/Practica_5/spambase/spambase.data", delimiter=",")

# 2. Separar entradas (X) y salida (y)
X = datos[:, :57]          # 57 características
y = datos[:, 57].reshape(-1, 1)   # etiqueta: 1 = SPAM, 0 = NO-SPAM

print("Forma de X:", X.shape)
print("Forma de y:", y.shape)
print("Porcentaje de SPAM:", y.mean() * 100, "%")

# 3. Mezclar los datos de forma aleatoria (semilla fija para repetir resultados)
np.random.seed(42)
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# 4. Dividir en entrenamiento (80%) y prueba (20%)
corte = int(0.8 * len(X))
X_train, X_test = X[:corte], X[corte:]
y_train, y_test = y[:corte], y[corte:]

# 5. Normalizar: usamos la media y desviación del conjunto de ENTRENAMIENTO
media = X_train.mean(axis=0)
desv = X_train.std(axis=0)
desv[desv == 0] = 1   # evitar dividir entre cero

X_train = (X_train - media) / desv
X_test = (X_test - media) / desv

print("Entrenamiento:", X_train.shape)
print("Prueba:", X_test.shape)


