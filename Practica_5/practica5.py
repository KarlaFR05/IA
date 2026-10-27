import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("imagenes", exist_ok=True)
# ------------------------------------------------------------
# PASO 1: Cargar y preparar los datos
# ------------------------------------------------------------

# 1. Cargar el dataset (separado por comas, sin encabezados)
datos = np.loadtxt("Practica_5/spambase/spambase.data", delimiter=",")

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




# ------------------------------------------------------------
# PASO 2: Implementar el MLP
# ------------------------------------------------------------

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

class MLP:
    def __init__(self, n_entrada, n_oculta, n_salida, tasa_aprendizaje):
        self.lr = tasa_aprendizaje
        # Pesos pequeños aleatorios y sesgos en cero
        self.W1 = np.random.randn(n_entrada, n_oculta) * 0.1
        self.b1 = np.zeros((1, n_oculta))
        self.W2 = np.random.randn(n_oculta, n_salida) * 0.1
        self.b2 = np.zeros((1, n_salida))

    def forward(self, X):
        """Propagación hacia adelante"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoide(self.z1)          # salida capa oculta
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoide(self.z2)          # salida final
        return self.a2

    def backward(self, X, y):
        """Retropropagación + actualización de pesos"""
        m = X.shape[0]   # número de ejemplos

        # Error en la salida (derivada del MSE * derivada de la sigmoide)
        delta2 = (self.a2 - y) * self.a2 * (1 - self.a2)

        # Error propagado a la capa oculta
        delta1 = (delta2 @ self.W2.T) * self.a1 * (1 - self.a1)

        # Gradientes (promediados sobre los ejemplos)
        dW2 = self.a1.T @ delta2 / m
        db2 = delta2.sum(axis=0, keepdims=True) / m
        dW1 = X.T @ delta1 / m
        db1 = delta1.sum(axis=0, keepdims=True) / m

        # Descenso de gradiente
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predecir(self, X):
        """Devuelve 1 (SPAM) si la salida >= 0.5, si no 0"""
        return (self.forward(X) >= 0.5).astype(int)

# Prueba rápida: ¿funciona el forward?
red = MLP(n_entrada=57, n_oculta=10, n_salida=1, tasa_aprendizaje=0.05)
salida = red.forward(X_train)
print("Forma de la salida:", salida.shape)
print("Primeras 5 salidas:", salida[:5].ravel())



# ------------------------------------------------------------
# PASO 3: Entrenar y graficar el error por épocas
# ------------------------------------------------------------

def mse(y_real, y_pred):
    """Error cuadrático medio"""
    return np.mean((y_real - y_pred) ** 2)

def entrenar(red, X, y, epocas):
    """Entrena la red y devuelve la lista de errores por época"""
    errores = []
    for epoca in range(epocas):
        salida = red.forward(X)          # 1. propagación hacia adelante
        error = mse(y, salida)           # 2. calcular el error
        errores.append(error)            # 3. guardarlo
        red.backward(X, y)               # 4. retropropagación y ajuste de pesos

        if epoca % 100 == 0:             # mostrar progreso cada 100 épocas
            print(f"Época {epoca}, error: {error:.4f}")
    return errores

# Crear la red y entrenarla
np.random.seed(0)
red = MLP(n_entrada=57, n_oculta=10, n_salida=1, tasa_aprendizaje=0.05)
errores = entrenar(red, X_train, y_train, epocas=1000)

# Graficar el error por épocas
plt.figure()
plt.plot(errores)
plt.xlabel("Época")
plt.ylabel("Error (MSE)")
plt.title("Error de entrenamiento por épocas (lr=0.05)")
plt.grid(True)
plt.savefig("Practica_5/graficas/paso3_error_lr0.05.png", dpi=150, bbox_inches="tight")   # guardar ANTES de show
plt.show()

# Precisión rápida (accuracy) para ver si aprendió
pred_train = red.predecir(X_train)
pred_test = red.predecir(X_test)
print("Exactitud en entrenamiento:", np.mean(pred_train == y_train))
print("Exactitud en prueba:", np.mean(pred_test == y_test))




