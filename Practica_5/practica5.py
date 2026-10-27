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




# ------------------------------------------------------------
# PASO 4: Experimentos con tasas de aprendizaje y épocas
# ------------------------------------------------------------

tasas = [0.001, 0.01, 0.05, 0.5, 1.0]
lista_epocas = [500, 1000, 2000]

resultados = []          # (lr, épocas, error final, exactitud train, exactitud test)
curvas = {}              # curva de error de cada experimento

for lr in tasas:
    for ep in lista_epocas:
        np.random.seed(0)    # misma inicialización para comparar justo
        red_exp = MLP(57, 10, 1, lr)

        errs = []
        for _ in range(ep):
            salida = red_exp.forward(X_train)
            errs.append(mse(y_train, salida))
            red_exp.backward(X_train, y_train)

        acc_train = np.mean(red_exp.predecir(X_train) == y_train)
        acc_test = np.mean(red_exp.predecir(X_test) == y_test)
        resultados.append((lr, ep, errs[-1], acc_train, acc_test))
        curvas[(lr, ep)] = errs

# Tabla de resultados
print(f"{'lr':>7} {'épocas':>7} {'error':>8} {'acc_train':>10} {'acc_test':>9}")
for lr, ep, err, at, ae in resultados:
    print(f"{lr:>7} {ep:>7} {err:>8.4f} {at:>10.4f} {ae:>9.4f}")

# Una gráfica (y una imagen) por cada cantidad de épocas
for ep in lista_epocas:
    plt.figure()
    for lr in tasas:
        plt.plot(curvas[(lr, ep)], label=f"lr={lr}")
    plt.xlabel("Época")
    plt.ylabel("Error (MSE)")
    plt.title(f"Error de entrenamiento por épocas ({ep} épocas)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Practica_5/graficas/paso4_error_{ep}_epocas.png", dpi=150, bbox_inches="tight")

plt.show()   # muestra las 3 ventanas al final



# ------------------------------------------------------------
# PASO 5: Métricas y matriz de confusión (mejor configuración)
# ------------------------------------------------------------

# 1. Entrenar la mejor red
np.random.seed(0)
mejor_red = MLP(57, 10, 1, 1.0)
errores_mejor = entrenar(mejor_red, X_train, y_train, epocas=2000)

# 2. Predicciones sobre el conjunto de PRUEBA
pred = mejor_red.predecir(X_test)

# 3. Matriz de confusión calculada a mano
VP = np.sum((pred == 1) & (y_test == 1))   # spam detectado como spam
VN = np.sum((pred == 0) & (y_test == 0))   # no-spam detectado como no-spam
FP = np.sum((pred == 1) & (y_test == 0))   # no-spam clasificado como spam
FN = np.sum((pred == 0) & (y_test == 1))   # spam que se coló como no-spam

matriz = np.array([[VN, FP],
                   [FN, VP]])

# 4. Métricas
exactitud = (VP + VN) / (VP + VN + FP + FN)
precision = VP / (VP + FP)
recuerdo = VP / (VP + FN)
f1 = 2 * precision * recuerdo / (precision + recuerdo)

print("\nMatriz de confusión (filas = real, columnas = predicho):")
print(matriz)
print(f"VP={VP}  VN={VN}  FP={FP}  FN={FN}")
print(f"Exactitud: {exactitud:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"Recuerdo:  {recuerdo:.4f}")
print(f"Medida F:  {f1:.4f}")

# 5. Graficar la matriz de confusión
plt.figure()
plt.imshow(matriz, cmap="Blues")
plt.title("Matriz de confusión (lr=1.0, 2000 épocas)")
plt.colorbar()
plt.xticks([0, 1], ["NO-SPAM", "SPAM"])
plt.yticks([0, 1], ["NO-SPAM", "SPAM"])
plt.xlabel("Predicho")
plt.ylabel("Real")
for i in range(2):
    for j in range(2):
        plt.text(j, i, matriz[i, j], ha="center", va="center",
                 color="white" if matriz[i, j] > matriz.max() / 2 else "black",
                 fontsize=14)
plt.savefig("Practica_5/graficas/paso5_matriz_confusion.png", dpi=150, bbox_inches="tight")

# 6. Gráfica del error por épocas de la mejor red
plt.figure()
plt.plot(errores_mejor)
plt.xlabel("Época")
plt.ylabel("Error (MSE)")
plt.title("Error de entrenamiento por épocas (mejor red: lr=1.0)")
plt.grid(True)
plt.savefig("Practica_5/graficas/paso5_error_mejor_red.png", dpi=150, bbox_inches="tight")

plt.show()