import copy
import os
import time

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------
RUTA_DATOS = "Practica_5/spambase/spambase.data"
CARPETA_GRAFICAS = "Practica_5/graficas2"
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

# Entrenar por ejemplo es lento (3680 actualizaciones por época).
# Pon True para una prueba rápida del código; False para los experimentos reales.
MODO_RAPIDO = False

if MODO_RAPIDO:
    LISTA_EPOCAS = [50, 100, 200]
else:
    LISTA_EPOCAS = [500, 1000, 2000]          # lo que pide la práctica

TASAS = [0.001, 0.01, 0.05]                   # lo que pide la práctica
# Arquitecturas a comparar: [entradas, capas ocultas..., salida]
ARQUITECTURAS = [[57, 5, 1], [57, 10, 1], [57, 20, 1],      # una capa oculta
                 [57, 10, 5, 1], [57, 20, 10, 1]]            # dos capas ocultas
LR_ARQUITECTURA = 0.05                        # tasa usada en el experimento de arquitectura

lineas_reporte = []                           # aquí guardamos lo que se imprime, para el reporte


def imprimir(texto=""):
    """Imprime en pantalla y lo guarda para resultados.txt"""
    print(texto)
    lineas_reporte.append(texto)


# ------------------------------------------------------------
# PASO 1: Cargar y preparar los datos
# ------------------------------------------------------------
datos = np.loadtxt(RUTA_DATOS, delimiter=",")

X = datos[:, :57]                      # 57 características (columnas 0-56)
y = datos[:, 57].reshape(-1, 1)        # etiqueta: 1 = SPAM, 0 = NO-SPAM

imprimir(f"Forma de X: {X.shape}")
imprimir(f"Forma de y: {y.shape}")
imprimir(f"Porcentaje de SPAM: {y.mean() * 100:.2f} %")

# Mezclar (semilla fija para repetir resultados)
np.random.seed(42)
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# 80% entrenamiento, 20% prueba
corte = int(0.8 * len(X))
X_train, X_test = X[:corte], X[corte:]
y_train, y_test = y[:corte], y[corte:]

# Normalizar con media y desviación del ENTRENAMIENTO
media = X_train.mean(axis=0)
desv = X_train.std(axis=0)
desv[desv == 0] = 1
X_train = (X_train - media) / desv
X_test = (X_test - media) / desv

imprimir(f"Entrenamiento: {X_train.shape}")
imprimir(f"Prueba: {X_test.shape}")


# ------------------------------------------------------------
# PASO 2: Implementar el MLP (57 entradas -> capas ocultas -> 1 salida)
# ------------------------------------------------------------
def sigmoide(z):
    return 1 / (1 + np.exp(-z))


class MLP:
    def __init__(self, capas, tasa_aprendizaje):
        """
        capas = lista con el tamaño de cada capa, por ejemplo:
            [57, 5, 1]      -> una capa oculta de 5 neuronas
            [57, 10, 5, 1]  -> dos capas ocultas (10 y 5 neuronas)
        """
        self.lr = tasa_aprendizaje                                   # alfa
        # Inicialización aleatoria de W y b (una pareja por cada capa)
        self.W = [np.random.randn(capas[i], capas[i + 1]) * 0.1 for i in range(len(capas) - 1)]
        self.b = [np.zeros((1, capas[i + 1])) for i in range(len(capas) - 1)]

    def forward(self, X):
        """1. Propagar hacia adelante: a^m = f^m(W^m a^(m-1) + b^m)"""
        self.a = [X]                                    # a[0] = entrada
        for W, b in zip(self.W, self.b):
            self.a.append(sigmoide(self.a[-1] @ W + b))
        return self.a[-1]

    def backward(self, X, t):
        """
        2. Propagar hacia atrás y 3. Actualizar (para UN ejemplo).
        Se debe llamar justo después de forward(X).
        """
        salida = self.a[-1]
        # e = t - a^M
        e = t - salida
        # s^M = -2 * F'^M(n^M) * e        (F' de la sigmoide = a(1-a))
        s = -2 * (salida * (1 - salida)) * e

        # Se recorren las capas de la última a la primera
        for m in range(len(self.W) - 1, -1, -1):
            if m > 0:
                # s^(m-1) = F'^(m-1)(n^(m-1)) * (W^m)^T * s^m   (con W antes de actualizarla)
                s_anterior = (self.a[m] * (1 - self.a[m])) * (s @ self.W[m].T)
            # W^m = W^m - alfa * s^m * (a^(m-1))^T      b^m = b^m - alfa * s^m
            # (en NumPy cada fila es un ejemplo, por eso las matrices van transpuestas)
            self.W[m] -= self.lr * (self.a[m].T @ s)
            self.b[m] -= self.lr * s
            if m > 0:
                s = s_anterior

    def predecir(self, X):
        """1 (SPAM) si la salida >= 0.5, si no 0"""
        return (self.forward(X) >= 0.5).astype(int)


# ------------------------------------------------------------
# PASO 3: Entrenamiento (algoritmo del pseudocódigo)
# ------------------------------------------------------------
def mse(y_real, y_pred):
    """Error cuadrático medio"""
    return np.mean((y_real - y_pred) ** 2)


def entrenar(red, X, y, epocas, checkpoints=(), mostrar=True):
    """
    Desde época = 1 a N_épocas:
        Desde q = 1 a Q: forward -> backward -> actualizar
    Devuelve la lista de errores (MSE de todo el conjunto al final de cada época)
    y copias de la red en las épocas indicadas en `checkpoints`.
    """
    errores = []
    copias = {}
    for epoca in range(1, epocas + 1):
        for q in range(len(X)):                  # ciclo por ejemplo
            xq = X[q:q + 1]                      # un solo ejemplo, forma (1, 57)
            tq = y[q:q + 1]
            red.forward(xq)                      # 1. hacia adelante
            red.backward(xq, tq)                 # 2 y 3. hacia atrás + actualizar

        errores.append(mse(y, red.forward(X)))   # error de la época (solo para graficar)

        if epoca in checkpoints:
            copias[epoca] = copy.deepcopy(red)
        if mostrar and epoca % 100 == 0:
            print(f"   Época {epoca}, error: {errores[-1]:.4f}")
    return errores, copias


def exactitud(red, X, y):
    return np.mean(red.predecir(X) == y)


# ------------------------------------------------------------
# PASO 4a: Experimento para decidir la ARQUITECTURA (neuronas ocultas)
# ------------------------------------------------------------
imprimir("\n=== Experimento de arquitectura (capas y neuronas ocultas) ===")
epocas_arq = LISTA_EPOCAS[0]
imprimir(f"lr={LR_ARQUITECTURA}, épocas={epocas_arq}")


def nombre_arq(capas):
    return "-".join(str(n) for n in capas)          # por ejemplo "57-10-5-1"


resultados_arq = []
plt.figure()
for capas in ARQUITECTURAS:
    t0 = time.time()
    np.random.seed(0)
    r = MLP(capas, LR_ARQUITECTURA)
    errs, _ = entrenar(r, X_train, y_train, epocas_arq, mostrar=False)
    a_tr, a_te = exactitud(r, X_train, y_train), exactitud(r, X_test, y_test)
    resultados_arq.append((capas, errs[-1], a_tr, a_te))
    plt.plot(errs, label=nombre_arq(capas))
    imprimir(f"  arq={nombre_arq(capas):>10}  error={errs[-1]:.4f}  acc_train={a_tr:.4f}  "
             f"acc_test={a_te:.4f}  ({time.time() - t0:.0f} s)")

plt.xlabel("Época")
plt.ylabel("Error (MSE)")
plt.title(f"Arquitectura: error por épocas (lr={LR_ARQUITECTURA})")
plt.legend()
plt.grid(True)
plt.savefig(f"{CARPETA_GRAFICAS}/paso4_arquitectura.png", dpi=150, bbox_inches="tight")

ARQ_ELEGIDA = max(resultados_arq, key=lambda r: r[3])[0]     # la de mayor exactitud en prueba
imprimir(f"Arquitectura elegida: {nombre_arq(ARQ_ELEGIDA)}  (sigmoide, MSE)")


# ------------------------------------------------------------
# PASO 4b: Experimentos con tasas de aprendizaje y épocas
# ------------------------------------------------------------
# Truco para ahorrar tiempo: se entrena UNA vez cada tasa hasta el máximo de épocas
# y se guardan copias de la red en 500, 1000 y 2000. Es equivalente a entrenar
# por separado, porque con la misma semilla la trayectoria es idéntica.
imprimir("\n=== Experimentos: tasa de aprendizaje y épocas ===")
epocas_max = max(LISTA_EPOCAS)

resultados = []      # (lr, épocas, error, acc_train, acc_test)
curvas = {}          # lr -> lista de errores por época
redes = {}           # (lr, épocas) -> red entrenada

for lr in TASAS:
    t0 = time.time()
    print(f"Entrenando lr={lr} ...")
    np.random.seed(0)
    r = MLP(ARQ_ELEGIDA, lr)
    errs, copias = entrenar(r, X_train, y_train, epocas_max, checkpoints=LISTA_EPOCAS)
    curvas[lr] = errs
    for ep in LISTA_EPOCAS:
        red_ep = copias[ep]
        redes[(lr, ep)] = red_ep
        resultados.append((lr, ep, errs[ep - 1],
                           exactitud(red_ep, X_train, y_train),
                           exactitud(red_ep, X_test, y_test)))
    print(f"   listo en {time.time() - t0:.0f} s")

imprimir(f"\n{'lr':>7} {'épocas':>7} {'error':>8} {'acc_train':>10} {'acc_test':>9}")
for lr, ep, err, at, ae in resultados:
    imprimir(f"{lr:>7} {ep:>7} {err:>8.4f} {at:>10.4f} {ae:>9.4f}")

# Una gráfica por cantidad de épocas, con las curvas de todas las tasas
for ep in LISTA_EPOCAS:
    plt.figure()
    for lr in TASAS:
        plt.plot(curvas[lr][:ep], label=f"lr={lr}")
    plt.xlabel("Época")
    plt.ylabel("Error (MSE)")
    plt.title(f"Error de entrenamiento por épocas ({ep} épocas)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{CARPETA_GRAFICAS}/paso4_error_{ep}_epocas.png", dpi=150, bbox_inches="tight")


# ------------------------------------------------------------
# PASO 5: Métricas y matriz de confusión (mejor configuración)
# ------------------------------------------------------------
mejor = max(resultados, key=lambda r: r[4])       # mayor exactitud en prueba
lr_mejor, ep_mejor = mejor[0], mejor[1]
mejor_red = redes[(lr_mejor, ep_mejor)]
imprimir(f"\nMejor configuración: lr={lr_mejor}, épocas={ep_mejor}, "
         f"arquitectura={nombre_arq(ARQ_ELEGIDA)}  (exactitud prueba={mejor[4]:.4f})")

pred = mejor_red.predecir(X_test)

# Matriz de confusión a mano
VP = np.sum((pred == 1) & (y_test == 1))   # spam detectado como spam
VN = np.sum((pred == 0) & (y_test == 0))   # no-spam detectado como no-spam
FP = np.sum((pred == 1) & (y_test == 0))   # no-spam clasificado como spam
FN = np.sum((pred == 0) & (y_test == 1))   # spam que se coló como no-spam
matriz = np.array([[VN, FP],
                   [FN, VP]])

exact = (VP + VN) / (VP + VN + FP + FN)
precision = VP / (VP + FP)
recuerdo = VP / (VP + FN)
f1 = 2 * precision * recuerdo / (precision + recuerdo)

imprimir("Matriz de confusión (filas = real, columnas = predicho):")
imprimir(str(matriz))
imprimir(f"VP={VP}  VN={VN}  FP={FP}  FN={FN}")
imprimir(f"Exactitud: {exact:.4f}")
imprimir(f"Precisión: {precision:.4f}")
imprimir(f"Recuerdo:  {recuerdo:.4f}")
imprimir(f"Medida F:  {f1:.4f}")

# Gráfica de la matriz de confusión
plt.figure()
plt.imshow(matriz, cmap="Blues")
plt.title(f"Matriz de confusión (lr={lr_mejor}, {ep_mejor} épocas)")
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
plt.savefig(f"{CARPETA_GRAFICAS}/paso5_matriz_confusion.png", dpi=150, bbox_inches="tight")

# Error de entrenamiento por épocas de la mejor red
plt.figure()
plt.plot(curvas[lr_mejor][:ep_mejor])
plt.xlabel("Época")
plt.ylabel("Error (MSE)")
plt.title(f"Error de entrenamiento por épocas (mejor red: lr={lr_mejor})")
plt.grid(True)
plt.savefig(f"{CARPETA_GRAFICAS}/paso5_error_mejor_red.png", dpi=150, bbox_inches="tight")

# Guardar todos los resultados en un archivo de texto (útil para el reporte)
with open(f"{CARPETA_GRAFICAS}/resultados.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lineas_reporte))

plt.show()