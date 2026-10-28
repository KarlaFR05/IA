# copy permite guardar una copia completa de la red en ciertos puntos del entrenamiento.
import copy
# os permite crear carpetas y trabajar con rutas de archivos.
import os
# time sirve para medir cuánto tarda cada experimento.
import time

# NumPy se usa para trabajar con matrices y hacer los cálculos de la red neuronal.
import numpy as np
# Matplotlib se usa para dibujar y guardar las gráficas de resultados.
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------
# Ruta del archivo que contiene los datos de correos y sus características.
RUTA_DATOS = "Practica_5/spambase/spambase.data"
# Carpeta donde se guardarán las gráficas y el archivo de resultados.
CARPETA_GRAFICAS = "Practica_5/graficas"
# Crea la carpeta si todavía no existe; exist_ok=True evita un error si ya existe.
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

# Entrenar por ejemplo es lento (3680 actualizaciones por época).
# Pon True para una prueba rápida del código; False para los experimentos reales.
# False ejecuta los experimentos completos que pide la práctica.
MODO_RAPIDO = False

# Si el modo rápido está activado, se usan menos épocas para probar el programa.
if MODO_RAPIDO:
    # Lista corta de épocas para una ejecución de prueba.
    LISTA_EPOCAS = [50, 100, 200]
# Si no se usa el modo rápido, se realizan los experimentos completos.
else:
    # Estas son las cantidades de épocas solicitadas en la práctica.
    LISTA_EPOCAS = [500, 1000, 2000]          # lo que pide la práctica

# Tasas de aprendizaje que se compararán en los experimentos.
TASAS = [0.001, 0.01, 0.05]                   # lo que pide la práctica
# Cantidades de neuronas que se probarán en la capa oculta.
NEURONAS_OCULTAS = [5, 10, 20]                # experimento para decidir la arquitectura
# Tasa de aprendizaje fija para comparar las arquitecturas.
LR_ARQUITECTURA = 0.05                        # tasa usada en el experimento de arquitectura

# Lista donde se guardan los mensajes para escribirlos después en un archivo.
lineas_reporte = []                           # aquí guardamos lo que se imprime, para el reporte


# Función auxiliar: muestra mensajes en pantalla y los guarda para el reporte.
def imprimir(texto=""):
    """Imprime en pantalla y lo guarda para resultados.txt"""
    # Muestra el mensaje en la terminal.
    print(texto)
    # Agrega el mensaje a la lista que se escribirá en resultados.txt.
    lineas_reporte.append(texto)


# ------------------------------------------------------------
# PASO 1 del pseudocódigo: cargar y preparar los ejemplos de entrenamiento.
# PASO 1: Cargar y preparar los datos
# ------------------------------------------------------------
# Lee el archivo separado por comas y convierte sus valores en una matriz numérica.
datos = np.loadtxt(RUTA_DATOS, delimiter=",")

# X contiene las primeras 57 columnas, que son las características de cada correo.
X = datos[:, :57]                      # 57 características (columnas 0-56)
# y contiene la etiqueta de cada correo: 1 significa SPAM y 0 significa NO-SPAM.
y = datos[:, 57].reshape(-1, 1)        # etiqueta: 1 = SPAM, 0 = NO-SPAM

# Muestra cuántas filas y columnas tiene la matriz de características.
imprimir(f"Forma de X: {X.shape}")
# Muestra la forma de las etiquetas; normalmente hay una etiqueta por correo.
imprimir(f"Forma de y: {y.shape}")
# Calcula qué porcentaje de los correos cargados están etiquetados como SPAM.
imprimir(f"Porcentaje de SPAM: {y.mean() * 100:.2f} %")

# Mezclar (semilla fija para repetir resultados)
# Fija la semilla para que la mezcla aleatoria pueda repetirse en otra ejecución.
np.random.seed(42)
# Genera un orden aleatorio de las posiciones de los correos.
indices = np.random.permutation(len(X))
# Reordena las características con ese orden.
X = X[indices]
# Reordena las etiquetas con exactamente el mismo orden para no separarlas de sus correos.
y = y[indices]

# 80% entrenamiento, 20% prueba
# Calcula el punto donde se separará el 80 % de los datos para entrenamiento.
corte = int(0.8 * len(X))
# Guarda el 80 % inicial en entrenamiento y el 20 % restante en prueba.
X_train, X_test = X[:corte], X[corte:]
# Separa las etiquetas usando el mismo corte que las características.
y_train, y_test = y[:corte], y[corte:]

# Normalizar con media y desviación del ENTRENAMIENTO
# Calcula el promedio de cada característica usando solo los datos de entrenamiento.
media = X_train.mean(axis=0)
# Calcula la desviación estándar de cada característica de entrenamiento.
desv = X_train.std(axis=0)
# Evita dividir entre cero si alguna característica tiene desviación igual a cero.
desv[desv == 0] = 1
# Normaliza los datos de entrenamiento usando promedio y desviación estándar.
X_train = (X_train - media) / desv
# Normaliza los datos de prueba con los valores calculados en entrenamiento.
X_test = (X_test - media) / desv

# Informa el tamaño del conjunto que se usará para aprender.
imprimir(f"Entrenamiento: {X_train.shape}")
# Informa el tamaño del conjunto reservado para comprobar el rendimiento.
imprimir(f"Prueba: {X_test.shape}")


# ------------------------------------------------------------
# PASO 2 del pseudocódigo: construir la red multicapa con entrada, capa oculta y salida.
# PASO 2: Implementar el MLP (57 entradas -> capa oculta -> 1 salida)
# ------------------------------------------------------------
# Define la función de activación sigmoide, que transforma valores a un rango entre 0 y 1.
def sigmoide(z):
    # Aplica la fórmula de la sigmoide a cada valor de z.
    return 1 / (1 + np.exp(-z))


# Define la clase que representa la red neuronal MLP.
class MLP:
    # El constructor recibe el número de entradas, neuronas ocultas, salidas y la tasa de aprendizaje.
    def __init__(self, n_entrada, n_oculta, n_salida, tasa_aprendizaje):
        # Guarda alfa, que indica qué tan grandes serán los cambios durante el aprendizaje.
        self.lr = tasa_aprendizaje                                   # alfa
        # Se inicializan al azar los pesos W y en cero los sesgos b.
        # Inicialización aleatoria de W y b
        # W1 conecta las 57 entradas con las neuronas de la capa oculta.
        self.W1 = np.random.randn(n_entrada, n_oculta) * 0.1
        # b1 es el sesgo de cada neurona oculta; empieza en cero.
        self.b1 = np.zeros((1, n_oculta))
        # W2 conecta las neuronas ocultas con la neurona de salida.
        self.W2 = np.random.randn(n_oculta, n_salida) * 0.1
        # b2 es el sesgo de la capa de salida; empieza en cero.
        self.b2 = np.zeros((1, n_salida))

    # PASO 1 del pseudocódigo: propagación hacia adelante por las capas de la red.
    def forward(self, X):
        """1. Propagar hacia adelante: a^m = f^m(W^m a^(m-1) + b^m)"""
        # Calcula las activaciones de la capa oculta.
        self.a1 = sigmoide(X @ self.W1 + self.b1)          # capa oculta
        # Calcula la salida final de la red, que representa la probabilidad estimada de SPAM.
        self.a2 = sigmoide(self.a1 @ self.W2 + self.b2)    # capa de salida
        # Devuelve la salida de la red.
        return self.a2

    # PASOS 2 y 3 del pseudocódigo: propagación hacia atrás y actualización de parámetros.
    def backward(self, X, t):
        """
        2. Propagar hacia atrás y 3. Actualizar (para UN ejemplo).
        Se debe llamar justo después de forward(X).
        """
        # El error es la etiqueta correcta menos la salida que produjo la red.
        # e = t - a^M
        # Calcula la diferencia entre el valor esperado t y la predicción a2.
        e = t - self.a2
        # Sensibilidad de salida: combina el error con la derivada de la sigmoide.
        # s^M = -2 * F'^M(n^M) * e        (F' de la sigmoide = a(1-a))
        # Calcula cuánto contribuye el error a la actualización de la neurona de salida.
        s2 = -2 * (self.a2 * (1 - self.a2)) * e
        # La sensibilidad de la capa anterior depende de la derivada y de los pesos de salida.
        # s^(m-1) = F'^(m-1)(n^(m-1)) * (W^m)^T * s^m
        # Calcula la sensibilidad de las neuronas ocultas.
        s1 = (self.a1 * (1 - self.a1)) * (s2 @ self.W2.T)

        # PASO 3 del pseudocódigo: actualizar los pesos y sesgos con descenso de gradiente.
        # W^m = W^m - alfa * s^m * (a^(m-1))^T      b^m = b^m - alfa * s^m
        # Las transpuestas permiten que las dimensiones de las matrices coincidan en NumPy.
        # (en NumPy cada fila es un ejemplo, por eso las matrices van transpuestas)
        # Actualiza los pesos que conectan la capa oculta con la salida.
        self.W2 -= self.lr * (self.a1.T @ s2)
        # Actualiza el sesgo de la neurona de salida.
        self.b2 -= self.lr * s2
        # Actualiza los pesos que conectan las entradas con la capa oculta.
        self.W1 -= self.lr * (X.T @ s1)
        # Actualiza los sesgos de las neuronas ocultas.
        self.b1 -= self.lr * s1

    # Convierte la salida numérica en una clase SPAM o NO-SPAM.
    def predecir(self, X):
        """1 (SPAM) si la salida >= 0.5, si no 0"""
        # Devuelve las predicciones como enteros 0 o 1.
        return (self.forward(X) >= 0.5).astype(int)


# ------------------------------------------------------------
# PASO 3 del pseudocódigo: repetir el aprendizaje por épocas y por cada ejemplo.
# PASO 3: Entrenamiento (algoritmo del pseudocódigo)
# ------------------------------------------------------------
# Calcula el error cuadrático medio, usado para observar el aprendizaje.
def mse(y_real, y_pred):
    """Error cuadrático medio"""
    # Calcula y devuelve el promedio de los errores al cuadrado.
    return np.mean((y_real - y_pred) ** 2)


def entrenar(red, X, y, epocas, checkpoints=(), mostrar=True):
    # Repite el aprendizaje por épocas y por cada ejemplo del conjunto de entrenamiento.
    """
    Desde época = 1 a N_épocas:
        Desde q = 1 a Q: forward -> backward -> actualizar
    Devuelve la lista de errores (MSE de todo el conjunto al final de cada época)
    y copias de la red en las épocas indicadas en `checkpoints`.
    """
    # Lista que almacenará el error al terminar cada época.
    errores = []
    # Diccionario para guardar copias de la red en épocas solicitadas.
    copias = {}
    # Repite el entrenamiento desde la época 1 hasta el total indicado.
    for epoca in range(1, epocas + 1):
        # Recorre uno por uno todos los ejemplos del conjunto de entrenamiento.
        for q in range(len(X)):                  # ciclo por ejemplo
            # Selecciona una sola fila de características para el ejemplo q.
            xq = X[q:q + 1]                      # un solo ejemplo, forma (1, 57)
            # Selecciona la etiqueta correcta correspondiente a ese mismo ejemplo.
            tq = y[q:q + 1]
            # PASO 1: calcula la salida de la red para el ejemplo actual.
            red.forward(xq)                      # 1. hacia adelante
            # PASOS 2 y 3: calcula sensibilidades y actualiza pesos y sesgos.
            red.backward(xq, tq)                 # 2 y 3. hacia atrás + actualizar

        # Al terminar la época, calcula el MSE de todo el conjunto para observar el avance.
        errores.append(mse(y, red.forward(X)))   # error de la época (solo para graficar)

        # Comprueba si esta época está entre las que se quieren guardar.
        if epoca in checkpoints:
            # Guarda una copia independiente de la red en esta época.
            copias[epoca] = copy.deepcopy(red)
        # Muestra el progreso cada 100 épocas si se solicitó mostrarlo.
        if mostrar and epoca % 100 == 0:
            # Imprime el número de época y su error más reciente.
            print(f"   Época {epoca}, error: {errores[-1]:.4f}")
    # Devuelve los errores de cada época y las copias guardadas.
    return errores, copias


# Calcula la exactitud: proporción de ejemplos clasificados correctamente.
def exactitud(red, X, y):
    # Compara predicciones con etiquetas reales y obtiene el promedio de aciertos.
    return np.mean(red.predecir(X) == y)


# ------------------------------------------------------------
# PASO 4a: comparar arquitecturas, es decir, diferentes cantidades de neuronas ocultas.
# PASO 4a: Experimento para decidir la ARQUITECTURA (neuronas ocultas)
# ------------------------------------------------------------
# Anuncia en pantalla el inicio del experimento de arquitectura.
imprimir("\n=== Experimento de arquitectura (neuronas en la capa oculta) ===")
# Usa la primera cantidad de épocas de la lista para comparar las arquitecturas.
epocas_arq = LISTA_EPOCAS[0]
# Muestra la tasa de aprendizaje y las épocas que se utilizarán.
imprimir(f"lr={LR_ARQUITECTURA}, épocas={epocas_arq}")

# Lista donde se guardará el resultado de cada arquitectura probada.
resultados_arq = []
# Crea una figura vacía para dibujar las curvas de error.
plt.figure()
# Prueba cada cantidad de neuronas ocultas definida en la configuración.
for n_oc in NEURONAS_OCULTAS:
    # Inicia el cronómetro para medir cuánto tarda este experimento.
    t0 = time.time()
    # Fija la semilla para que la inicialización aleatoria sea comparable.
    np.random.seed(0)
    # Crea una red con 57 entradas, n_oc neuronas ocultas y una salida.
    r = MLP(57, n_oc, 1, LR_ARQUITECTURA)
    # Entrena la red y guarda los errores por época.
    errs, _ = entrenar(r, X_train, y_train, epocas_arq, mostrar=False)
    # Calcula la exactitud sobre entrenamiento y sobre prueba.
    a_tr, a_te = exactitud(r, X_train, y_train), exactitud(r, X_test, y_test)
    # Guarda cantidad de neuronas, error final y exactitudes obtenidas.
    resultados_arq.append((n_oc, errs[-1], a_tr, a_te))
    # Dibuja la curva de error de esta arquitectura y le pone una etiqueta.
    plt.plot(errs, label=f"{n_oc} neuronas")
    # Muestra el error final, la exactitud y el tiempo usado en este experimento.
    imprimir(f"  oculta={n_oc:>3}  error={errs[-1]:.4f}  acc_train={a_tr:.4f}  "
             f"acc_test={a_te:.4f}  ({time.time() - t0:.0f} s)")

# Nombra el eje horizontal: número de época.
plt.xlabel("Época")
# Nombra el eje vertical: error cuadrático medio.
plt.ylabel("Error (MSE)")
# Agrega un título que indica la tasa usada en la comparación.
plt.title(f"Arquitectura: error por épocas (lr={LR_ARQUITECTURA})")
# Muestra la leyenda con la cantidad de neuronas de cada curva.
plt.legend()
# Agrega una cuadrícula para facilitar la lectura.
plt.grid(True)
# Guarda la gráfica de comparación de arquitecturas como imagen PNG.
plt.savefig(f"{CARPETA_GRAFICAS}/paso4_arquitectura.png", dpi=150, bbox_inches="tight")

# Elige la arquitectura que obtuvo la mayor exactitud en el conjunto de prueba.
N_OCULTA = max(resultados_arq, key=lambda r: r[3])[0]     # la de mayor exactitud en prueba
# Informa la arquitectura final elegida: entradas, capa oculta y salida.
imprimir(f"Arquitectura elegida: 57 - {N_OCULTA} - 1  (sigmoide, MSE)")


# ------------------------------------------------------------
# PASO 4b: comparar las tasas de aprendizaje y las cantidades de épocas solicitadas.
# PASO 4b: Experimentos con tasas de aprendizaje y épocas
# ------------------------------------------------------------
# Para ahorrar tiempo, cada tasa se entrena una sola vez hasta la época máxima.
# Truco para ahorrar tiempo: se entrena UNA vez cada tasa hasta el máximo de épocas
# Se guardan copias de la red en las épocas de interés.
# y se guardan copias de la red en 500, 1000 y 2000. Es equivalente a entrenar
# Así se pueden comparar esos puntos del mismo entrenamiento.
# por separado, porque con la misma semilla la trayectoria es idéntica.
# Anuncia el inicio de los experimentos de tasas y épocas.
imprimir("\n=== Experimentos: tasa de aprendizaje y épocas ===")
# Encuentra la mayor cantidad de épocas que se entrenará.
epocas_max = max(LISTA_EPOCAS)

# Cada resultado guarda tasa, épocas, error y exactitud en entrenamiento y prueba.
resultados = []      # (lr, épocas, error, acc_train, acc_test)
# Diccionario que almacena las curvas de error, una por tasa de aprendizaje.
curvas = {}          # lr -> lista de errores por época
# Diccionario que guarda la red correspondiente a cada tasa y número de épocas.
redes = {}           # (lr, épocas) -> red entrenada

# Recorre todas las tasas de aprendizaje que se quieren comparar.
for lr in TASAS:
    # Inicia el cronómetro para esta tasa.
    t0 = time.time()
    # Muestra qué tasa se está entrenando.
    print(f"Entrenando lr={lr} ...")
    # Reinicia la semilla para comenzar cada experimento con pesos iniciales comparables.
    np.random.seed(0)
    # Crea la red usando la arquitectura seleccionada y la tasa actual.
    r = MLP(57, N_OCULTA, 1, lr)
    # Entrena hasta el máximo de épocas y guarda copias en las épocas indicadas.
    errs, copias = entrenar(r, X_train, y_train, epocas_max, checkpoints=LISTA_EPOCAS)
    # Guarda la curva de errores de esta tasa.
    curvas[lr] = errs
    # Recorre las cantidades de épocas que se van a evaluar.
    for ep in LISTA_EPOCAS:
        # Obtiene la copia de la red al llegar a esta cantidad de épocas.
        red_ep = copias[ep]
        # Guarda esa red para poder evaluarla después.
        redes[(lr, ep)] = red_ep
        # Agrega la tasa, épocas y error correspondiente al resultado.
        resultados.append((lr, ep, errs[ep - 1],
                           # Añade la exactitud de la red en los datos de entrenamiento.
                           exactitud(red_ep, X_train, y_train),
                           # Añade la exactitud de la red en los datos de prueba.
                           exactitud(red_ep, X_test, y_test)))
    # Muestra cuánto tardó en completarse el entrenamiento de esta tasa.
    print(f"   listo en {time.time() - t0:.0f} s")

# Imprime los títulos de las columnas de la tabla de resultados.
imprimir(f"\n{'lr':>7} {'épocas':>7} {'error':>8} {'acc_train':>10} {'acc_test':>9}")
# Recorre los resultados de todas las combinaciones de tasa y épocas.
for lr, ep, err, at, ae in resultados:
    # Imprime cada combinación con su error y exactitudes.
    imprimir(f"{lr:>7} {ep:>7} {err:>8.4f} {at:>10.4f} {ae:>9.4f}")

# Genera una gráfica para cada cantidad de épocas comparando todas las tasas.
# Una gráfica por cantidad de épocas, con las curvas de todas las tasas
# Recorre las cantidades de épocas que se evaluarán.
for ep in LISTA_EPOCAS:
    # Crea una figura nueva para esta gráfica.
    plt.figure()
    # Dibuja la curva de cada tasa de aprendizaje.
    for lr in TASAS:
        # Usa solo las primeras epocas de cada curva y etiqueta la tasa.
        plt.plot(curvas[lr][:ep], label=f"lr={lr}")
    # Etiqueta el eje horizontal.
    plt.xlabel("Época")
    # Etiqueta el eje vertical.
    plt.ylabel("Error (MSE)")
    # Coloca un título que indica cuántas épocas se muestran.
    plt.title(f"Error de entrenamiento por épocas ({ep} épocas)")
    # Muestra la leyenda de las tasas.
    plt.legend()
    # Agrega una cuadrícula.
    plt.grid(True)
    # Guarda la gráfica de errores para esta cantidad de épocas.
    plt.savefig(f"{CARPETA_GRAFICAS}/paso4_error_{ep}_epocas.png", dpi=150, bbox_inches="tight")


# ------------------------------------------------------------
# PASO 5: evaluar la mejor configuración con métricas y matriz de confusión.
# PASO 5: Métricas y matriz de confusión (mejor configuración)
# ------------------------------------------------------------
# Elige el resultado que obtuvo mayor exactitud en el conjunto de prueba.
mejor = max(resultados, key=lambda r: r[4])       # mayor exactitud en prueba
# Recupera la tasa y el número de épocas de esa mejor combinación.
lr_mejor, ep_mejor = mejor[0], mejor[1]
# Obtiene la red entrenada que corresponde a esa configuración.
mejor_red = redes[(lr_mejor, ep_mejor)]
# Muestra la configuración elegida y su exactitud en prueba.
imprimir(f"\nMejor configuración: lr={lr_mejor}, épocas={ep_mejor}, "
         f"oculta={N_OCULTA}  (exactitud prueba={mejor[4]:.4f})")

# Predice las etiquetas del conjunto de prueba usando la mejor red.
pred = mejor_red.predecir(X_test)

# Construye la matriz de confusión contando los cuatro tipos de resultado.
# Matriz de confusión a mano
# Verdadero positivo: un correo SPAM correctamente identificado como SPAM.
VP = np.sum((pred == 1) & (y_test == 1))   # spam detectado como spam
# Verdadero negativo: un correo NO-SPAM correctamente identificado como NO-SPAM.
VN = np.sum((pred == 0) & (y_test == 0))   # no-spam detectado como no-spam
# Falso positivo: un correo NO-SPAM clasificado por error como SPAM.
FP = np.sum((pred == 1) & (y_test == 0))   # no-spam clasificado como spam
# Falso negativo: un correo SPAM clasificado por error como NO-SPAM.
FN = np.sum((pred == 0) & (y_test == 1))   # spam que se coló como no-spam
# Organiza los conteos en una matriz: filas reales y columnas predichas.
matriz = np.array([[VN, FP],
                   [FN, VP]])

# Exactitud: proporción total de correos clasificados correctamente.
exact = (VP + VN) / (VP + VN + FP + FN)
# Precisión: de los correos predichos como SPAM, qué proporción sí era SPAM.
precision = VP / (VP + FP)
# Recuerdo: de los correos que realmente eran SPAM, qué proporción se detectó.
recuerdo = VP / (VP + FN)
# Medida F1: combina precisión y recuerdo en una sola medida.
f1 = 2 * precision * recuerdo / (precision + recuerdo)

# Imprime una explicación de cómo leer filas y columnas de la matriz.
imprimir("Matriz de confusión (filas = real, columnas = predicho):")
# Convierte la matriz a texto y la muestra en el reporte.
imprimir(str(matriz))
# Muestra los conteos de verdaderos positivos, verdaderos negativos y errores.
imprimir(f"VP={VP}  VN={VN}  FP={FP}  FN={FN}")
# Muestra la exactitud calculada.
imprimir(f"Exactitud: {exact:.4f}")
# Muestra la precisión calculada.
imprimir(f"Precisión: {precision:.4f}")
# Muestra el recuerdo calculado.
imprimir(f"Recuerdo:  {recuerdo:.4f}")
# Muestra la medida F1 calculada.
imprimir(f"Medida F:  {f1:.4f}")

# Dibuja la matriz de confusión como una imagen.
# Gráfica de la matriz de confusión
# Crea una figura nueva.
plt.figure()
# Muestra los valores de la matriz como colores.
plt.imshow(matriz, cmap="Blues")
# Agrega un título que identifica la configuración usada.
plt.title(f"Matriz de confusión (lr={lr_mejor}, {ep_mejor} épocas)")
# Añade una barra que relaciona los colores con los valores.
plt.colorbar()
# Escribe las etiquetas de las clases en el eje horizontal.
plt.xticks([0, 1], ["NO-SPAM", "SPAM"])
# Escribe las etiquetas de las clases en el eje vertical.
plt.yticks([0, 1], ["NO-SPAM", "SPAM"])
# El eje horizontal representa la clase predicha.
plt.xlabel("Predicho")
# El eje vertical representa la clase real.
plt.ylabel("Real")
# Recorre las filas de la matriz para escribir cada valor.
for i in range(2):
    # Recorre las columnas de la matriz.
    for j in range(2):
        # Escribe el conteo en el centro de cada celda.
        plt.text(j, i, matriz[i, j], ha="center", va="center",
                 # Elige texto blanco u oscuro según el color de fondo para que se lea mejor.
                 color="white" if matriz[i, j] > matriz.max() / 2 else "black",
                 # Define el tamaño del texto dentro de cada celda.
                 fontsize=14)
# Guarda la imagen de la matriz de confusión.
plt.savefig(f"{CARPETA_GRAFICAS}/paso5_matriz_confusion.png", dpi=150, bbox_inches="tight")

# Dibuja la evolución del error de la mejor configuración.
# Error de entrenamiento por épocas de la mejor red
# Crea una figura nueva para esta curva.
plt.figure()
# Dibuja los errores de la tasa y hasta la época elegidas.
plt.plot(curvas[lr_mejor][:ep_mejor])
# Etiqueta el eje horizontal.
plt.xlabel("Época")
# Etiqueta el eje vertical.
plt.ylabel("Error (MSE)")
# Coloca un título con la tasa de aprendizaje de la mejor red.
plt.title(f"Error de entrenamiento por épocas (mejor red: lr={lr_mejor})")
# Agrega una cuadrícula.
plt.grid(True)
# Guarda la gráfica del error de la mejor red.
plt.savefig(f"{CARPETA_GRAFICAS}/paso5_error_mejor_red.png", dpi=150, bbox_inches="tight")

# Guarda en un archivo de texto los mensajes que se fueron generando.
# Guardar todos los resultados en un archivo de texto (útil para el reporte)
# Abre el archivo en modo escritura y con codificación UTF-8.
with open(f"{CARPETA_GRAFICAS}/resultados.txt", "w", encoding="utf-8") as f:
    # Escribe todos los mensajes, separados por saltos de línea.
    f.write("\n".join(lineas_reporte))

# Muestra las gráficas en pantalla al terminar el programa.
plt.show()
