import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#------------------------------
# Clase Perceptrón
class Perceptron:
    def __init__(self, lr=0.01, n_iter=10):
        self.lr = lr          # Tasa de aprendizaje
        self.n_iter = n_iter  # Número de épocas
        self.pesos = None     # Vector de pesos (w)
        self.sesgo = None     # Valor del sesgo (b)
        self.errores_epoca = [] # Lista para guardar errores por época

    def fit(self, X, y):
        # INICIALIZAR pesos y sesgo(bias) en cero
        self.pesos = np.zeros(X.shape[1]) # Un peso por cada característica
        self.sesgo = 0.0
        self.errores_epoca = []

        # PARA cada época hasta n_iter:
        for _ in range(self.n_iter):
            errores = 0
            
            # PARA cada ejemplo(xi, yi):
            for xi, yi in zip(X, y):
                # calcular salida = f(w. xi + b)
                z = np.dot(self.pesos, xi) + self.sesgo
                salida = 1 if z >= 0 else 0  # Función escalón
                
                # calcular error = yi - salida
                error = yi - salida
                
                # actualizar pesos: w = w + lr * error * xi
                self.pesos = self.pesos + self.lr * error * xi
                
                # actualizar sesgo: b = b + lr * error
                self.sesgo = self.sesgo + self.lr * error
                
                # si error != 0: errores += 1
                if error != 0:
                    errores += 1
                    
            # guardar errores de la época
            self.errores_epoca.append(errores)
            
        return self

    def predict(self, X):
        # Función para predecir clases nuevas
        z = np.dot(X, self.pesos) + self.sesgo
        return np.where(z >= 0, 1, 0)

#------------------------------
# Cargar datos y Normalizar

# Cargar el dataset Iris
iris = datasets.load_iris()

# Seleccionar las primeras 100 muestras (Setosa y Versicolor) 
# y las dos primeras características (longitud y ancho del sépalo)
X = iris.data[:100, :2]  
y = iris.target[:100]    # 0 para Setosa, 1 para Versicolor

# Normalizar las características usando estandarización (z-score)
media = X.mean(axis=0)
desv_est = X.std(axis=0)
X_norm = (X - media) / desv_est

# ----------------------------------
# Entrenamiento y Gráficas

# Función sencilla para graficar la región de decisión
def graficar_region_decision(X, y, modelo, titulo, nombre_archivo):
    # Crear una malla de puntos para pintar el fondo
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))
    
    # Predecir la clase de cada punto de la malla
    Z = modelo.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # Pintar las regiones
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='coolwarm')
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Graficar los puntos reales
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c='white', edgecolors='black',
                    marker='o', label=f'Clase {cl}')
        
    plt.xlabel('Longitud del sépalo (Estandarizado)')
    plt.ylabel('Ancho del sépalo (Estandarizado)')
    plt.title(titulo)
    plt.legend()
    
    # GUARDAR LA IMAGEN
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() # Liberar memoria

# --- Experimentación con Tasas de Aprendizaje (lr) ---
tasas_aprendizaje = [0.01, 0.1, 1.0]
for lr in tasas_aprendizaje:
    modelo = Perceptron(lr=lr, n_iter=50)
    modelo.fit(X_norm, y)
    
    plt.figure(figsize=(8, 5)) # Tamaño de la figura
    plt.plot(range(1, len(modelo.errores_epoca) + 1), modelo.errores_epoca, marker='o', color='blue')
    plt.xlabel('Época')
    plt.ylabel('Errores')
    plt.title(f'Errores por época (Tasa de aprendizaje = {lr})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # GUARDAR LA IMAGEN
    nombre_archivo = f'errores_tasa_aprendizaje_{lr}.png'
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() # ¡Muy importante! Limpia la figura para la siguiente iteración

# --- Experimentación con Número de Épocas (n_iter) ---
numero_epocas = [10, 25, 50, 100]
for n_iter in numero_epocas:
    modelo = Perceptron(lr=0.1, n_iter=n_iter)
    modelo.fit(X_norm, y)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(modelo.errores_epoca) + 1), modelo.errores_epoca, marker='s', color='green')
    plt.xlabel('Época')
    plt.ylabel('Errores')
    plt.title(f'Errores por época (Número de épocas = {n_iter})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # GUARDAR LA IMAGEN
    nombre_archivo = f'errores_num_epocas_{n_iter}.png'
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() # ¡Muy importante! Limpia la figura para la siguiente iteración

# --- Gráfica de Regiones de Decisión ---
modelo_final = Perceptron(lr=0.1, n_iter=50)
modelo_final.fit(X_norm, y)
graficar_region_decision(X_norm, y, modelo_final, 'Región de Decisión del Perceptrón', 'region_decision.png')

print("¡Todas las imágenes se han generado y guardado exitosamente en tu carpeta!")