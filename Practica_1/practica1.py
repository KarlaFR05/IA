import time

# FUNCIÓN PARA CARGAR EL LABERINTO DESDE ARCHIVO
def cargar_laberinto(nombre_archivo):
    """
    Lee un archivo de texto línea por línea y lo convierte en una lista de strings.
    Cada línea representa una fila del laberinto.
    """
    laberinto = []
    with open(nombre_archivo, 'r') as archivo:
        for linea in archivo:
            linea_limpia = linea.strip()
            if linea_limpia:
                laberinto.append(linea_limpia)
    return laberinto

# CARGAR EL LABERINTO
#laberinto = cargar_laberinto("laberinto_comparacion_BFS_DFS_UCS.txt")
laberinto = cargar_laberinto("laberinto.txt")

# FUNCIÓN PARA OBTENER EL COSTO DE UNA CELDA
def obtener_costo(celda):
    """Devuelve el costo de moverse a una celda según su símbolo."""
    if celda == '.': return 1
    if celda == ',': return 5
    if celda == '~': return 10
    if celda == 'S' or celda == 'G': return 1
    return float('inf')  # Pared '#' = inaccesible

# FUNCIÓN PARA BUSCAR INICIO Y META
def buscar_posiciones(laberinto):
    """Recorre el laberinto para encontrar las coordenadas de 'S' y 'G'."""
    inicio = meta = None
    for fila in range(len(laberinto)):
        for columna in range(len(laberinto[fila])):
            if laberinto[fila][columna] == 'S':
                inicio = (fila, columna)
            elif laberinto[fila][columna] == 'G':
                meta = (fila, columna)
    return inicio, meta

# BFS (BÚSQUEDA EN ANCHURA)
def bfs(laberinto, inicio, meta):
    tiempo_inicio = time.time()
    
    cola = [inicio]
    visitados = set()
    visitados.add(inicio)
    padres = {inicio: None}
    
    while cola:
        actual = cola.pop(0)  # FIFO: saca el primero
        
        if actual == meta:
            break
        
        fila, columna = actual
        # Orden: Izquierda, Abajo, Derecha, Arriba
        direcciones = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        for df, dc in direcciones:
            nueva_fila, nueva_columna = fila + df, columna + dc
            
            if 0 <= nueva_fila < len(laberinto) and 0 <= nueva_columna < len(laberinto[0]):
                if laberinto[nueva_fila][nueva_columna] != '#' and (nueva_fila, nueva_columna) not in visitados:
                    vecino = (nueva_fila, nueva_columna)
                    visitados.add(vecino)
                    padres[vecino] = actual
                    cola.append(vecino)
    
    tiempo_fin = time.time()
    tiempo_ms = (tiempo_fin - tiempo_inicio) * 1000
    
    # Reconstruir camino
    camino = []
    actual = meta
    if meta in padres:
        while actual is not None:
            camino.append(actual)
            actual = padres[actual]
        camino.reverse()
    
    # Calcular métricas
    longitud_ruta = len(camino) - 1 if camino else 0
    costo_total = sum(obtener_costo(laberinto[f][c]) for f, c in camino)
    
    return {
        "camino": camino,
        "longitud_pasos": longitud_ruta,
        "costo_total": costo_total,
        "nodos_visitados": len(visitados),
        "tiempo_ms": round(tiempo_ms, 4)
    }

def dfs(laberinto, inicio, meta):
    tiempo_inicio = time.time()
    
    pila = [inicio]
    visitados = set()
    visitados.add(inicio)
    padres = {inicio: None}
    
    while pila:
        actual = pila.pop()  # LIFO: saca el último
        
        if actual == meta:
            break
        
        fila, columna = actual
        # Orden: Izquierda, Abajo, Derecha, Arriba
        direcciones = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        for df, dc in direcciones:
            nueva_fila, nueva_columna = fila + df, columna + dc
            
            if 0 <= nueva_fila < len(laberinto) and 0 <= nueva_columna < len(laberinto[0]):
                if laberinto[nueva_fila][nueva_columna] != '#' and (nueva_fila, nueva_columna) not in visitados:
                    vecino = (nueva_fila, nueva_columna)
                    visitados.add(vecino)
                    padres[vecino] = actual
                    pila.append(vecino)
    
    tiempo_fin = time.time()
    tiempo_ms = (tiempo_fin - tiempo_inicio) * 1000
    
    # Reconstruir camino
    camino = []
    actual = meta
    if meta in padres:
        while actual is not None:
            camino.append(actual)
            actual = padres[actual]
        camino.reverse()
    
    # Calcular métricas
    longitud_ruta = len(camino) - 1 if camino else 0
    costo_total = sum(obtener_costo(laberinto[f][c]) for f, c in camino)
    
    return {
        "camino": camino,
        "longitud_pasos": longitud_ruta,
        "costo_total": costo_total,
        "nodos_visitados": len(visitados),
        "tiempo_ms": round(tiempo_ms, 4)
    }


# FUNCIÓN PARA RESULTADOS
def mostrar_resultados(nombre_algoritmo, resultados):
    """Muestra las métricas de forma bonita y uniforme."""
    print(f"\n--- MÉTRICAS {nombre_algoritmo} ---")
    print(f"Longitud de ruta (pasos): {resultados['longitud_pasos']}")
    print(f"Costo total de la ruta  : {resultados['costo_total']}")
    print(f"Nodos visitados         : {resultados['nodos_visitados']}")
    print(f"Tiempo de ejecución     : {resultados['tiempo_ms']} ms")
    print(f"\nCamino: {resultados['camino']}")

#menu
if __name__ == "__main__":
    inicio, meta = buscar_posiciones(laberinto)
    
    opcion = 0
    while opcion != 4:
        print("\n--- MENÚ DE ALGORITMOS ---")
        print("1. DFS")
        print("2. BFS")
        print("3. UCS")
        print("4. Salir")
        
        opcion = int(input("Elige una opción del 1 al 4: "))
        
        if opcion == 1:
            print("\nEjecutando DFS (Búsqueda en Profundidad)...")
            resultados = dfs(laberinto, inicio, meta)
            mostrar_resultados("DFS", resultados)
            
        elif opcion == 2:
            print("\nEjecutando BFS (Búsqueda en Anchura)...")
            resultados = bfs(laberinto, inicio, meta)
            mostrar_resultados("BFS", resultados)
            
        elif opcion == 3:
            print("\nEjecutando UCS (Búsqueda de Costo Uniforme)")
            
        elif opcion == 4:
            print("\nSaliendo")
            
        else:
            print("Opción no válida")