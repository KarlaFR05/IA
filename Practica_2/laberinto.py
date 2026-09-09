import heapq
import math
import time
import os
import matplotlib.pyplot as plt

# ---------------- laberinto ----------------

laberinto_texto = [
"###################",
"#S..,,...........G#",
"#.###.###########.#",
"#...#.......,,,,..#",
"#.#.#.#########.#.#",
"#.#.#.....~~~~~.#.#",
"#.#.###########.#.#",
"#.#.............#.#",
"#.###############.#",
"#.................#",
"###################",
]

laberinto_2 = [
    "###################",
    "#S,,~~...........G#",
    "#.###.###########.#",
    "#...#.......,,,,..#",
    "#.#.#.#########.#.#",
    "#.#.#.....~~~~~.#.#",
    "#.#.###########.#.#",
    "#.#.~~~,,~~~....#.#",
    "#.###,#######,###.#",
    "#........,,.......#",
    "###################",
]
# costo de cada tipo de celda
costos = {
    '.': 1,
    ',': 5,
    '~': 10,
    'S': 0,
    'G': 1,
}

def cargar_laberinto(texto):
    grid = [list(fila) for fila in texto]
    inicio = None
    meta = None
    for i, fila in enumerate(grid):
        for j, c in enumerate(fila):
            if c == 'S':
                inicio = (i, j)
            elif c == 'G':
                meta = (i, j)
    return grid, inicio, meta

def costo_celda(grid, pos):
    i, j = pos
    c = grid[i][j]
    if c == '#':
        return None
    return costos.get(c, 1)

def vecinos(grid, pos):
    i, j = pos
    filas = len(grid)
    cols = len(grid[0])
    posibles = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    result = []
    for p in posibles:
        pi, pj = p
        if 0 <= pi < filas and 0 <= pj < cols and grid[pi][pj] != '#':
            result.append(p)
    return result

# ---------------- heuristicas ----------------

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidiana(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ---------------- A* ----------------

def astar(grid, inicio, meta, heuristica, guardar_pasos=False):
    if heuristica == 'manhattan':
        h_func = manhattan
    else:
        h_func = euclidiana

    frontera = []
    contador = 0
    heapq.heappush(frontera, (h_func(inicio, meta), contador, inicio))

    g = {inicio: 0}
    padres = {}
    visitados = set()
    pasos = []  # para la visualizacion

    while frontera:
        f_actual, _, actual = heapq.heappop(frontera)

        if actual in visitados:
            continue
        visitados.add(actual)

        if guardar_pasos:
            # Reconstruir camino parcial hasta el nodo actual
            parcial = []
            curr = actual
            while curr in padres:
                parcial.append(curr)
                curr = padres[curr]
            parcial.append(inicio)
            parcial.reverse()
            pasos.append((set(visitados), set(p for _,_,p in frontera), parcial))

        if actual == meta:
            break

        for vec in vecinos(grid, actual):
            costo = costo_celda(grid, vec)
            if costo is None:
                continue
            nuevo_g = g[actual] + costo
            if vec not in g or nuevo_g < g[vec]:
                g[vec] = nuevo_g
                padres[vec] = actual
                contador += 1
                f = nuevo_g + h_func(vec, meta)
                heapq.heappush(frontera, (f, contador, vec))

    # reconstruir camino
    camino = []
    if meta in g:
        nodo = meta
        while nodo != inicio:
            camino.append(nodo)
            nodo = padres[nodo]
        camino.append(inicio)
        camino.reverse()

    resultado = {
        'camino': camino,
        'costo_total': g.get(meta, None),
        'nodos_visitados': len(visitados),
        'pasos': pasos,
    }
    return resultado

# ---------------- UCS (Dijkstra, es A* con h=0) ----------------

def ucs(grid, inicio, meta):
    return astar(grid, inicio, meta, heuristica='manhattan', guardar_pasos=False) if False else _ucs(grid, inicio, meta)

def _ucs(grid, inicio, meta):
    frontera = []
    contador = 0
    heapq.heappush(frontera, (0, contador, inicio))
    g = {inicio: 0}
    padres = {}
    visitados = set()

    while frontera:
        costo_actual, _, actual = heapq.heappop(frontera)
        if actual in visitados:
            continue
        visitados.add(actual)
        if actual == meta:
            break
        for vec in vecinos(grid, actual):
            costo = costo_celda(grid, vec)
            if costo is None:
                continue
            nuevo_g = g[actual] + costo
            if vec not in g or nuevo_g < g[vec]:
                g[vec] = nuevo_g
                padres[vec] = actual
                contador += 1
                heapq.heappush(frontera, (nuevo_g, contador, vec))

    camino = []
    if meta in g:
        nodo = meta
        while nodo != inicio:
            camino.append(nodo)
            nodo = padres[nodo]
        camino.append(inicio)
        camino.reverse()

    return {
        'camino': camino,
        'costo_total': g.get(meta, None),
        'nodos_visitados': len(visitados),
    }

# ---------------- visualizacion ----------------

colores = {
    '#': '#333333',
    '.': '#ffffff',
    ',': '#d9b382',
    '~': '#7fb3ff',
    'S': '#ffe600',
    'G': '#ff3b30',
}

def dibujar_laberinto(grid, ax, visitados=None, frontera=None, camino=None, camino_parcial=None):
    filas = len(grid)
    cols = len(grid[0])
    ax.clear()
    for i in range(filas):
        for j in range(cols):
            c = grid[i][j]
            color = colores.get(c, '#ffffff')
            ax.add_patch(plt.Rectangle((j, filas-1-i), 1, 1, color=color, ec='gray', lw=0.3))

    if visitados:
        for (i, j) in visitados:
            if grid[i][j] not in ('S', 'G'):
                ax.add_patch(plt.Rectangle((j, filas-1-i), 1, 1, color='#a0a0a0', ec='gray', lw=0.3))

    if frontera:
        for (i, j) in frontera:
            if grid[i][j] not in ('S', 'G'):
                ax.add_patch(plt.Rectangle((j, filas-1-i), 1, 1, color='#4da6ff', ec='gray', lw=0.3))

    if camino_parcial:
        for (i, j) in camino_parcial:
            if grid[i][j] not in ('S', 'G'):
                ax.add_patch(plt.Rectangle((j, filas-1-i), 1, 1, color='#90ee90', ec='gray', lw=0.3))
    if camino:
        for (i, j) in camino:
            if grid[i][j] not in ('S', 'G'):
                ax.add_patch(plt.Rectangle((j, filas-1-i), 1, 1, color='#1f9e1f', ec='gray', lw=0.3))

    ax.set_xlim(0, cols)
    ax.set_ylim(0, filas)
    ax.set_aspect('equal')
    ax.axis('off')

def generar_gif(grid, resultado, nombre_archivo):
    pasos = resultado['pasos']
    if not pasos:
        return
    carpeta = 'frames_' + nombre_archivo.replace('.gif', '')
    os.makedirs(carpeta, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    frames_paths = []

    salto = max(1, len(pasos) // 40)  # no guardar demasiados frames
    for idx in range(0, len(pasos), salto):
        visitados, frontera, parcial = pasos[idx]
        dibujar_laberinto(grid, ax, visitados=visitados, frontera=frontera, camino_parcial=parcial)
        ruta = os.path.join(carpeta, f'frame_{idx:04d}.png')
        fig.savefig(ruta, dpi=80, bbox_inches='tight')
        frames_paths.append(ruta)

    # ultimo frame con el camino final
    dibujar_laberinto(grid, ax, visitados=pasos[-1][0], camino=resultado['camino'])
    ruta_final = os.path.join(carpeta, 'frame_final.png')
    fig.savefig(ruta_final, dpi=80, bbox_inches='tight')
    frames_paths.append(ruta_final)
    frames_paths.append(ruta_final)
    frames_paths.append(ruta_final)  # repetir para que se quede un poco mas al final

    plt.close(fig)

    import imageio.v2 as imageio
    imagenes = [imageio.imread(p) for p in frames_paths]
    imageio.mimsave(nombre_archivo, imagenes, duration=0.15)

def guardar_captura_final(grid, resultado, nombre_archivo):
    fig, ax = plt.subplots(figsize=(6, 5))
    dibujar_laberinto(grid, ax, camino=resultado['camino'])
    fig.savefig(nombre_archivo, dpi=120, bbox_inches='tight')
    plt.close(fig)

# ---------------- programa principal ----------------
def main():
    # Lista de laberintos a evaluar (Nombre, Datos, Nombre de carpeta)
    laberintos_a_evaluar = [
        ("Laberinto 1 (Original)", laberinto_texto, "laberinto_1"),
        ("Laberinto 2 (Más difícil)", laberinto_2, "laberinto_2")
    ]
    
    filas_tabla = []

    for nombre_lab, texto_lab, carpeta_lab in laberintos_a_evaluar:
        print(f"\n{'='*40}")
        print(f"Evaluando: {nombre_lab}")
        print(f"{'='*40}")
        
        # Crear carpeta para este laberinto
        os.makedirs(carpeta_lab, exist_ok=True)
        
        grid, inicio, meta = cargar_laberinto(texto_lab)
        print('inicio:', inicio, ' meta:', meta)

        # Generamos animaciones para ambos laberintos
        generar_animacion = True

        for h in ['manhattan', 'euclidiana']:
            t0 = time.time()
            res = astar(grid, inicio, meta, h, guardar_pasos=generar_animacion)
            t1 = time.time()
            tiempo_ms = (t1 - t0) * 1000

            print(f'--- A* con heuristica {h} ---')
            print('longitud de la ruta:', len(res['camino']))
            print('costo total:', res['costo_total'])
            print('nodos visitados:', res['nodos_visitados'])
            print('tiempo (ms):', round(tiempo_ms, 3))

            if generar_animacion:
                # Guardar archivos dentro de la carpeta del laberinto
                guardar_captura_final(grid, res, os.path.join(carpeta_lab, f'resultado_{h}.png'))
                generar_gif(grid, res, os.path.join(carpeta_lab, f'avance_{h}.gif'))

            filas_tabla.append([
                nombre_lab,
                'A* ' + h,
                len(res['camino']),
                res['costo_total'],
                res['nodos_visitados'],
                round(tiempo_ms, 3),
            ])

        t0 = time.time()
        res_ucs = _ucs(grid, inicio, meta)
        t1 = time.time()
        tiempo_ms = (t1 - t0) * 1000

        print('--- UCS ---')
        print('longitud de la ruta:', len(res_ucs['camino']))
        print('costo total:', res_ucs['costo_total'])
        print('nodos visitados:', res_ucs['nodos_visitados'])
        print('tiempo (ms):', round(tiempo_ms, 3))

        if generar_animacion:
            # Guardar archivo dentro de la carpeta del laberinto
            guardar_captura_final(grid, res_ucs, os.path.join(carpeta_lab, 'resultado_ucs.png'))

        filas_tabla.append([
            nombre_lab,
            'UCS',
            len(res_ucs['camino']),
            res_ucs['costo_total'],
            res_ucs['nodos_visitados'],
            round(tiempo_ms, 3),
        ])

    # guardar tabla comparativa en csv 
    with open('tabla_comparativa.csv', 'w') as f:
        f.write('laberinto,metodo,longitud_ruta,costo_total,nodos_visitados,tiempo_ms\n')
        for fila in filas_tabla:
            f.write(','.join(str(x) for x in fila) + '\n')

    print()
    print('Tabla comparativa guardada en tabla_comparativa.csv')
    print('Archivos generados en carpetas: laberinto_1/ y laberinto_2/')

if __name__ == '__main__':
    main()