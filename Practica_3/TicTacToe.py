import math

# Contador para saber cuantos nodos se expanden (para las metricas)
nodos = 0

def mostrar_tablero(tab):
    print("")
    print(" " + tab[0] + " | " + tab[1] + " | " + tab[2])
    print("-----------")
    print(" " + tab[3] + " | " + tab[4] + " | " + tab[5])
    print("-----------")
    print(" " + tab[6] + " | " + tab[7] + " | " + tab[8])
    print("")

def hay_ganador(tab):
    # Revisar filas, columnas y diagonales
    for i in range(3):
        # Filas
        if tab[i*3] == tab[i*3+1] == tab[i*3+2] and tab[i*3] != " ":
            return tab[i*3]
        # Columnas
        if tab[i] == tab[i+3] == tab[i+6] and tab[i] != " ":
            return tab[i]
    # Diagonales
    if tab[0] == tab[4] == tab[8] and tab[0] != " ":
        return tab[0]
    if tab[2] == tab[4] == tab[6] and tab[2] != " ":
        return tab[2]
    # Empate
    if " " not in tab:
        return "Empate"
    return None

# Funcion principal del minimax con poda alfa-beta
def minimax(tab, prof, es_max, alfa, beta):
    global nodos
    nodos = nodos + 1
    
    resultado = hay_ganador(tab)
    
    # Casos base: si alguien gano o hay empate
    if resultado == "O":
        return 1
    if resultado == "X":
        return -1
    if resultado == "Empate":
        return 0
    
    espacios = "   " * prof  # Sangria para ver la profundidad en consola
    
    if es_max:
        # Turno del maximizador (la maquina, juega con O)
        mejor = -math.inf
        for i in range(9):
            if tab[i] == " ":
                tab[i] = "O"
                valor = minimax(tab, prof + 1, False, alfa, beta)
                tab[i] = " "  # Deshacer el movimiento
                
                if valor > mejor:
                    mejor = valor
                if valor > alfa:
                    alfa = valor
                
                # Aqui se ve claramente como cambia alfa en cada iteracion
                print(espacios + "MAX puso O en " + str(i) + " | valor: " + str(valor) + " | alfa: " + str(alfa) + " | beta: " + str(beta))
                
                # Poda beta: si beta es menor o igual a alfa, cortamos
                if beta <= alfa:
                    print(espacios + "--- PODA BETA ---")
                    break
        return mejor
    else:
        # Turno del minimizador (el humano, juega con X)
        peor = math.inf
        for i in range(9):
            if tab[i] == " ":
                tab[i] = "X"
                valor = minimax(tab, prof + 1, True, alfa, beta)
                tab[i] = " "  # Deshacer el movimiento
                
                if valor < peor:
                    peor = valor
                if valor < beta:
                    beta = valor
                
                # Aqui se ve claramente como cambia beta en cada iteracion
                print(espacios + "MIN puso X en " + str(i) + " | valor: " + str(valor) + " | alfa: " + str(alfa) + " | beta: " + str(beta))
                
                # Poda alfa: si beta es menor o igual a alfa, cortamos
                if beta <= alfa:
                    print(espacios + "--- PODA ALFA ---")
                    break
        return peor

def mejor_jugada(tab):
    mejor_valor = -math.inf
    jugada = -1
    alfa = -math.inf
    beta = math.inf
    
    print("\n>>> La maquina esta pensando...")
    
    for i in range(9):
        if tab[i] == " ":
            tab[i] = "O"
            valor = minimax(tab, 0, False, alfa, beta)
            tab[i] = " "
            
            if valor > mejor_valor:
                mejor_valor = valor
                jugada = i
    
    print(">>> Mejor jugada encontrada: casilla " + str(jugada) + " con valor " + str(mejor_valor) + "\n")
    return jugada

def jugar():
    tablero = [" "] * 9
    print("Tic Tac Toe - Tu eres X, la maquina es O")
    mostrar_tablero(tablero)
    
    while True:
        # Turno del humano
        mov = int(input("Tu turno, elige casilla (0-8): "))
        if tablero[mov] != " ":
            print("Casilla ocupada, prueba otra")
            continue
        
        tablero[mov] = "X"
        mostrar_tablero(tablero)
        
        if hay_ganador(tablero):
            break
            
        # Turno de la maquina
        mov_maquina = mejor_jugada(tablero)
        tablero[mov_maquina] = "O"
        mostrar_tablero(tablero)
        
        if hay_ganador(tablero):
            break
    
    resultado = hay_ganador(tablero)
    print("Fin del juego. Resultado: " + str(resultado))
    print("Total de nodos expandidos: " + str(nodos))

jugar()