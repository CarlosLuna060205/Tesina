import numpy as np # Importa numpy para operaciones vectorizadas eficientes
import matplotlib.pyplot as plt # Importa pyplot para la generacion de la grafica
from scipy.stats import beta # Importa la distribucion beta para la inferencia

def simulador_maestro():
    # 1. Definicion de los parametros verdaderos ocultos
    true_theta = 0.60 # Probabilidad real de iniciar en estado 1
    true_alpha = 0.30 # Probabilidad real de transicion de 0 a 1
    true_beta  = 0.70 # Probabilidad real de transicion de 1 a 0
    
    # 2. Configuracion del experimento general
    num_experimentos = 10000 # Total de repeticiones para el calculo frecuentista
    M = 100 # Numero de trayectorias independientes por experimento
    confianza = 0.95 # Nivel de confianza para el intervalo
    nivel_alpha = 1 - confianza # Nivel de significancia (0.05)
    a_pri, b_pri = 0.5, 0.5 # Hiperparametros de la a priori de Jeffreys
    
    # Contadores de exitos para el reporte de texto (10,000 simulaciones)
    exitos_theta = 0
    exitos_alpha = 0
    exitos_beta  = 0
    
    # Listas para guardar datos de la grafica (solo las primeras 100 simulaciones)
    num_grafica = 200
    limites_inf_alpha = []
    limites_sup_alpha = []
    medias_alpha = []
    exitos_grafica = []
    
    # Imprime el encabezado exacto solicitado en la consola
    print(f"Iniciando {num_experimentos} simulaciones...")
    print(f"Cada simulacion genera {M} trayectorias con longitudes distintas.")
    print("-" * 60)
    
    # 3. Bucle principal de simulacion
    for i in range(num_experimentos):
        # Generar longitudes aleatorias entre 10 y 60 saltos
        longitudes = np.random.randint(10, 61, size=M)
        max_len = longitudes.max()
        
        # Matriz para simular todas las trayectorias simultaneamente
        estados = np.zeros((M, max_len + 1), dtype=int)
        estados[:, 0] = np.random.binomial(1, true_theta, size=M)
        
        # Llenado paso a paso de los estados
        for t in range(max_len):
            estado_actual = estados[:, t]
            probs = np.where(estado_actual == 0, true_alpha, 1 - true_beta)
            estados[:, t+1] = np.random.binomial(1, probs)
            
        # Filtro con mascara booleana para respetar las longitudes aleatorias
        mascara = np.arange(max_len) < longitudes[:, None]
        trans_origen = estados[:, :-1][mascara]
        trans_destino = estados[:, 1:][mascara]
        
        # Conteos globales de la simulacion actual
        y0  = np.sum(estados[:, 0])
        N00 = np.sum((trans_origen == 0) & (trans_destino == 0))
        N01 = np.sum((trans_origen == 0) & (trans_destino == 1))
        N10 = np.sum((trans_origen == 1) & (trans_destino == 0))
        N11 = np.sum((trans_origen == 1) & (trans_destino == 1))
        
        # Actualizacion Bayesiana para los tres parametros
        a_post_theta, b_post_theta = y0 + a_pri, (M - y0) + b_pri
        a_post_alpha, b_post_alpha = N01 + a_pri, N00 + b_pri
        a_post_beta,  b_post_beta  = N10 + a_pri, N11 + b_pri
        
        # Calculo de intervalos de credibilidad
        int_theta = beta.ppf([nivel_alpha/2, 1 - nivel_alpha/2], a_post_theta, b_post_theta)
        int_alpha = beta.ppf([nivel_alpha/2, 1 - nivel_alpha/2], a_post_alpha, b_post_alpha)
        int_beta  = beta.ppf([nivel_alpha/2, 1 - nivel_alpha/2], a_post_beta, b_post_beta)
        
        # Evaluacion booleana de las coberturas
        exito_t = int_theta[0] <= true_theta <= int_theta[1]
        exito_a = int_alpha[0] <= true_alpha <= int_alpha[1]
        exito_b = int_beta[0]  <= true_beta  <= int_beta[1]
        
        # Sumar a los contadores globales si el intervalo fue exitoso
        if exito_t: exitos_theta += 1
        if exito_a: exitos_alpha += 1
        if exito_b: exitos_beta += 1
        
        # Guardar datos especificos de Alpha para la grafica (solo las primeras 100 iteraciones)
        if i < num_grafica:
            media_bayesiana = a_post_alpha / (a_post_alpha + b_post_alpha)
            medias_alpha.append(media_bayesiana)
            limites_inf_alpha.append(int_alpha[0])
            limites_sup_alpha.append(int_alpha[1])
            exitos_grafica.append(exito_a)

    # 4. Impresion del reporte final exacto en la consola
    print("RESULTADOS DE COBERTURA FRECUENTISTA (Intervalos Aleatorios del 95%):")
    print(f"El valor oculto de Theta ({true_theta}) cayo en el intervalo el {exitos_theta / num_experimentos * 100:.2f}% de las veces.")
    print(f"El valor oculto de Alpha ({true_alpha}) cayo en el intervalo el {exitos_alpha / num_experimentos * 100:.2f}% de las veces.")
    print(f"El valor oculto de Beta  ({true_beta}) cayo en el intervalo el {exitos_beta / num_experimentos * 100:.2f}% de las veces.")

    # 5. Construccion de la visualizacion (Caterpillar Plot)
    plt.figure(figsize=(14, 6)) # Define tamano del lienzo
    
    # Bucle para dibujar los 100 intervalos guardados
    for i in range(num_grafica):
        color_linea = 'blue' if exitos_grafica[i] else 'red' # Asigna color segun el exito
        plt.plot([i, i], [limites_inf_alpha[i], limites_sup_alpha[i]], color=color_linea, alpha=0.7) # Dibuja intervalo
        plt.plot(i, medias_alpha[i], marker='o', markersize=4, color=color_linea) # Dibuja estimacion puntual
        
    plt.axhline(y=true_alpha, color='black', linestyle='--', linewidth=2) # Linea del valor real
    
    # Textos y etiquetas de la grafica
    plt.title('Validacion Empirica: Cobertura de Intervalos de Credibilidad del 95% (Parametro Alpha)')
    plt.xlabel('Numero de Experimento (Simulacion Independiente)')
    plt.ylabel('Valor Estimado del Parametro Alpha')
    
    # Construccion manual de la leyenda
    import matplotlib.lines as mlines
    leyenda_azul = mlines.Line2D([], [], color='blue', marker='o', markersize=5, label='Intervalo Exitoso')
    leyenda_roja = mlines.Line2D([], [], color='red', marker='o', markersize=5, label='Intervalo Fallido')
    leyenda_negra = mlines.Line2D([], [], color='black', linestyle='--', label=f'Valor Real ({true_alpha})')
    plt.legend(handles=[leyenda_negra, leyenda_azul, leyenda_roja], loc='upper right')
    
    plt.grid(True, alpha=0.2) # Activa cuadricula ligera
    plt.tight_layout() # Ajusta margenes
    plt.show() # Muestra la ventana con la grafica final

# Llamada a la funcion para iniciar el programa
simulador_maestro()