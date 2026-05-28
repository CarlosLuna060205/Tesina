import numpy as np
from scipy.stats import beta

def simulador_cobertura_bayesiana():
    # 1. Definición de los parámetros verdaderos (la "realidad oculta")
    true_theta = 0.60
    true_alpha = 0.30
    true_beta  = 0.70
    
    # 2. Configuración del simulador
    num_experimentos = 10000 # Cuántas veces calcularemos intervalos aleatorios
    M = 500                   # Número de trayectorias por experimento
    confianza = 0.95         # Intervalos del 95%
    nivel_alpha = 1 - confianza
    
    # Contadores: ¿Cuántas veces el intervalo logró capturar el valor real?
    exitos_theta = 0
    exitos_alpha = 0
    exitos_beta  = 0
    
    # Hiperparámetros de la distribución a priori (Beta uniforme)
    a_pri, b_pri = 0.5, 0.5
    
    print(f"Iniciando {num_experimentos} simulaciones...")
    print(f"Cada simulación genera {M} trayectorias con longitudes distintas.")
    print("-" * 60)

    for i in range(num_experimentos):
        # 3. Generar longitudes aleatorias distintas para las M trayectorias (ej. entre 10 y 60 saltos)
        longitudes = np.random.randint(10, 61, size=M)
        max_len = longitudes.max()
        
        # 4. MATRIZ DE SIMULACIÓN VECTORIZADA (Arreglos NumPy)
        # Matriz de estados de tamaño (M trayectorias x máxima longitud + 1)
        estados = np.zeros((M, max_len + 1), dtype=int)
        
        # Estado inicial (X_0) distribuido como Bernoulli(true_theta)
        estados[:, 0] = np.random.binomial(1, true_theta, size=M)
        
        # Llenado simultáneo de las M trayectorias paso a paso
        for t in range(max_len):
            estado_actual = estados[:, t]
            
            # Si es 0, la prob de éxito es alpha. Si es 1, la prob de éxito es (1 - beta)
            probabilidades_salto = np.where(estado_actual == 0, true_alpha, 1 - true_beta)
            estados[:, t+1] = np.random.binomial(1, probabilidades_salto)
            
        # 5. Filtrar la matriz usando una máscara para respetar las longitudes distintas
        # Se crea una cuadrícula booleana que apaga las celdas que superan la longitud de su trayectoria
        mascara = np.arange(max_len) < longitudes[:, None]
        
        transiciones_origen = estados[:, :-1][mascara]
        transiciones_destino = estados[:, 1:][mascara]
        
        # 6. Conteos empíricos totales sumando todas las trayectorias
        y0  = np.sum(estados[:, 0])
        N00 = np.sum((transiciones_origen == 0) & (transiciones_destino == 0))
        N01 = np.sum((transiciones_origen == 0) & (transiciones_destino == 1))
        N10 = np.sum((transiciones_origen == 1) & (transiciones_destino == 0))
        N11 = np.sum((transiciones_origen == 1) & (transiciones_destino == 1))
        
        # 7. Actualización Bayesiana (A posteriori)
        a_post_theta, b_post_theta = y0 + a_pri, (M - y0) + b_pri
        a_post_alpha, b_post_alpha = N01 + a_pri, N00 + b_pri
        a_post_beta,  b_post_beta  = N10 + a_pri, N11 + b_pri
        
        # 8. Aproximaciones por Intervalos de Longitud Mínima (HPD / Credibilidad)
        # Nota: Al tener tantas muestras, el intervalo central simétrico de las colas converge
        # analíticamente al intervalo de longitud mínima (Highest Posterior Density).
        int_theta = beta.ppf([nivel_alpha/2, 1 - nivel_alpha/2], a_post_theta, b_post_theta)
        int_alpha = beta.ppf([nivel_alpha/2, 1 - nivel_alpha/2], a_post_alpha, b_post_alpha)
        int_beta  = beta.ppf([nivel_alpha/2, 1 - nivel_alpha/2], a_post_beta, b_post_beta)
        
        # 9. Verificación empírica: ¿Cayó el verdadero valor (desconocido) en el intervalo calculado?
        if int_theta[0] <= true_theta <= int_theta[1]: exitos_theta += 1
        if int_alpha[0] <= true_alpha <= int_alpha[1]: exitos_alpha += 1
        if int_beta[0]  <= true_beta  <= int_beta[1]:  exitos_beta += 1

    # 10. Resultados Finales
    print("RESULTADOS DE COBERTURA FRECUENTISTA (Intervalos Aleatorios del 95%):")
    print(f"El valor oculto de Theta ({true_theta}) cayó en el intervalo el {exitos_theta / num_experimentos * 100:.2f}% de las veces.")
    print(f"El valor oculto de Alpha ({true_alpha}) cayó en el intervalo el {exitos_alpha / num_experimentos * 100:.2f}% de las veces.")
    print(f"El valor oculto de Beta  ({true_beta}) cayó en el intervalo el {exitos_beta / num_experimentos * 100:.2f}% de las veces.")

# Ejecutar simulador
simulador_cobertura_bayesiana()