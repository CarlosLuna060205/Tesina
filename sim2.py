import numpy as np # Importa numpy para operaciones vectorizadas
import matplotlib.pyplot as plt # Importa pyplot para graficar
from scipy.stats import beta # Importa la distribucion beta

def graficar_cobertura_intervalos():
    # 1. Configuracion inicial
    true_alpha = 0.30 # Valor real oculto que intentamos estimar
    num_experimentos = 1000 # Graficaremos solo 100 simulaciones para claridad visual
    M = 50 # 50 trayectorias por experimento
    nivel_alpha = 0.05 # Para construir intervalos del 95%
    a_pri, b_pri = 0.5, 0.5 # A priori de Jeffreys
    
    # Listas para guardar los resultados de cada uno de los 100 experimentos
    limites_inferiores = []
    limites_superiores = []
    estimaciones_medias = []
    captura_exitosa = [] # Guardara True si el intervalo contiene a true_alpha, False si no
    
    print("Generando simulaciones y construyendo grafica...")
    
    # 2. Bucle de simulacion (simplificado para enfocarnos en Alpha)
    for i in range(num_experimentos):
        longitudes = np.random.randint(10, 61, size=M)
        max_len = longitudes.max()
        
        estados = np.zeros((M, max_len + 1), dtype=int)
        estados[:, 0] = np.random.binomial(1, 0.60, size=M)
        
        # Simulacion de las transiciones
        for t in range(max_len):
            estado_actual = estados[:, t]
            probs = np.where(estado_actual == 0, true_alpha, 1 - 0.70)
            estados[:, t+1] = np.random.binomial(1, probs)
            
        mascara = np.arange(max_len) < longitudes[:, None]
        trans_origen = estados[:, :-1][mascara]
        trans_destino = estados[:, 1:][mascara]
        
        # Conteos especificos para el parametro Alpha (transicion 0 -> 1)
        N00 = np.sum((trans_origen == 0) & (trans_destino == 0))
        N01 = np.sum((trans_origen == 0) & (trans_destino == 1))
        
        # Actualizacion a posteriori
        a_post = N01 + a_pri
        b_post = N00 + b_pri
        
        # 3. Calculo de metricas para la grafica
        media_bayesiana = a_post / (a_post + b_post)
        intervalo = beta.ppf([nivel_alpha/2, 1 - nivel_alpha/2], a_post, b_post)
        
        # Almacenar datos del experimento iterado
        estimaciones_medias.append(media_bayesiana)
        limites_inferiores.append(intervalo[0])
        limites_superiores.append(intervalo[1])
        captura_exitosa.append(intervalo[0] <= true_alpha <= intervalo[1])
        
    # 4. Construccion de la visualizacion
    plt.figure(figsize=(14, 6)) # Define un lienzo ancho
    
    # Dibuja los 100 intervalos uno por uno
    for i in range(num_experimentos):
        # Asigna color azul si fue exitoso, rojo si el intervalo fallo
        color_linea = 'blue' if captura_exitosa[i] else 'red'
        
        # Trazar la linea vertical (el rango del intervalo)
        plt.plot([i, i], [limites_inferiores[i], limites_superiores[i]], color=color_linea, alpha=0.7)
        # Trazar un punto en la estimacion puntual (la media)
        plt.plot(i, estimaciones_medias[i], marker='o', markersize=4, color=color_linea)
        
    # Dibuja la linea horizontal punteada que representa el valor absoluto real
    plt.axhline(y=true_alpha, color='black', linestyle='--', linewidth=2)
    
    # 5. Configuracion estetica y leyendas
    plt.title('Validacion Empirica: Cobertura de Intervalos de Credibilidad del 95%')
    plt.xlabel('Numero de Experimento (Simulacion Independiente)')
    plt.ylabel('Valor Estimado del Parametro Alpha')
    
    # Construccion manual de la leyenda para mayor claridad
    import matplotlib.lines as mlines
    leyenda_azul = mlines.Line2D([], [], color='blue', marker='o', markersize=5, label='Intervalo Exitoso')
    leyenda_roja = mlines.Line2D([], [], color='red', marker='o', markersize=5, label='Intervalo Fallido')
    leyenda_negra = mlines.Line2D([], [], color='black', linestyle='--', label=f'Valor Real ({true_alpha})')
    plt.legend(handles=[leyenda_negra, leyenda_azul, leyenda_roja], loc='upper right')
    
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

graficar_cobertura_intervalos()