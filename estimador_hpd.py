import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import matplotlib.lines as mlines
import time # Para medir el tiempo de ejecucion

def hpd_beta_arreglo(a, b, confianza=0.95, resolucion=1000):
    """
    Calcula el Intervalo de Maxima Densidad a Posteriori (HPD) 
    optimizando la longitud del intervalo mediante arreglos vectorizados.
    """
    alpha = 1 - confianza
    
    # 1. Arreglo de cuantiles para la cola izquierda
    q_grid = np.linspace(0, alpha, resolucion)
    
    # 2. Calculo vectorizado de los limites
    limites_inf = beta.ppf(q_grid, a, b)
    limites_sup = beta.ppf(q_grid + confianza, a, b)
    
    # 3. Minimizar la longitud del intervalo
    longitudes = limites_sup - limites_inf
    idx_minimo = np.argmin(longitudes)
    
    return np.array([limites_inf[idx_minimo], limites_sup[idx_minimo]])

def calcular_map_beta(a, b):
    """
    Calcula el estimador MAP (Moda) para una distribucion Beta.
    Maneja los casos extremos donde la campana se pega al 0 o al 1.
    """
    if a <= 1 and b > 1: return 0.0
    if b <= 1 and a > 1: return 1.0
    if a <= 1 and b <= 1: return 0.5 # Caso bimodal en los extremos
    return (a - 1) / (a + b - 2)

def simulador_maestro_hpd_map():
    # Iniciar cronometro
    inicio_tiempo = time.time()
    
    # 1. Definicion de parametros reales (Cadena de 2 estados)
    true_theta = 0.70
    true_alpha = 0.05
    true_beta  = 0.01
    
    # 2. Configuracion global
    num_experimentos = 10000
    M = 50
    a_pri, b_pri = 0.5, 0.5 # A priori de Jeffreys
    
    # Contadores para consola
    exitos_t_total, exitos_a_total, exitos_b_total = 0, 0, 0
    
    # Listas para la grafica (solo 100 experimentos)
    num_grafica = 1000
    datos_grafica = {
        'theta': {'map': [], 'inf': [], 'sup': [], 'exito': []},
        'alpha': {'map': [], 'inf': [], 'sup': [], 'exito': []},
        'beta':  {'map': [], 'inf': [], 'sup': [], 'exito': []}
    }
    
    print(f"Iniciando {num_experimentos} simulaciones con calculo HPD y estimador MAP...")
    print("-" * 75)
    
    # 3. Bucle de simulacion
    for i in range(num_experimentos):
        longitudes = np.random.randint(10, 61, size=M)
        max_len = longitudes.max()
        
        estados = np.zeros((M, max_len + 1), dtype=int)
        estados[:, 0] = np.random.binomial(1, true_theta, size=M)
        
        for t in range(max_len):
            estado_actual = estados[:, t]
            probs = np.where(estado_actual == 0, true_alpha, 1 - true_beta)
            estados[:, t+1] = np.random.binomial(1, probs)
            
        mascara = np.arange(max_len) < longitudes[:, None]
        trans_origen = estados[:, :-1][mascara]
        trans_destino = estados[:, 1:][mascara]
        
        y0  = np.sum(estados[:, 0])
        N00 = np.sum((trans_origen == 0) & (trans_destino == 0))
        N01 = np.sum((trans_origen == 0) & (trans_destino == 1))
        N10 = np.sum((trans_origen == 1) & (trans_destino == 0))
        N11 = np.sum((trans_origen == 1) & (trans_destino == 1))
        
        a_post_theta, b_post_theta = y0 + a_pri, (M - y0) + b_pri
        a_post_alpha, b_post_alpha = N01 + a_pri, N00 + b_pri
        a_post_beta,  b_post_beta  = N10 + a_pri, N11 + b_pri
        
        # Calculo de intervalos HPD vectorizados
        int_t = hpd_beta_arreglo(a_post_theta, b_post_theta)
        int_a = hpd_beta_arreglo(a_post_alpha, b_post_alpha)
        int_b = hpd_beta_arreglo(a_post_beta, b_post_beta)
        
        exito_t = int_t[0] <= true_theta <= int_t[1]
        exito_a = int_a[0] <= true_alpha <= int_a[1]
        exito_b = int_b[0] <= true_beta <= int_b[1]
        
        if exito_t: exitos_t_total += 1
        if exito_a: exitos_a_total += 1
        if exito_b: exitos_b_total += 1
        
        # Guardar datos para la grafica usando MAP en lugar de Media
        if i < num_grafica:
            datos_grafica['theta']['map'].append(calcular_map_beta(a_post_theta, b_post_theta))
            datos_grafica['theta']['inf'].append(int_t[0])
            datos_grafica['theta']['sup'].append(int_t[1])
            datos_grafica['theta']['exito'].append(exito_t)
            
            datos_grafica['alpha']['map'].append(calcular_map_beta(a_post_alpha, b_post_alpha))
            datos_grafica['alpha']['inf'].append(int_a[0])
            datos_grafica['alpha']['sup'].append(int_a[1])
            datos_grafica['alpha']['exito'].append(exito_a)
            
            datos_grafica['beta']['map'].append(calcular_map_beta(a_post_beta, b_post_beta))
            datos_grafica['beta']['inf'].append(int_b[0])
            datos_grafica['beta']['sup'].append(int_b[1])
            datos_grafica['beta']['exito'].append(exito_b)

    # 4. Impresion en consola
    fin_tiempo = time.time()
    print(f"RESULTADOS DE COBERTURA FRECUENTISTA (Intervalos HPD al 95%):")
    print(f"El valor oculto de Theta ({true_theta}) cayo en el intervalo el {exitos_t_total / num_experimentos * 100:.2f}% de las veces.")
    print(f"El valor oculto de Alpha ({true_alpha}) cayo en el intervalo el {exitos_a_total / num_experimentos * 100:.2f}% de las veces.")
    print(f"El valor oculto de Beta  ({true_beta}) cayo en el intervalo el {exitos_b_total / num_experimentos * 100:.2f}% de las veces.")
    print(f"Tiempo total de ejecucion: {fin_tiempo - inicio_tiempo:.2f} segundos.")

    # 5. Construccion del panel de 3 graficas
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Validacion Empirica: Cobertura de Intervalos HPD del 95% y Estimador MAP', fontsize=16)
    
    parametros = [
        ('theta', true_theta, 'Probabilidad Inicial (Theta)'),
        ('alpha', true_alpha, 'Transicion 0 -> 1 (Alpha)'),
        ('beta',  true_beta,  'Transicion 1 -> 0 (Beta)')
    ]
    
    for idx, (llave, valor_real, titulo) in enumerate(parametros):
        ax = axs[idx]
        for i in range(num_grafica):
            color = 'blue' if datos_grafica[llave]['exito'][i] else 'red'
            ax.plot([i, i], [datos_grafica[llave]['inf'][i], datos_grafica[llave]['sup'][i]], color=color, alpha=0.7)
            # Dibuja el estimador MAP como el punto en la grafica
            ax.plot(i, datos_grafica[llave]['map'][i], marker='o', markersize=3, color=color)
            
        ax.axhline(y=valor_real, color='black', linestyle='--', linewidth=2)
        ax.set_title(titulo)
        ax.set_xlabel('Simulacion')
        ax.set_ylabel('Estimador MAP / Limites HPD')
        ax.grid(True, alpha=0.2)
        
        if llave == 'theta': ax.set_ylim(0.0, 1.0)
        elif llave == 'alpha': ax.set_ylim(-0.1, 1.0)
        elif llave == 'beta': ax.set_ylim(-0.1, 1.0)

    leyenda_azul = mlines.Line2D([], [], color='blue', marker='o', markersize=5, label='Intervalo HPD Exitoso')
    leyenda_roja = mlines.Line2D([], [], color='red', marker='o', markersize=5, label='Intervalo HPD Fallido')
    leyenda_negra = mlines.Line2D([], [], color='black', linestyle='--', label='Valor Real')
    fig.legend(handles=[leyenda_negra, leyenda_azul, leyenda_roja], loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('cobertura_intervalos_hpd_map.pdf', format='pdf', bbox_inches='tight')
    plt.show()

simulador_maestro_hpd_map()