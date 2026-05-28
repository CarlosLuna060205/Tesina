import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import minimize_scalar
import matplotlib.lines as mlines
import time

def sacar_intervalos_hpd_simulacion(a, b, nivel_confianza=0.95):
    """
    Calcula el intervalo HPD adaptado para correr en un bucle automático.
    Maneja curvas unimodales y topologías en forma de 'U' (asíntotas).
    """
    mi_beta = beta(a, b)
    lo_que_sobra = 1.0 - nivel_confianza
    
    # CASO "U": Ambos parámetros son menores a 1
    if a < 1 and b < 1:
        lim_arriba_izq = mi_beta.ppf(nivel_confianza)
        ancho_1 = lim_arriba_izq - 0.0
        
        lim_abajo_der = mi_beta.ppf(lo_que_sobra)
        ancho_2 = 1.0 - lim_abajo_der
        
        # Elegimos el intervalo más estrecho
        if ancho_1 <= ancho_2:
            return np.array([0.0, lim_arriba_izq])
        else:
            return np.array([lim_abajo_der, 1.0])

    # CASO NORMAL: Es una campana o rampa hacia un solo lado
    else:
        def que_tan_ancho_es(cola_izq):
            pared_izq = mi_beta.ppf(cola_izq)
            pared_der = mi_beta.ppf(cola_izq + nivel_confianza)
            return pared_der - pared_izq

        busqueda = minimize_scalar(
            que_tan_ancho_es, 
            bounds=(0, lo_que_sobra), 
            method='bounded'
        )
        
        mejor_inicio = busqueda.x
        limite_abajo = mi_beta.ppf(mejor_inicio)
        limite_arriba = mi_beta.ppf(mejor_inicio + nivel_confianza)
        
        return np.array([limite_abajo, limite_arriba])

# NUEVA FUNCIÓN AGREGADA: Cálculo del MAP
def calcular_map_beta(a, b):
    """
    Calcula el estimador MAP (Moda) para una distribución Beta.
    Maneja las asíntotas y casos donde la campana se pega al 0 o al 1.
    """
    if a <= 1 and b > 1: 
        return 0.0
    if b <= 1 and a > 1: 
        return 1.0
    if a <= 1 and b <= 1: 
        # En una "U" simétrica o asimétrica, la moda real está en los extremos (0 o 1).
        # Por convención visual y al ser bimodal en los bordes, si ambos son <= 1
        # solemos asignar 0.5 para indicar máxima incertidumbre, o evaluamos cuál pico es más alto.
        # Aquí asignamos el extremo con mayor masa, o 0.5 si son iguales (a priori pura de Jeffreys).
        if a < b: return 0.0
        if b < a: return 1.0
        return 0.5 
    
    # Fórmula estándar de la moda para Beta(a,b) cuando a>1 y b>1
    return (a - 1) / (a + b - 2)

def simulador_maestro_panel():
    inicio_tiempo = time.time()
    
    # 1. Definicion de parametros reales
    true_theta = 0.60
    true_alpha = 0.30
    true_beta  = 0.70
    
    # 2. Configuracion global
    num_experimentos = 10000
    M = 50
    nivel_confianza = 0.95
    a_pri, b_pri = 0.5, 0.5 # A priori de Jeffreys
    
    # Contadores para consola
    exitos_t_total, exitos_a_total, exitos_b_total = 0, 0, 0
    
    # Listas para la grafica (solo 100 experimentos)
    num_grafica = 100
    
    # ACTUALIZACIÓN: Cambiamos 'medias' por 'map' en los diccionarios
    datos_grafica = {
        'theta': {'map': [], 'inf': [], 'sup': [], 'exito': []},
        'alpha': {'map': [], 'inf': [], 'sup': [], 'exito': []},
        'beta':  {'map': [], 'inf': [], 'sup': [], 'exito': []}
    }
    
    print(f"Iniciando {num_experimentos} simulaciones con cálculo HPD y estimador MAP...")
    print(f"Esto evaluará algoritmos de optimización miles de veces, espera unos segundos.")
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
        
        # Intervalos HPD
        int_t = sacar_intervalos_hpd_simulacion(a_post_theta, b_post_theta, nivel_confianza)
        int_a = sacar_intervalos_hpd_simulacion(a_post_alpha, b_post_alpha, nivel_confianza)
        int_b = sacar_intervalos_hpd_simulacion(a_post_beta,  b_post_beta,  nivel_confianza)
        
        exito_t = int_t[0] <= true_theta <= int_t[1]
        exito_a = int_a[0] <= true_alpha <= int_a[1]
        exito_b = int_b[0] <= true_beta <= int_b[1]
        
        if exito_t: exitos_t_total += 1
        if exito_a: exitos_a_total += 1
        if exito_b: exitos_b_total += 1
        
        # ACTUALIZACIÓN: Guardar el estimador MAP en lugar de la media
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
    print("RESULTADOS DE COBERTURA FRECUENTISTA (Intervalos HPD y Estimador MAP):")
    print(f"El valor oculto de Theta ({true_theta}) cayo en el intervalo el {exitos_t_total / num_experimentos * 100:.2f}% de las veces.")
    print(f"El valor oculto de Alpha ({true_alpha}) cayo en el intervalo el {exitos_a_total / num_experimentos * 100:.2f}% de las veces.")
    print(f"El valor oculto de Beta  ({true_beta}) cayo en el intervalo el {exitos_b_total / num_experimentos * 100:.2f}% de las veces.")
    print(f"Tiempo de ejecución: {fin_tiempo - inicio_tiempo:.2f} segundos.")

    # 5. Construccion del panel de 3 graficas
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Validacion Empirica: Cobertura de Intervalos HPD y Estimador MAP al 95%', fontsize=16)
    
    parametros = [
        ('theta', true_theta, 'Probabilidad Inicial (Theta)'),
        ('alpha', true_alpha, 'Transicion 0 -> 1 (Alpha)'),
        ('beta',  true_beta,  'Transicion 1 -> 0 (Beta)')
    ]
    
    # Bucle para dibujar cada una de las 3 subgraficas
    for idx, (llave, valor_real, titulo) in enumerate(parametros):
        ax = axs[idx]
        for i in range(num_grafica):
            color = 'blue' if datos_grafica[llave]['exito'][i] else 'red'
            ax.plot([i, i], [datos_grafica[llave]['inf'][i], datos_grafica[llave]['sup'][i]], color=color, alpha=0.7)
            # ACTUALIZACIÓN: Graficar los valores del MAP
            ax.plot(i, datos_grafica[llave]['map'][i], marker='o', markersize=3, color=color)
            
        ax.axhline(y=valor_real, color='black', linestyle='--', linewidth=2)
        ax.set_title(titulo)
        ax.set_xlabel('Simulacion')
        ax.set_ylabel('Estimador MAP y Límites HPD')
        ax.grid(True, alpha=0.2)
        
        # Fijar los limites del eje Y para que se vea claro el intervalo entre 0 y 1
        if llave == 'theta': ax.set_ylim(0.2, 1.0)
        elif llave == 'alpha': ax.set_ylim(0.0, 0.6)
        elif llave == 'beta': ax.set_ylim(0.4, 1.0)

    # Leyenda general para todo el panel
    leyenda_azul = mlines.Line2D([], [], color='blue', marker='o', markersize=5, label='Intervalo HPD Exitoso')
    leyenda_roja = mlines.Line2D([], [], color='red', marker='o', markersize=5, label='Intervalo HPD Fallido')
    leyenda_negra = mlines.Line2D([], [], color='black', linestyle='--', label='Valor Real')
    fig.legend(handles=[leyenda_negra, leyenda_azul, leyenda_roja], loc='upper right', bbox_to_anchor=(0.95, 0.95))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('cobertura_intervalos_hpd_map.pdf', format='pdf', bbox_inches='tight')
    plt.show()

simulador_maestro_panel()