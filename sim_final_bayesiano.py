import numpy as np # Importa numpy para operaciones vectorizadas
import matplotlib.pyplot as plt # Importa pyplot para graficar
from scipy.stats import beta # Importa la distribucion beta
import matplotlib.lines as mlines # Importa utilidades para crear la leyenda

def simulador_maestro_panel():
    # 1. Definicion de parametros reales
    true_theta = 0.60
    true_alpha = 0.30
    true_beta  = 0.70
    
    # 2. Configuracion global
    num_experimentos = 10000
    M = 50
    nivel_alpha = 0.05
    a_pri, b_pri = 0.5, 0.5
    
    # Contadores para consola
    exitos_t_total, exitos_a_total, exitos_b_total = 0, 0, 0
    
    # Listas para la grafica (solo 100 experimentos)
    num_grafica = 100
    
    # Diccionarios para guardar los datos de los 3 parametros de forma ordenada
    datos_grafica = {
        'theta': {'medias': [], 'inf': [], 'sup': [], 'exito': []},
        'alpha': {'medias': [], 'inf': [], 'sup': [], 'exito': []},
        'beta':  {'medias': [], 'inf': [], 'sup': [], 'exito': []}
    }
    
    print(f"Iniciando {num_experimentos} simulaciones...")
    print(f"Cada simulacion genera {M} trayectorias con longitudes distintas.")
    print("-" * 60)
    
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
        
        int_t = beta.ppf([nivel_alpha/2, 1 - nivel_alpha/2], a_post_theta, b_post_theta)
        int_a = beta.ppf([nivel_alpha/2, 1 - nivel_alpha/2], a_post_alpha, b_post_alpha)
        int_b = beta.ppf([nivel_alpha/2, 1 - nivel_alpha/2], a_post_beta, b_post_beta)
        
        exito_t = int_t[0] <= true_theta <= int_t[1]
        exito_a = int_a[0] <= true_alpha <= int_a[1]
        exito_b = int_b[0] <= true_beta <= int_b[1]
        
        if exito_t: exitos_t_total += 1
        if exito_a: exitos_a_total += 1
        if exito_b: exitos_b_total += 1
        
        # Guardar datos para los 3 parametros si estamos en las primeras 100 iteraciones
        if i < num_grafica:
            datos_grafica['theta']['medias'].append(a_post_theta / (a_post_theta + b_post_theta))
            datos_grafica['theta']['inf'].append(int_t[0])
            datos_grafica['theta']['sup'].append(int_t[1])
            datos_grafica['theta']['exito'].append(exito_t)
            
            datos_grafica['alpha']['medias'].append(a_post_alpha / (a_post_alpha + b_post_alpha))
            datos_grafica['alpha']['inf'].append(int_a[0])
            datos_grafica['alpha']['sup'].append(int_a[1])
            datos_grafica['alpha']['exito'].append(exito_a)
            
            datos_grafica['beta']['medias'].append(a_post_beta / (a_post_beta + b_post_beta))
            datos_grafica['beta']['inf'].append(int_b[0])
            datos_grafica['beta']['sup'].append(int_b[1])
            datos_grafica['beta']['exito'].append(exito_b)

    # 4. Impresion en consola
    print("RESULTADOS DE COBERTURA FRECUENTISTA (Intervalos Aleatorios del 95%):")
    print(f"El valor oculto de Theta ({true_theta}) cayo en el intervalo el {exitos_t_total / num_experimentos * 100:.2f}% de las veces.")
    print(f"El valor oculto de Alpha ({true_alpha}) cayo en el intervalo el {exitos_a_total / num_experimentos * 100:.2f}% de las veces.")
    print(f"El valor oculto de Beta  ({true_beta}) cayo en el intervalo el {exitos_b_total / num_experimentos * 100:.2f}% de las veces.")

    # 5. Construccion del panel de 3 graficas
    fig, axs = plt.subplots(1, 3, figsize=(18, 6)) # 1 fila, 3 columnas
    fig.suptitle('Validacion Empirica: Cobertura de Intervalos de Credibilidad del 95%', fontsize=16)
    
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
            ax.plot(i, datos_grafica[llave]['medias'][i], marker='o', markersize=3, color=color)
            
        ax.axhline(y=valor_real, color='black', linestyle='--', linewidth=2)
        ax.set_title(titulo)
        ax.set_xlabel('Simulacion')
        ax.set_ylabel('Valor Estimado')
        ax.grid(True, alpha=0.2)
        
        # Fijar los limites del eje Y para que se vea claro el intervalo entre 0 y 1
        if llave == 'theta': ax.set_ylim(0.2, 1.0)
        elif llave == 'alpha': ax.set_ylim(0.0, 0.6)
        elif llave == 'beta': ax.set_ylim(0.4, 1.0)

    # Leyenda general para todo el panel
    leyenda_azul = mlines.Line2D([], [], color='blue', marker='o', markersize=5, label='Intervalo Exitoso')
    leyenda_roja = mlines.Line2D([], [], color='red', marker='o', markersize=5, label='Intervalo Fallido')
    leyenda_negra = mlines.Line2D([], [], color='black', linestyle='--', label='Valor Real')
    fig.legend(handles=[leyenda_negra, leyenda_azul, leyenda_roja], loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajusta espacios para no superponer el titulo principal
    plt.savefig('cobertura_intervalos.pdf', format='pdf', bbox_inches='tight')
    plt.show()

simulador_maestro_panel()