import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import minimize_scalar

def sacar_intervalos_hpd(a, b, nivel_confianza=0.95):
    """
    Calcula el intervalo HPD. Si la distribución tiene forma de "U", 
    devuelve las dos soluciones posibles (anclada a la izquierda y a la derecha).
    """
    mi_beta = beta(a, b)
    lo_que_sobra = 1.0 - nivel_confianza
    
    # CASO "U": Ambos parámetros son menores a 1
    if a < 1 and b < 1:
        # Solución 1: Pegada totalmente a la pared izquierda (0)
        lim_arriba_izq = mi_beta.ppf(nivel_confianza)
        ancho_1 = lim_arriba_izq - 0.0
        solucion_1 = (0.0, lim_arriba_izq, ancho_1)
        
        # Solución 2: Pegada totalmente a la pared derecha (1)
        lim_abajo_der = mi_beta.ppf(lo_que_sobra)
        ancho_2 = 1.0 - lim_abajo_der
        solucion_2 = (lim_abajo_der, 1.0, ancho_2)
        
        return "forma_u", (solucion_1, solucion_2)

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
        
        return "normal", (limite_abajo, limite_arriba, busqueda.fun)

# ==========================================
# 1. Definimos los datos y sacamos las cuentas
# ==========================================
aciertos = 5
fallos = 2
certeza = 0.95

tipo_curva, resultados = sacar_intervalos_hpd(aciertos, fallos, certeza)

print(f"Distribución: Beta({aciertos}, {fallos})")

if tipo_curva == "forma_u":
    sol_1, sol_2 = resultados
    print("\n¡Se detectó una curva en forma de 'U'!")
    print("Hay dos posibles soluciones de un solo bloque continuo:")
    
    print(f"\nOpción A (Recargada al 0):")
    print(f"Va desde: {sol_1[0]:.4f} hasta: {sol_1[1]:.4f}")
    print(f"Ancho total: {sol_1[2]:.4f}")
    
    print(f"\nOpción B (Recargada al 1):")
    print(f"Va desde: {sol_2[0]:.4f} hasta: {sol_2[1]:.4f}")
    print(f"Ancho total: {sol_2[2]:.4f}")
else:
    abajo, arriba, ancho = resultados
    print(f"\nIntervalo más denso (HPD) al {certeza*100}%:")
    print(f"Va desde: {abajo:.4f} hasta: {arriba:.4f}")
    print(f"Ancho total: {ancho:.4f}")

# ==========================================
# 2. ¡A dibujar la(s) gráfica(s)!
# ==========================================
mi_beta_dibujo = beta(aciertos, fallos)
# Evitamos exactamente el 0 y el 1 en el dibujo para que no lance error por el infinito
eje_x = np.linspace(0.001, 0.999, 500)
eje_y = mi_beta_dibujo.pdf(eje_x)

if tipo_curva == "forma_u":
    # Dibujamos dos paneles para comparar las dos opciones
    fig, (grafica_1, grafica_2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Dibujo Opción A ---
    grafica_1.plot(eje_x, eje_y, color='black', linewidth=2)
    zona_1 = (eje_x >= sol_1[0]) & (eje_x <= sol_1[1])
    grafica_1.fill_between(eje_x, eje_y, where=zona_1, color='skyblue', alpha=0.6)
    grafica_1.axvline(sol_1[1], color='red', linestyle='--')
    grafica_1.set_title("Opción A: Anclada en el 0", fontsize=12)
    grafica_1.grid(axis='y', linestyle=':', alpha=0.7)
    
    # --- Dibujo Opción B ---
    grafica_2.plot(eje_x, eje_y, color='black', linewidth=2)
    zona_2 = (eje_x >= sol_2[0]) & (eje_x <= sol_2[1])
    grafica_2.fill_between(eje_x, eje_y, where=zona_2, color='lightgreen', alpha=0.6)
    grafica_2.axvline(sol_2[0], color='red', linestyle='--')
    grafica_2.set_title("Opción B: Anclada en el 1", fontsize=12)
    grafica_2.grid(axis='y', linestyle=':', alpha=0.7)
    
    fig.suptitle(f'Distribución Beta({aciertos}, {fallos}) - Dos soluciones posibles', fontsize=14)
    plt.tight_layout()
    plt.show()

else:
    # Dibujo normal de un solo panel
    plt.figure(figsize=(8, 5))
    plt.plot(eje_x, eje_y, color='black', linewidth=2)
    zona = (eje_x >= resultados[0]) & (eje_x <= resultados[1])
    plt.fill_between(eje_x, eje_y, where=zona, color='skyblue', alpha=0.6)
    plt.axvline(resultados[0], color='red', linestyle='--')
    plt.axvline(resultados[1], color='red', linestyle='--')
    plt.title('Distribución Beta y su Intervalo Más Denso (HPD)', fontsize=14)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()