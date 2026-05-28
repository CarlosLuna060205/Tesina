import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize_scalar

def sacar_intervalo_mas_denso(a, b, nivel_confianza=0.95):
    """
    Calcula el intervalo HPD (el pedazo donde están los valores más probables) 
    de una distribución Beta.
    """
    # OPTIMIZACIÓN: Congelamos la distribución para no recalcular 'a' y 'b' en cada paso
    mi_beta = beta(a, b)
    
    # La probabilidad que dejamos fuera del intervalo
    lo_que_sobra = 1.0 - nivel_confianza
    
    # Función que mide la distancia entre las dos "paredes" del intervalo
    def que_tan_ancho_es(cola_izq):
        pared_izq = mi_beta.ppf(cola_izq)
        pared_der = mi_beta.ppf(cola_izq + nivel_confianza)
        return pared_der - pared_izq

    # Le pedimos a SciPy que encuentre el punto de inicio que nos dé el ancho más pequeño
    busqueda = minimize_scalar(
        que_tan_ancho_es, 
        bounds=(0, lo_que_sobra), 
        method='bounded'
    )
    
    if not busqueda.success:
        raise ValueError("No se pudo encontrar el intervalo, revisa los datos.")
        
    # Sacamos el mejor punto de partida que encontró la optimización
    mejor_inicio = busqueda.x
    
    # Calculamos dónde empiezan y terminan nuestros límites finales
    limite_abajo = mi_beta.ppf(mejor_inicio)
    limite_arriba = mi_beta.ppf(mejor_inicio + nivel_confianza)
    
    return limite_abajo, limite_arriba, busqueda.fun

# ==========================================
# Ejemplo de uso
# ==========================================
aciertos = 0.5
fallos = 1.5
certeza = 0.95

abajo, arriba, ancho_minimo = sacar_intervalo_mas_denso(aciertos, fallos, certeza)

print(f"Distribución: Beta({aciertos}, {fallos})")
print(f"Intervalo más denso (HPD) al {certeza*100}%:")
print(f"Va desde: {abajo:.4f}")
print(f"Llega hasta: {arriba:.4f}")
print(f"Ancho total: {ancho_minimo:.4f}")

# Comparación rápida con el intervalo tradicional (mitad y mitad de lo que sobra)
mi_beta_ejemplo = beta(aciertos, fallos) 
mitad_fuera = (1.0 - certeza) / 2

tradi_abajo = mi_beta_ejemplo.ppf(mitad_fuera)
tradi_arriba = mi_beta_ejemplo.ppf(1.0 - mitad_fuera)

print(f"\nIntervalo tradicional (colas iguales) al {certeza*100}%:")
print(f"[{tradi_abajo:.4f}, {tradi_arriba:.4f}] - Ancho total: {(tradi_arriba - tradi_abajo):.4f}")
