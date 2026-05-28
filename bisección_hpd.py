import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize_scalar, bisect
import time

def hpd_minimizacion(a, b, cred_mass=0.95):
    """Método 1: Minimizando la amplitud (PPF)"""
    alpha_c = 1.0 - cred_mass
    def amplitud(p):
        return beta.ppf(p + cred_mass, a, b) - beta.ppf(p, a, b)

    res = minimize_scalar(amplitud, bounds=(0, alpha_c), method='bounded')
    p_opt = res.x
    return beta.ppf(p_opt, a, b), beta.ppf(p_opt + cred_mass, a, b)

def hpd_biseccion(a, b, cred_mass=0.95):
    """Método 2: Búsqueda de raíz usando f(L) = f(U) (PDF y PPF)"""
    alpha_c = 1.0 - cred_mass
    
    def diferencia_densidad(p):
        L = beta.ppf(p, a, b)
        U = beta.ppf(p + cred_mass, a, b)
        # Retorna f(L) - f(U)
        return beta.pdf(L, a, b) - beta.pdf(U, a, b)

    # El algoritmo de bisección encuentra la raíz (donde la diferencia es 0)
    p_opt = bisect(diferencia_densidad, 0, alpha_c)
    return beta.ppf(p_opt, a, b), beta.ppf(p_opt + cred_mass, a, b)

# ==========================================
# Comparación y Benchmark
# ==========================================
a_param, b_param = 2, 5
iteraciones = 1000

# 1. Probar Minimización
start_time = time.perf_counter()
for _ in range(iteraciones):
    lim_inf_min, lim_sup_min = hpd_minimizacion(a_param, b_param)
tiempo_min = time.perf_counter() - start_time

# 2. Probar Bisección
start_time = time.perf_counter()
for _ in range(iteraciones):
    lim_inf_bis, lim_sup_bis = hpd_biseccion(a_param, b_param)
tiempo_bis = time.perf_counter() - start_time

# Resultados
print(f"--- Resultados para Beta({a_param}, {b_param}) ---")
print(f"Método Minimización : [{lim_inf_min:.6f}, {lim_sup_min:.6f}]")
print(f"Método Bisección    : [{lim_inf_bis:.6f}, {lim_sup_bis:.6f}]\n")

print(f"--- Rendimiento ({iteraciones} iteraciones) ---")
print(f"Tiempo Minimización : {tiempo_min:.4f} segundos")
print(f"Tiempo Bisección    : {tiempo_bis:.4f} segundos")

if tiempo_min < tiempo_bis:
    print(f"\nGanador: Minimización es {tiempo_bis/tiempo_min:.2f}x más rápido.")
else:
    print(f"\nGanador: Bisección es {tiempo_min/tiempo_bis:.2f}x más rápido.")