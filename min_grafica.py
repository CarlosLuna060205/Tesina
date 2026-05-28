import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import minimize_scalar

def sacar_intervalo_mas_denso(a, b, nivel_confianza=0.95):
    """
    Calcula el intervalo HPD (el pedazo donde están los valores más probables) 
    de una distribución Beta.
    """
    mi_beta = beta(a, b)
    lo_que_sobra = 1.0 - nivel_confianza
    
    def que_tan_ancho_es(cola_izq):
        pared_izq = mi_beta.ppf(cola_izq)
        pared_der = mi_beta.ppf(cola_izq + nivel_confianza)
        return pared_der - pared_izq

    busqueda = minimize_scalar(
        que_tan_ancho_es, 
        bounds=(0, lo_que_sobra), 
        method='bounded'
    )
    
    if not busqueda.success:
        raise ValueError("No se pudo encontrar el intervalo, revisa los datos.")
        
    mejor_inicio = busqueda.x
    limite_abajo = mi_beta.ppf(mejor_inicio)
    limite_arriba = mi_beta.ppf(mejor_inicio + nivel_confianza)
    
    return limite_abajo, limite_arriba, busqueda.fun

# ==========================================
# 1. Definimos los datos y sacamos las cuentas
# ==========================================
aciertos = 0.5
fallos = 0.5
certeza = 0.95

abajo, arriba, ancho_minimo = sacar_intervalo_mas_denso(aciertos, fallos, certeza)

print(f"Distribución: Beta({aciertos}, {fallos})")
print(f"Intervalo más denso (HPD) al {certeza*100}%:")
print(f"Va desde: {abajo:.4f} hasta: {arriba:.4f}")
print(f"Ancho total: {ancho_minimo:.4f}")

# ==========================================
# 2. ¡A dibujar la gráfica!
# ==========================================
# Congelamos la distribución para usarla en el dibujo
mi_beta_dibujo = beta(aciertos, fallos)

# Creamos 500 puntitos entre 0 y 1 para trazar la línea suavemente
eje_x = np.linspace(0, 1, 500)
# Calculamos la altura de la curva para cada puntito
eje_y = mi_beta_dibujo.pdf(eje_x)

# Empezamos a armar el lienzo
plt.figure(figsize=(8, 5))

# Dibujamos la línea principal de la distribución
plt.plot(eje_x, eje_y, color='black', linewidth=2, label=f'Beta({aciertos}, {fallos})')

# Coloreamos el intervalo HPD que calculamos
# Solo rellenamos si el punto 'x' está entre nuestro límite de abajo y de arriba
zona_hpd = (eje_x >= abajo) & (eje_x <= arriba)
plt.fill_between(eje_x, eje_y, where=zona_hpd, color='skyblue', alpha=0.6, 
                 label=f'HPD al {certeza*100}%')

# Le ponemos unas líneas punteadas rojas para marcar las "paredes" del intervalo
plt.axvline(abajo, color='red', linestyle='--', alpha=0.8)
plt.axvline(arriba, color='red', linestyle='--', alpha=0.8)

# Detalles de estética (títulos, etiquetas y leyenda)
plt.title('Distribución Beta y su Intervalo Más Denso (HPD)', fontsize=14)
plt.xlabel('Valores posibles', fontsize=12)
plt.ylabel('Densidad de probabilidad (qué tan probable es)', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)

# Le decimos a Python que nos muestre la obra de arte
plt.tight_layout()
plt.show()