import numpy as np # Importa la libreria principal para manejo de arreglos numericos
import matplotlib.pyplot as plt # Importa el modulo pyplot para crear las figuras y graficas
from scipy.stats import beta # Extrae unicamente la distribucion beta del modulo estadistico

# 1. Definicion de hiperparametros a priori (asumiendo distribucion uniforme para todos)
a_pri = 1 # Asigna 1 al parametro 'a' para reflejar ignorancia total inicial
b_pri = 1 # Asigna 1 al parametro 'b' para reflejar ignorancia total inicial

# 2. Definicion de las observaciones empiricas de la serie de tiempo estocastica
x0 = 1 # Define el estado inicial observado de la cadena de Markov (debe ser 0 o 1)
n00 = 12 # Contabiliza cuantas veces la cadena se mantuvo en el estado 0
n01 = 5  # Contabiliza cuantas veces la cadena transito del estado 0 al 1
n10 = 6  # Contabiliza cuantas veces la cadena transito del estado 1 al 0
n11 = 15 # Contabiliza cuantas veces la cadena se mantuvo en el estado 1

# 3. Calculo de los parametros exactos para las distribuciones a posteriori
a_theta_post = x0 + a_pri # Actualiza el parametro 'a' para la probabilidad inicial theta
b_theta_post = (1 - x0) + b_pri # Actualiza el parametro 'b' para la probabilidad inicial theta

a_alpha_post = n01 + a_pri # Actualiza el parametro 'a' para la transicion alpha (0 a 1)
b_alpha_post = n00 + b_pri # Actualiza el parametro 'b' para la transicion alpha (0 a 1)

a_beta_post = n10 + a_pri # Actualiza el parametro 'a' para la transicion beta (1 a 0)
b_beta_post = n11 + b_pri # Actualiza el parametro 'b' para la transicion beta (1 a 0)

# 4. Generacion del espacio parametrico y calculo de densidades
x = np.linspace(0, 1, 500) # Construye un vector de 500 valores continuos entre 0 y 1

pdf_priori = beta.pdf(x, a_pri, b_pri) # Calcula la densidad uniforme base para graficarla como referencia

pdf_theta = beta.pdf(x, a_theta_post, b_theta_post) # Calcula la curva de densidad a posteriori para theta
pdf_alpha = beta.pdf(x, a_alpha_post, b_alpha_post) # Calcula la curva de densidad a posteriori para alpha
pdf_beta = beta.pdf(x, a_beta_post, b_beta_post) # Calcula la curva de densidad a posteriori para beta

# 5. Configuracion del entorno de visualizacion con subgraficas
fig, axs = plt.subplots(1, 3, figsize=(18, 5)) # Crea una figura ancha que contiene 1 fila y 3 columnas de graficas

# 6. Trazado de la primera grafica: Parametro Theta
axs[0].plot(x, pdf_priori, color='gray', linestyle='--', label='A Priori') # Dibuja la linea base punteada
axs[0].plot(x, pdf_theta, color='green', label=f'Posteriori Beta({a_theta_post}, {b_theta_post})') # Dibuja la curva actualizada
axs[0].set_title('Probabilidad Inicial (Theta)') # Asigna el titulo especifico al primer panel
axs[0].set_xlabel('Valor del parametro') # Nombra el eje horizontal del primer panel
axs[0].set_ylabel('Densidad') # Nombra el eje vertical del primer panel
axs[0].legend() # Muestra las etiquetas de las curvas
axs[0].grid(True, alpha=0.3) # Activa una cuadricula suave de fondo

# 7. Trazado de la segunda grafica: Parametro Alpha
axs[1].plot(x, pdf_priori, color='gray', linestyle='--', label='A Priori') # Dibuja la linea base punteada
axs[1].plot(x, pdf_alpha, color='blue', label=f'Posteriori Beta({a_alpha_post}, {b_alpha_post})') # Dibuja la curva actualizada
axs[1].set_title('Transicion 0 -> 1 (Alpha)') # Asigna el titulo especifico al segundo panel
axs[1].set_xlabel('Valor del parametro') # Nombra el eje horizontal del segundo panel
axs[1].legend() # Muestra las etiquetas de las curvas
axs[1].grid(True, alpha=0.3) # Activa una cuadricula suave de fondo

# 8. Trazado de la tercera grafica: Parametro Beta
axs[2].plot(x, pdf_priori, color='gray', linestyle='--', label='A Priori') # Dibuja la linea base punteada
axs[2].plot(x, pdf_beta, color='purple', label=f'Posteriori Beta({a_beta_post}, {b_beta_post})') # Dibuja la curva actualizada
axs[2].set_title('Transicion 1 -> 0 (Beta)') # Asigna el titulo especifico al tercer panel
axs[2].set_xlabel('Valor del parametro') # Nombra el eje horizontal del tercer panel
axs[2].legend() # Muestra las etiquetas de las curvas
axs[2].grid(True, alpha=0.3) # Activa una cuadricula suave de fondo

# 9. Ajustes finales y ejecucion
plt.tight_layout() # Ajusta automaticamente los espaciados para que no se superpongan los textos
plt.show() # Renderiza y muestra el panel completo en pantalla