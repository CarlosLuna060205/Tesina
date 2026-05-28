import numpy as np # Importa numpy para la generación de arreglos numéricos
import matplotlib.pyplot as plt # Importa pyplot para la creación de gráficas
from scipy.stats import beta # Importa la distribución beta de scipy para calcular la densidad

# Definición de hiperparámetros de la distribución a priori (Beta uniforme)
a_alpha_priori = 1 # Parámetro a de la priori para la transición 0 a 1
b_alpha_priori = 1 # Parámetro b de la priori para la transición 0 a 1

# Definición de las observaciones empíricas de la cadena de Markov
n00 = 2 # Número de transiciones observadas del estado 0 al estado 0
n01 = 8 # Número de transiciones observadas del estado 0 al estado 1

# Cálculo de los parámetros de la distribución a posteriori
a_alpha_posteriori = n01 + a_alpha_priori # Suma de transiciones 0 a 1 al parámetro a priori
b_alpha_posteriori = n00 + b_alpha_priori # Suma de transiciones 0 a 0 al parámetro a priori

# Generación del espacio paramétrico continuo para evaluar la función
x = np.linspace(0, 1, 500) # Crea un arreglo de 500 puntos equiespaciados entre 0 y 1

# Cálculo de la función de densidad de probabilidad (PDF)
pdf_priori = beta.pdf(x, a_alpha_priori, b_alpha_priori) # Evalúa la densidad priori en el vector x
pdf_posteriori = beta.pdf(x, a_alpha_posteriori, b_alpha_posteriori) # Evalúa la densidad posteriori en el vector x

# Configuración del lienzo de la gráfica
plt.figure(figsize=(10, 6)) # Crea una figura con dimensiones de 10x6 pulgadas

# Trazado de las curvas de densidad
plt.plot(x, pdf_priori, label='A Priori: Beta(1, 1)', color='gray', linestyle='--') # Dibuja la priori con línea punteada gris
plt.plot(x, pdf_posteriori, label=f'A Posteriori: Beta({a_alpha_posteriori}, {b_alpha_posteriori})', color='blue') # Dibuja la posteriori con línea sólida azul

# Agregado de elementos descriptivos a la gráfica
plt.title('Actualizacion Bayesiana para el parametro de transicion Alpha') # Define el título superior
plt.xlabel('Valor del parametro Alpha (Probabilidad)') # Etiqueta el eje de las abscisas
plt.ylabel('Densidad') # Etiqueta el eje de las ordenadas
plt.legend() # Inserta el cuadro con las leyendas de las curvas
plt.grid(True, alpha=0.3) # Activa una cuadrícula de fondo con transparencia del 30%

# Ejecución de la visualización
plt.show() # Despliega la ventana con la gráfica generada