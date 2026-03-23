Inferencia Bayesiana para Cadenas de Markov
Este repositorio contiene un script en Python diseñado para realizar inferencia bayesiana exacta sobre las probabilidades de transición de una Cadena de Markov en tiempo discreto con dos estados ($0$ y $1$).
El modelo toma observaciones empíricas (conteos de transiciones) y actualiza una distribución a priori uniforme utilizando la conjugación matemática entre las distribuciones Binomial y Beta, obteniendo así las distribuciones a posteriori analíticas.
Casos de Uso
Este tipo de modelado estocástico es fundamental para la cuantificación de incertidumbre en diversas áreas:
* Procesos estocásticos y estadística avanzada: Estimación robusta de parámetros cuando las muestras son pequeñas.
* Evaluación de riesgos actuariales: Modelado de probabilidades de transición entre estados de salud o estatus de pólizas.
* Modelos predictivos deportivos: Análisis de dinámicas de juego a partir de datos secuenciales, como la predicción probabilística de pasar de "posesión del balón en medio campo" a "tiro a gol" en un partido de fútbol.
Fundamento Matemático
El modelo define tres parámetros principales en el espacio continuo $[0, 1]$:
* $\theta$: Probabilidad de que el estado inicial sea $1$.
* $\alpha$: Probabilidad de transición del estado $0$ al estado $1$.
* $\beta$: Probabilidad de transición del estado $1$ al estado $0$.
Dada una secuencia de observaciones $X_{\text{obs}}$, la función de verosimilitud conjunta se define como:
$$L(\theta, \alpha, \beta | X_{\text{obs}}) = \theta^{x_0} (1-\theta)^{1-x_0} (1-\alpha)^{n_{00}} \alpha^{n_{01}} \beta^{n_{10}} (1-\beta)^{n_{11}}$$
Asumiendo distribuciones a priori independientes $\text{Beta}(a, b)$ para cada parámetro, la actualización bayesiana garantiza distribuciones a posteriori cerradas. Por ejemplo, para la transición $\alpha$:
$$\alpha | X_{\text{obs}} \sim \text{Beta}(n_{01} + a_{\alpha}, n_{00} + b_{\alpha})$$
Requisitos
Para ejecutar el script, necesitas tener instalado Python 3.x y las siguientes librerías de análisis de datos:
* numpy
* scipy
* matplotlib
Uso
1. Clona este repositorio en tu máquina local.
2. Abre el archivo principal del script.
3. Modifica la sección de "Observaciones empíricas" con los conteos ($n_{00}$, $n_{01}$, etc.) de tu propia serie de tiempo.
4. Ejecuta el script.
El programa generará automáticamente un panel con tres subgráficas que comparan la distribución a priori contra la a posteriori para $\theta$, $\alpha$ y $\beta$, permitiendo visualizar la reducción de la incertidumbre (varianza) tras incorporar la evidencia empírica.
Autor
Código desarrollado para el análisis estadístico y modelado de procesos estocásticos.
