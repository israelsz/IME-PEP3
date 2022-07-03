# Ejercicio práctico N°13
# Grupo N°2
# Integrantes:
# Christofer Rodriguez - Christian Méndez  - Israel Arias

#Importación de paquetes
if(!require(tidyverse)){
  install.packages("tidyverse",dependencies = TRUE)
  require(tidyverse)
}

if(!require(ggpubr)){
  install.packages("ggpubr",dependencies = TRUE)
  require(ggpubr)
}

if(!require(car)){
  install.packages("car",dependencies = TRUE)
  require(car)
}

# Se carga el archivo de datos CSV
datos <- read.csv2(file.choose(new = FALSE))

# Se pide construir un modelo de regresión lineal múltiple para predecir la
# variable Peso, de acuerdo con las siguientes instrucciones:
###########################################################################
# 1. Definir la semilla a utilizar, que corresponde a los últimos cuatro
# dígitos del RUN (sin considerar el dígito verificador) del integrante
# de menor edad del equipo.
###########################################################################
semilla <- 9786

###########################################################################
# 2. Seleccionar una muestra de 50 mujeres (si la semilla es un número par)
# o 50 hombres (si la semilla es impar).
###########################################################################
datos_mujeres <- datos %>% filter(Gender == 0)
# Se fija la seed
set.seed(semilla)
# Se consiguen 50 datos del dataset
muestra_mujeres <- sample_n(datos_mujeres, 50)

##########################################################################
# 3. Seleccionar de forma aleatoria ocho posibles variables predictoras.
##########################################################################
indicesValidos <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                    24)
set.seed(semilla)
variables_aleatorias <- muestra_mujeres[,sample(indicesValidos,8)]

##############################################################################
# 4. Seleccionar, de las otras variables, una que el equipo considere que
# podría ser útil para predecir la variable Peso, justificando bien
# esta selección.
##############################################################################

# https://www.colchonescefiro.es/blog/ventajas-de-un-colchon-con-zonas-de-firmeza-diferenciada/
# Tomando en consideración que al subir de peso hay un aumento de la masa
# corporal, gran porción debería acumularse en las caderas en el caso de
# una mujer debido a que este es su centro de masa y la grasa se acumula 
# en el centro del cuerpo de acuerdo con el estudio adjuntado. Es por esto
# por lo que se selecciona el diámetro de la cintura como variable predictora.

# variable_cintura <- muestra_mujeres$Waist.Girth

##############################################################################
# 5. Usando el entorno R, construir un modelo de regresión lineal simple
# con el predictor seleccionado en el paso anterior.
###############################################################################
modeloLS <- lm(Weight ~ Waist.Girth, data = muestra_mujeres)
print(summary(modeloLS))


##############################################################################
# 6. Usando herramientas para la exploración de modelos del entorno R, buscar
# entre dos y cinco predictores de entre las variables seleccionadas al azar
# en el punto 3, para agregar al modelo de regresión lineal simple obtenido
# en el paso 5.
##############################################################################

# Para conseguir seleccionar los predictores más adecuados dentro de los que
# fueron elegidos aleatoriamente es que se usa la estrategia de exploración
# de datos con selección hacía adelante, disponible 
# con la función step () de R.

# Se le concatena la variable a predecir Weight a las variables aleatorias
Weight <- muestra_mujeres$Weight
variables_aleatorias <- cbind(variables_aleatorias, Weight)

# Además se le concatena la variable del diámetro de cintura, para forzar su
# aparición en el nuevo modelo a ajustar
Waist.Girth <- muestra_mujeres$Waist.Girth
variables_aleatorias <- cbind(variables_aleatorias, Waist.Girth)

# Se crea un modelo con todas las variables aleatorias para poder usarlos
# en el método de selección hacía adelante como cota superior
modeloConVariablesAleatorias <- lm(Weight ~ ., data = variables_aleatorias)

# Ajustar modelo RLS con selección hacia adelante.
modeloRLM <- step(modeloLS,
                 scope = list(upper = modeloConVariablesAleatorias),
                 direction = "forward",
                 trace = 0)

cat("=== Modelo RLM con selección hacia adelante ===\n")
print(summary(modeloRLM))

# Las variables predictoras del modelo RLM son:
# Waist.Girth | Knee.Girth | Bitrochanteric.diameter | Height |Forearm.Girth
# Grosor a la altura de la cintura, Grosor promedio de ambas rodillas,
# Diámetro bitrocantéreo (a la altura de las caderas), Estatura,
# Grosor promedio de ambos antebrazos.

##############################################################################
# 7. Evaluar los modelos y “arreglarlos” en caso de que tengan algún problema
# con las condiciones que deben cumplir.
##############################################################################

#Modelo RLS:
# Se deben cumplir las siguientes condiciones en el modelo:
# 1. Los datos deben presentar una relación lineal.
# 2. La distribución de los residuos debe ser cercana a la normal.
# 3. La variabilidad de los puntos en torno a la línea de mínimos
#    cuadrados debe ser aproximadamente constante.
# 4. Las observaciones deben ser independientes entre sí. Esto significa
#    que no se puede usar regresión lineal con series de tiempo 


# Para verificar la primera condición se grafica el modelo:

# Graficar el modelo RLS.
p <- ggscatter (muestra_mujeres , x = "Waist.Girth", y = "Weight", color = "blue", fill = "blue",
                xlab = "Diametro Cintura [cm]", ylab = "Peso [kg]")

p <- p + geom_smooth ( method = lm , se = FALSE , colour = "red")
print (p)

# Es posible observar del gráfico que efectivamente los datos presentan
# una relación lineal, por lo que se verifica la primera condición.


# Para verificar las condiciones restantes, se usa la función plot con argumento 
# el modelo, lo que genera 4 gráficos (de residuos, Q-Q de residuos,
# residuos estandarizados y apalancamiento).
plot(modeloLS)

#Verificar normalidad de los residuos con la prueba de shapiro.
cat ( "Prueba de normalidad para los residuos\n" )
print (shapiro.test(modeloLS$residuals))

# La prueba de Shapiro arroja un p valor de 0.1204, mayor al nivel de 
# significación 0.05, por ende, los datos están distribuidos de una
# forma cercana a la normal. Esto se valida visualmente observando el
# grafico Q-Q, en el que se observa una distribución razonablemente
# cercana a la normal. En conclusión, se valida la segunda condición.

# Podemos apreciar en el grafico de residuos (residuals vs fitted)
# que la variabilidad de los residuos es relativamente constante y no presenta
# ningún comportamiento atipico. 
# Por otra parte, las observaciones son independientes entre sí, pues han sido 
# seleccionadas de manera aleatoria en un estudio y corresponden a menos del 10 % de
# la población. En consecuencia, se verifica el cumplimiento de todas las condiciones necesarias para emplear
# un modelo de RLS ajustado mediante mínimos cuadrados



# Modelo RLM:
# Se deben cumplir las siguientes condiciones en el modelo:
# 1. Las variables predictoras deben ser cuantitativas o
#   dicotómicas (de ahí la necesidad de variables indicadoras para manejar más de dos niveles).
# 2. La variable de respuesta debe ser cuantitativa y continua, sin restricciones para su variabilidad.
# 3. Los predictores deben tener algún grado de variabilidad 
#   (su varianza no debe ser igual a cero). En otras palabras, no pueden ser constantes.
# 4. No debe existir multicolinealidad. Esto significa que no deben existir relaciones lineales fuertes entre
#   dos o más predictores (coeficientes de correlación altos).
# 5. Los residuos deben ser homocedásticos (con varianzas similares) para cada nivel de los predictores.
# 6. Los residuos deben seguir una distribución cercana a la normal centrada en cero.
# 7. Los valores de la variable de respuesta son independientes entre sí.
# 8. Cada predictor se relaciona linealmente con la variable de respuesta.

# Verificación de las condiciones.
# Las variables predictores son de tipo cuantitativas, además ninguna de ella 
# corresponde a una constante. En cuanto a la variable de respuesta, esta es de tipo
# cuantitativa continua sin restricciones. Además cada predictor
# se relaciona linealmente con la variable de respuesta.
# Entonces se cumplen las condiciones 1,2,3 y 8.


# Comprobar independencia de los residuos.
cat("Prueba de Durbin-Watson para autocorrelaciones ")
cat("entre errores:\n")
print(durbinWatsonTest(modeloRLM))

# El p-valor de la prueba de durbin Watson es de 0.586, al ser mayor al
# nivel de significancia es posible concluir que los residuos son independientes.

# Comprobar normalidad de los residuos.
cat("\nPrueba de normalidad para los residuos:\n")
print(shapiro.test(modeloRLM$residuals))

# El p-valor de la prueba de shapiro es de 0.07134, mayor al nivel de significancia,
# por ende, se puede concluir que los residuos se distribuyen de forma normal.


# Comprobar homocedasticidad de los residuos.
cat("Prueba de homocedasticidad para los residuos:\n")
print(ncvTest(modeloRLM))

# El p-valor de la prueba de Breusch-Pagan-Godfre (ncvTest) es igual a 0.023051,
# mayor al nivel de significación por ende es posible concluir que se 
# cumple el supuesto de homocedasticidad.


# Comprobar la multicolinealidad.
vifs <- vif(modeloRLM)
cat("\nVerificar la multicolinealidad:\n")
cat("- VIFs:\n")
print(vifs)
cat("- Tolerancias:\n")
print(1 / vifs)
cat("- VIF medio:", mean(vifs), "\n")

# Tomando en consideración los resultados del factor de inflación de varianza (vif) y
# la tolerancia (1/vif), no hay valores de VIF preocupantes mayor a 5 o 10, como tampoco
# hay valores preocupantes menores a 0.2 para el caso de las tolerancias, sin embargo, el VIF
# medio es mayor a 1, específicamente es 3.2452, esto podría implicar la posibilidad que el
# modelo podría estar sesgado, esto no es posible corregirlo eliminando predictores
# del modelo, ya que como es posible observar, no existe valor a vif menor a 1, por
# ende es un factor riesgoso a tomar en cuenta.

# Con esto se verifican todas las condiciones restantes para el Modelo RLM


##############################################################################
# 8. Evaluar el poder predictivo del modelo en datos no utilizados para
# construirlo (o utilizando validación cruzada).
##############################################################################
datos <- muestra_mujeres

# Para modelo RLS:

# Crear conjuntos de entrenamiento y prueba.
set.seed (101)
n <- nrow(datos)
n_entrenamiento <- floor(0.7 * n)
muestra <- sample.int(n = n , size = n_entrenamiento, replace = FALSE)
entrenamiento <- datos[muestra, ]
prueba <- datos[-muestra, ]

# Ajustar modelo con el conjunto de entrenamiento .
modelo <- lm(Weight ~ Waist.Girth, data = entrenamiento)
print(summary(modelo))

# Calcular error cuadrado promedio para el conjunto de entrenamiento .
mse_entrenamiento <- mean(modelo$residuals ** 2)
cat ("MSE para el conjunto de entrenamiento :", mse_entrenamiento, "\n")

# Hacer predicciones para el conjunto de prueba .
predicciones <- predict(modelo, prueba)

# Calcular error cuadrado promedio para el conjunto de prueba .
error <- prueba [["Weight"]] - predicciones
mse_prueba <- mean(error ** 2)
cat ("MSE para el conjunto de prueba :", mse_prueba)

# Al realizar la validación cruzada para el modelo RLS se obtuvieron los siguientes valores 
# para el conjunto de entrenamiento y para el conjunto de prueba:
# MSE para el conjunto de entrenamiento : 25.80884 
# MSE para el conjunto de prueba : 24.8105
# A pesar de que el MSE es elevado, al obtener valores similares para el conjunto de entranamiento
# como para el conjunto de prueba, podemos decir que el modelo sí podría ser generalizable.


# Para modelo RLM:
        
modeloRLM2 <- lm(Weight ~ Waist.Girth + Knee.Girth + Bitrochanteric.diameter + Height + Forearm.Girth, data = entrenamiento)
print(summary(modeloRLM2))

# Calcular error cuadrado promedio para el conjunto de entrenamiento .
mse_entrenamiento <- mean(modeloRLM2$residuals ** 2)
cat ("MSE para el conjunto de entrenamiento :", mse_entrenamiento, "\n")

# Hacer predicciones para el conjunto de prueba .
predicciones <- predict(modeloRLM2, prueba)

# Calcular error cuadrado promedio para el conjunto de prueba .
error <- prueba [["Weight"]] - predicciones
mse_prueba <- mean(error ** 2)
cat ("MSE para el conjunto de prueba :", mse_prueba)

# Al realizar la validación cruzada para el modelo RLM se obtuvieron los siguientes valores 
# para el conjunto de entrenamiento y para el conjunto de prueba:
# MSE para el conjunto de entrenamiento : 5.309277 
# MSE para el conjunto de prueba : 10.90113
# A pesar de que el MSE es elevado, al obtener valores similares para el conjunto de entranamiento
# como para el conjunto de prueba, podemos decir que el modelo sí podría ser generalizable. 



#Es posible observar que el modelo RLM mejoro significativamente sus predicciones en comparación
# al modelo de RLS, al menos para este caso, lo que indica que tener más predictores mejoro para este caso.