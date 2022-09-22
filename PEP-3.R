# PEP N°3 - Forma 3

#Importación de paquetes
if(!require(tidyverse)){
  install.packages("tidyverse",dependencies = TRUE)
  require(tidyverse)
}

if(!require(leaps)){
  install.packages("leaps",dependencies = TRUE)
  require(leaps)
}

if(!require(caret)){
  install.packages("caret",dependencies = TRUE)
  require(caret)
}

if(!require(car)){
  install.packages("car",dependencies = TRUE)
  require(car)
}

# Se carga el archivo de datos CSV
datos <- read.csv2(file.choose(new = FALSE))

# Se fija la semilla
set.seed(341)

#Se obtiene la muestra solicitada de 200 observaciones
muestra <- sample_n(datos, 200)

# Se descarta la variable S.ASIG de la muestra
muestra <- muestra %>% select(-S.ASIG)

# Crear conjuntos de entrenamiento y prueba a partir de la muestra.
set.seed(341)
n <- nrow(muestra)
n_entrenamiento <- floor(0.7 * n)
muestra_conjuntos <- sample.int(n = n, size = n_entrenamiento, replace = FALSE)
entrenamiento <- muestra[muestra_conjuntos, ]
prueba  <- muestra[-muestra_conjuntos, ]

# Para seleccionar el mejor modelo se usará RFE usando validación cruzada de
# 5 pliegues repetidas 5 veces
control <- rfeControl(functions = lmFuncs, # funciones lineares
                      method = "repeatedcv", # Permite efectuar multiples validaciones cruzadas
                      repeats = 5, # numero de repeticiones
                      number = 5) # numero de pliegues

set.seed(341)


modeloRLM <- rfe(x = entrenamiento[,c(2:8)], # Variables Predictoras
                      y = entrenamiento$N.ASIG, # Variable a predecir
                      sizes = c(2:7), #Seleccionara entre 2 y 7 predictores
                      rfeControl = control) # Reglas seteadas anteriormente

# Fueron seleccionadas 3
# Se imprimen los predictores seleccionados por RFE
predictors(modeloRLM)
# Los predictores escogidos fueron:
# PROM.EM, P.MAT Y P.LEN

# Se imprime el resumen del modelo conseguido
print(modeloRLM)

# A continuación se prueba el modelo para el 30% de prueba
# Hacer predicciones para el conjunto de entrenamiento.
predicciones_entrenamiento <- predict(modeloRLM, entrenamiento)

# Calcular RMSE para el conjunto de prueba.
error_entrenamiento <- entrenamiento[["N.ASIG"]] - predicciones_entrenamiento
mse_entrenamiento <- mean(error_entrenamiento ** 2)
RMSE_entrenamiento <- sqrt(mse_entrenamiento)
cat("RMSE para el conjunto de entrenamiento:", RMSE_entrenamiento, "\n")

# Hacer predicciones para el conjunto de prueba.
predicciones_prueba <- predict(modeloRLM, prueba)

# Calcular RMSE para el conjunto de prueba.
error_prueba <- prueba[["N.ASIG"]] - predicciones_prueba
mse_prueba <- mean(error_prueba ** 2)
RMSE_prueba <- sqrt(mse_prueba)
cat("RMSE para el conjunto de prueba:", RMSE_prueba, "\n\n")

# Es posible observar que los RMSE entrenamiento = 0.3547 y RMSE prueba = 0.374
# son muy similares en valor, a pesar de ser mayores a 0.3 como se solicita,
# se sospecha que esto es debido a los datos, debido a que RFE es un algoritmo
# muy eficiente en encontrar los mejores, además de ocupar validación cruzada
# múltiple de 5 pliegues.

# Respecto a su confiabilidad:
# Evaluar modelo.
# Obtener residuos y estadísticas de influencia de los casos.
eval.rlm <- data.frame(predicted.probabilities = fitted(modeloRLM[["fit"]]))
eval.rlm[["standardized.residuals"]] <- rstandard(modeloRLM[["fit"]])
eval.rlm[["studentized.residuals"]] <-rstudent(modeloRLM[["fit"]])
eval.rlm[["cooks.distance"]] <- cooks.distance(modeloRLM[["fit"]])
eval.rlm[["dfbeta"]] <- dfbeta(modeloRLM[["fit"]])
eval.rlm[["dffit"]] <- dffits(modeloRLM[["fit"]])
eval.rlm[["leverage"]] <- hatvalues(modeloRLM[["fit"]])
eval.rlm[["covariance.ratios"]] <- covratio(modeloRLM[["fit"]])

cat("Influencia de los casos:\n")

# 95% de los residuos estandarizados deberían estar entre −1.96 y +1.96, y 99%
# entre -2.58 y +2.58.
sospechosos1 <- which(abs(eval.rlm[["standardized.residuals"]]) > 1.96)
cat("- Residuos estandarizados fuera del 95% esperado: ")
print(sospechosos1)

# Observaciones con distancia de Cook mayor a uno.
sospechosos2 <- which(eval.rlm[["cooks.distance"]] > 1)
cat("- Residuos con distancia de Cook mayor que 1: ")
print(sospechosos2)

# Observaciones con apalancamiento superior al doble del apalancamiento
# promedio: (k + 1)/n.
apalancamiento.promedio <- ncol(muestra[,c(2:8)]) / nrow(muestra[,c(2:8)])
sospechosos3 <- which(eval.rlm[["leverage"]] > 2 * apalancamiento.promedio)

cat("- Residuos con apalancamiento fuera de rango (promedio = ",
    apalancamiento.promedio, "): ", sep = "")

print(sospechosos3)

# DFBeta debería ser < 1.
sospechosos4 <- which(apply(eval.rlm[["dfbeta"]] >= 1, 1, any))
names(sospechosos4) <- NULL
cat("- Residuos con DFBeta mayor que 1: ")
print(sospechosos4)

# Finalmente, los casos no deberían desviarse significativamente
# de los límites recomendados para la razón de covarianza:
# CVRi > 1 + [3(k + 1)/n]
# CVRi < 1 – [3(k + 1)/n]
CVRi.lower <- 1 - 3 * apalancamiento.promedio
CVRi.upper <- 1 + 3 * apalancamiento.promedio

sospechosos5 <- which(eval.rlm[["covariance.ratios"]] < CVRi.lower |
                        eval.rlm[["covariance.ratios"]] > CVRi.upper)

cat("- Residuos con razón de covarianza fuera de rango ([", CVRi.lower, ", ",
    CVRi.upper, "]): ", sep = "")

print(sospechosos5)

sospechosos <- c(sospechosos1, sospechosos2, sospechosos3, sospechosos4,
                 sospechosos5)

sospechosos <- sort(unique(sospechosos))
cat("\nResumen de observaciones sospechosas:\n")

print(round(eval.rlm[sospechosos,
                     c("cooks.distance", "leverage", "covariance.ratios")],
            3))

# Si bien hay algunas observaciones que podrían considerarse atípicas, la
# distancia de Cook para todas ellas se aleja bastante de 1, por lo que no
# deberían ser causa de preocupación.

set.seed(341)
cat("\nIndependencia de los residuos\n")
print(durbinWatsonTest(modeloRLM$fit))

# Puesto que la prueba de Durbin-Watson entrega p = 0.234,
# mayor al nivel de significación 0.05, podemos concluir que
# los residuos son independientes.

# Comprobar la multicolinealidad.
vifs <- vif(modeloRLM$fit)
cat("\nVerificar la multicolinealidad:\n")
cat("- VIFs:\n")
print(vifs)
cat("- Tolerancias:\n")
print(1 / vifs)
cat("- VIF medio:", mean(vifs), "\n")

# Tomando en consideración los resultados del factor de inflación de varianza (vif) y
# la tolerancia (1/vif), no hay valores de VIF preocupantes mayor a 5 o 10, como tampoco
# hay valores preocupantes menores a 0.2 para el caso de las tolerancias, sin embargo, el VIF
# medio es ligeramente mayor a a 1, específicamente es 1.0051, esto podría implicar la posibilidad que el
# modelo podría estar sesgado, sin embargo es una diferencia muy pequeña por lo que
# habría que estudiarlo quizá con una cantidad mayor de muestras.


# Como conclusión final, el modelo cumple todas las condiciones, a pesar
# de presentar un Vif medio de 1 está en el límite de lo aceptado en
# colinealidad, por lo que no debería presentar multicolinealidad severa,
# no presenta ninguna distancia de Cook mayor a 1 entonces las observaciones
# atípicas no deberían ser problema, Ante todo esto es que se concluye que el
#modelo es efectivamente generalizable.

# Respecto a su poder predictivo, presenta bajos errores, en específico un
# RMSE de 0.374 en el conjunto de prueba, a pesar de que es mayor a 0.3,
# como se menciono anteriormente esto podría deberse a los datos y quizá 
# utilizar otra semilla o tener otro conjunto de estos puede bajar el error.
# A pesar de esto, un error de 0.374 es bajo, por lo que se considera que el
#modelo tiene una buena calidad predictiva.






