# Ejercicio práctico N°15
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

if(!require(pROC)){
  install.packages("pROC",dependencies = TRUE)
  require(pROC)
}

if(!require(car)){
  install.packages("car",dependencies = TRUE)
  require(car)
}

if(!require(caret)){
  install.packages("caret",dependencies = TRUE)
  require(caret)
}

if(!require(leaps)){
  install.packages("leaps",dependencies = TRUE)
  require(leaps)
}


# Se carga el archivo de datos CSV
datos <- read.csv2(file.choose(new = FALSE))

###############################################################################
# 1. Definir la semilla a utilizar, que corresponde a los primeros cinco dígitos
# del RUN del integrante de mayor edad del equipo.
###############################################################################
semilla <- 20110
set.seed(semilla)
###############################################################################
# 2. Seleccionar una muestra de 100 personas, asegurando que la mitad tenga
# estado nutricional “sobrepeso” y la otra mitad “no sobrepeso”.
###############################################################################
# Calculo del IMC 
imc_datos <- datos$Weight/((datos$Height / 100) ^ 2)
# Asignación según estado nutricional
EN <- vector()
for(k in 1:507){
  if(imc_datos[k] >= 25){
    EN[k] <- 1 # Sobrepeso
  }
  else{
    EN[k] <- 0 # No sobrepeso
  }
}
EN <- as.factor(EN)

datos[["IMC"]] <- imc_datos
datos[["EN"]] <- EN

# Selección de 50 personas con sobrepeso
datos_sobrepeso <- datos %>% filter(EN == 1)
datos_sobrepeso <- sample_n(datos_sobrepeso, 50)

# Selección de 50 personas con no sobrepeso
semilla <- 20110
set.seed(semilla)
datos_no_sobrepeso <- datos %>% filter(EN == 0)
datos_no_sobrepeso <- sample_n(datos_no_sobrepeso, 50)

# Se pasan ambas muestras a un mismo conjunto
muestra <- rbind(datos_sobrepeso, datos_no_sobrepeso)

###############################################################################
# 3. Usando las herramientas del paquete leaps, realizar una búsqueda
# exhaustiva para seleccionar entre dos y ocho predictores que ayuden a 
# estimar la variable Peso (Weight), obviamente sin considerar las nuevas 
# variables IMC ni EN, y luego utilizar las funciones del paquete caret para
# construir un modelo de regresión lineal múltiple con los predictores
# escogidos y evaluarlo usando bootstrapping.
###############################################################################
# Ajustar modelo con todos los subconjuntos.
modelos <- regsubsets (Weight ~ . , data = muestra[,1:25], method = "exhaustive", 
                            nbest = 1, nvmax = 8)

# Se imprime la gráfica de los predictores
plot(modelos)

# Los predictores escogidos fueron:
# Knees.diameter + Chest.Girth + Waist.Girth + Thigh.Girth +
# Calf.Maximum.Girth + Height


set.seed(semilla)
# Se fijan las configuraciones de train, estableciendo evaluación
# por bootstrapping con 300 repeticiones
train.control <- trainControl(method = "boot", number = 300)
# Entrenar el modelo
modelo_predictor_peso <- train(Weight ~ Knees.diameter + Chest.Girth + Waist.Girth
               + Thigh.Girth + Calf.Maximum.Girth + Height,
               data = muestra, method = "lm",
               trControl = train.control)

print(modelo_predictor_peso)
# Mientras mayor el Rsquared mejor es el modelo, este fue evaluado
# con bootstrapping tomando 300 muestras
# El RMSE(Root Mean Squared Error) y  MAE(Mean Absolute Error) miden
# el error de las predicciones

###############################################################################
# 4. Haciendo un poco de investigación sobre el paquete caret, en particular
# cómo hacer Recursive Feature Elimination (RFE), construir un modelo de
# regresión lineal múltiple para predecir la variable IMC que incluya entre
# 10 y 20 predictores, seleccionando el conjunto de variables que maximice
# R2 y que use cinco repeticiones de validación cruzada de cinco pliegues
# para evitar el sobreajuste (obviamente no se debe considerar las variables
# Peso, Estatura ni estado nutricional –Weight, Height, EN respectivamente).
###############################################################################

# Reglas que usara RFE 
control <- rfeControl(functions = lmFuncs, # funciones lineares
                      method = "repeatedcv", # Permite efectuar multiples validaciones cruzadas
                      repeats = 5, # numero de repeticiones
                      number = 5) # numero de pliegues

set.seed(semilla)
# RFE
resultado_rfe1 <- rfe(x = muestra[,c(1:22,25)], # Variables Predictoras
                   y = muestra$IMC, # Variable a predecir
                   sizes = c(10:20), #Seleccionara entre 10 y 20 predictores
                   rfeControl = control, #Reglas definidas enteriormente
                   metric = "Rsquared", # Se maximiza el Rcuadrado
                   maximize = TRUE) 

# Fueron seleccionadas 17 variables
# Se imprimen los predictores seleccionados por RFE
predictors(resultado_rfe1)
# Los predictores escogidos fueron:
# Gender + Knees.diameter + Wrist.Minimum.Girth + Elbows.diameter + 
# Forearm.Girth + Calf.Maximum.Girth + Ankle.Minimum.Girth + 
# Knee.Girth + Biiliac.diameter + Wrists.diameter + Waist.Girth +
# Ankles.diameter + Hip.Girth + Chest.Girth + Bitrochanteric.diameter +
# Biacromial.diameter + Chest.diameter

# Se imprime el resumen realizado
print(resultado_rfe1)

# El modelo seleccionado fue escogido a partir del criterio de 
#maximización de R2.


###############################################################################
# 5. Usando RFE, construir un modelo de regresión logística múltiple para
# la variable EN que incluya el conjunto,de entre dos y seis, predictores
# que entregue la mejor curva ROC y que utilice validación cruzada dejando uno
# fuera para evitar el sobreajuste (obviamente no se debe considerar las 
# variables Peso, Estatura– Weight y Height respectivamente– ni IMC).
###############################################################################

# Reglas que usara RFE 
lrFuncs$summary <- twoClassSummary

# Reglas de control
control2 <- rfeControl(functions = lrFuncs, # Funciones para Regresión Logística
                      method = "LOOCV", # Validación cruzada dejando uno fuera
                      number = 5) # Número de pliegues

set.seed(semilla)

# Reglas para que se calcule la curva ROC
trainctrl <- trainControl(classProbs= TRUE,
                          summaryFunction = twoClassSummary)
# RFE
resultado_rfe2 <- rfe(x = muestra[,c(1:22,25)], # Variables Predictoras
                      y = muestra$EN, # Variable a predecir
                      sizes = c(2:6), #Seleccionara entre 2 y 6 predictores
                      rfeControl = control2, #Reglas definidas enteriormente
                      trControl = trainctrl,
                      metric = "ROC", # Se maximiza la curva ROC
                      maximize = TRUE) 

# Se imprimen los predictores seleccionados
predictors(resultado_rfe2)
# Los predictores seleccionados fueron: "Age" y "Knee.Girth"
# O sea la edad y la Suma de los diámetros de las rodillas
# Se imprime el resumen del modelo de regresión logística conseguido
print(resultado_rfe2)

###############################################################################
# 6. Pronunciarse sobre la confiabilidad y el poder predictivo de los modelos.
###############################################################################
# Para medir poder, falta confiabilidad

######### Modelo 1 - Regresión lineal multiple con búsqueda exhaustiva ######
print(modelo_predictor_peso)

# Respecto al poder de este modelo que predice el Peso (Weight) es posible
# observar que posee un R cuadrado con valor 0.964 muy cercano a 1 (variabilidad
# del rendimiento), además posee un RMSE de 2.621 y MAE de 1.97, lo que 
# significa que este modelo tiene un alto poder predictivo y un bajo nivel
# de errores al predecir.
# Respecto a su confiabilidad:
# Evaluar modelo.
# Obtener residuos y estadísticas de influencia de los casos.
eval.rlm <- data.frame(predicted.probabilities = fitted(modelo_predictor_peso[["finalModel"]]))
eval.rlm[["standardized.residuals"]] <- rstandard(modelo_predictor_peso[["finalModel"]])
eval.rlm[["studentized.residuals"]] <-rstudent(modelo_predictor_peso[["finalModel"]])
eval.rlm[["cooks.distance"]] <- cooks.distance(modelo_predictor_peso[["finalModel"]])
eval.rlm[["dfbeta"]] <- dfbeta(modelo_predictor_peso[["finalModel"]])
eval.rlm[["dffit"]] <- dffits(modelo_predictor_peso[["finalModel"]])
eval.rlm[["leverage"]] <- hatvalues(modelo_predictor_peso[["finalModel"]])
eval.rlm[["covariance.ratios"]] <- covratio(modelo_predictor_peso[["finalModel"]])

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
apalancamiento.promedio <- ncol(muestra[,1:25]) / nrow(muestra[,1:25])
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

cat("\nIndependencia de los residuos\n")
set.seed(semilla)
print(durbinWatsonTest(modelo_predictor_peso[["finalModel"]]))

# Puesto que la prueba de Durbin-Watson entrega p = 0.348, podemos concluir que
# los residuos son independientes.



######## Modelo 2 - Regresión lineal multiple con RFE ######
print(resultado_rfe1)

# Respecto al poder de este modelo que predice el IMC es posible
# observar que posee un R cuadrado con valor 0.8697 muy cercano a 1 (variabilidad
# del rendimiento), además posee un RMSE de 1.259 y MAE de 0.9830, lo que 
# significa que este modelo tiene un alto poder predictivo y un bajo nivel
# de errores al predecir.
# Respecto a su confiabilidad:
# Evaluar modelo.
# Obtener residuos y estadísticas de influencia de los casos.
eval.rlm <- data.frame(predicted.probabilities = fitted(resultado_rfe1[["fit"]]))
eval.rlm[["standardized.residuals"]] <- rstandard(resultado_rfe1[["fit"]])
eval.rlm[["studentized.residuals"]] <-rstudent(resultado_rfe1[["fit"]])
eval.rlm[["cooks.distance"]] <- cooks.distance(resultado_rfe1[["fit"]])
eval.rlm[["dfbeta"]] <- dfbeta(resultado_rfe1[["fit"]])
eval.rlm[["dffit"]] <- dffits(resultado_rfe1[["fit"]])
eval.rlm[["leverage"]] <- hatvalues(resultado_rfe1[["fit"]])
eval.rlm[["covariance.ratios"]] <- covratio(resultado_rfe1[["fit"]])

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
apalancamiento.promedio <- ncol(muestra[,c(1:22,25)]) / nrow(muestra[,c(1:22,25)])
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

cat("\nIndependencia de los residuos\n")
set.seed(semilla)
print(durbinWatsonTest(resultado_rfe1[["fit"]]))

# Puesto que la prueba de Durbin-Watson entrega p = 0.974, podemos concluir que
# los residuos son independientes.


######## Modelo 3 - Regresión logística múltiple con RFE ######

print(resultado_rfe2)

# Respecto al poder de este modelo que predice el EN es posible observar
# que tiene un ROC de 0.7896, una sensibilidad de 0.74 y especificidad de 0.72 
#por ende tiene un buen poder predictor.


# Respecto a su confiabilidad:
# Evaluar modelo.
# Obtener residuos y estadísticas de influencia de los casos.
eval.rlm <- data.frame(predicted.probabilities = fitted(resultado_rfe2[["fit"]]))
eval.rlm[["standardized.residuals"]] <- rstandard(resultado_rfe2[["fit"]])
eval.rlm[["studentized.residuals"]] <-rstudent(resultado_rfe2[["fit"]])
eval.rlm[["cooks.distance"]] <- cooks.distance(resultado_rfe2[["fit"]])
eval.rlm[["dfbeta"]] <- dfbeta(resultado_rfe2[["fit"]])
eval.rlm[["dffit"]] <- dffits(resultado_rfe2[["fit"]])
eval.rlm[["leverage"]] <- hatvalues(resultado_rfe2[["fit"]])
eval.rlm[["covariance.ratios"]] <- covratio(resultado_rfe2[["fit"]])

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
apalancamiento.promedio <- ncol(muestra[,c(1:22,25)]) / nrow(muestra[,c(1:22,25)])
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