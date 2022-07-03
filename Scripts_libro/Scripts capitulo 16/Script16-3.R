library(car)

set.seed(1313)

# Cargar los datos.
datos <- mtcars
am <- factor(datos$am)
datos$am <- NULL
datos <- cbind(am, datos)

# Separar conjuntos de entrenamiento y prueba.
n <- nrow(datos)
n_entrenamiento <- floor(0.8 * n)
muestra <- sample.int(n = n, size = n_entrenamiento, replace = FALSE)
entrenamiento <- datos[muestra, ]
prueba  <- datos[-muestra, ]

# Ajustar modelo nulo.
nulo <- glm(am ~ 1, family = binomial(link = "logit"), data = entrenamiento)

# Ajustar modelo completo.
cat("\n\n")
completo <- glm(am ~ ., family = binomial(link = "logit"),
                data = entrenamiento)

# Ajustar modelo con regresión escalonada.
cat("Modelo con regresión escalonada\n")
cat("--------------------------------------\n")
mejor <- step(nulo, scope = list(lower = nulo, upper = completo),
              direction = "both", trace = 0)

print(summary(mejor))

# Verificación de multicolinealidad.
cat("Verificación de colinealidad\n")
cat("--------------------------------------\n")
cat("\nVIF:\n")
vifs <- vif(mejor)
print(vifs)
cat("\nPromedio VIF: ")
print(mean(vifs))

# Ajustar modelo con el peso como predictor.
cat("Modelo con el peso como predictor\n")
cat("--------------------------------------\n")
modelo_peso <- glm(am ~ wt, family = binomial(link = "logit"),
                   data = entrenamiento)

print(summary(modelo_peso))

# Ajustar modelo con la potencia como predictor.
cat("Modelo con la potencia como predictor\n")
cat("--------------------------------------\n")
modelo_potencia <- glm(am ~ hp, family = binomial(link = "logit"),
                       data = entrenamiento)

print(summary(modelo_potencia))

# Comparar los modelos con el peso y la potencia como predictores.
cat("\n\n")
cat("Likelihood Ratio Test para los modelos\n")
cat("--------------------------------------\n")
print(anova(modelo_peso, modelo_potencia, test = "LRT"))

# A modo de ejercicio, comparar el modelo obtenido mediante
# regresión escalonada con el que solo tiene el peso como predictor.
cat("\n\n")
cat("Likelihood Ratio Test para los modelos\n")
cat("--------------------------------------\n")
print(anova(modelo_peso, mejor, test = "LRT"))

# Independencia de los residuos.
cat("Verificación de independencia de los residuos\n")
cat("--------------------------------------\n")
print(durbinWatsonTest(modelo_peso, max.lag = 5))

# Detectar posibles valores atípicos.
cat("Identificación de posibles valores atípicos\n")
cat("--------------------------------------\n")
plot(mejor)

# Obtener los residuos y las estadísticas.
output <- data.frame(predicted.probabilities = fitted(modelo_peso))
output[["standardized.residuals"]] <- rstandard(modelo_peso)
output[["studentized.residuals"]] <- rstudent(modelo_peso)
output[["cooks.distance"]] <- cooks.distance(modelo_peso)
output[["dfbeta"]] <- dfbeta(modelo_peso)
output[["dffit"]] <- dffits(modelo_peso)
output[["leverage"]] <- hatvalues(modelo_peso)

# Evaluar residuos estandarizados que escapen a la normalidad.
# 95% de los residuos estandarizados deberían estar entre
# -1.96 y 1.96, y 99% entre -2.58 y 2.58.
sospechosos1 <- which(abs(output[["standardized.residuals"]]) > 1.96)
sospechosos1 <- sort(sospechosos1)
cat("\n\n")
cat("Residuos estandarizados fuera del 95% esperado\n")
cat("------------------------------------------------\n")
print(rownames(entrenamiento[sospechosos1, ]))

# Revisar casos con distancia de Cook mayor a uno.
sospechosos2 <- which(output[["cooks.distance"]] > 1)
sospechosos2 <- sort(sospechosos2)
cat("\n\n")
cat("Residuales con una distancia de Cook alta\n")
cat("-----------------------------------------\n")
print(rownames(entrenamiento[sospechosos2, ]))

# Revisar casos cuyo apalancamiento sea más del doble
# o triple del apalancamiento promedio.
leverage.promedio <- ncol(entrenamiento) / nrow(datos)
sospechosos3 <- which(output[["leverage"]] > leverage.promedio)
sospechosos3 <- sort(sospechosos3)
cat("\n\n")
cat("Residuales con levarage fuera de rango (> ")
cat(round(leverage.promedio, 3), ")", "\n", sep = "")
cat("--------------------------------------\n")
print(rownames(entrenamiento[sospechosos3, ]))

# Revisar casos con DFBeta >= 1.
sospechosos4 <- which(apply(output[["dfbeta"]] >= 1, 1, any))
sospechosos4 <- sort(sospechosos4)
names(sospechosos4) <- NULL
cat("\n\n")
cat("Residuales con DFBeta sobre 1\n")
cat("-----------------------------\n")
print(rownames(entrenamiento[sospechosos4, ]))

# Detalle de las observaciones posiblemente atípicas.
sospechosos <- c(sospechosos1, sospechosos2, sospechosos3, sospechosos4)
sospechosos <- sort(unique(sospechosos))
cat("\n\n")
cat("Casos sospechosos\n")
cat("-----------------\n")
print(entrenamiento[sospechosos, ])
cat("\n\n")
print(output[sospechosos, ])