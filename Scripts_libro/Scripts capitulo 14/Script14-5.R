library(caret)

# Cargar los datos.
datos <- mtcars

# Crear conjuntos de entrenamiento y prueba.
set.seed(101)
n <- nrow(datos)
n_entrenamiento <- floor(0.7 * n)
muestra <- sample.int(n = n, size = n_entrenamiento, replace = FALSE)
entrenamiento <- datos[muestra, ]
prueba  <- datos[-muestra, ]

# Ajustar modelo usando validación cruzada de 5 pliegues.
modelo <- train(mpg ~ wt, data = entrenamiento, method = "lm",
                trControl = trainControl(method = "cv", number = 5))

print(summary(modelo))

# Hacer predicciones para el conjunto de entrenamiento.
predicciones_entrenamiento <- predict(modelo, entrenamiento)

# Calcular error cuadrado promedio para el conjunto de prueba.
error_entrenamiento <- entrenamiento[["mpg"]] - predicciones_entrenamiento
mse_entrenamiento <- mean(error_entrenamiento ** 2)
cat("MSE para el conjunto de entrenamiento:", mse_entrenamiento, "\n")

# Hacer predicciones para el conjunto de prueba.
predicciones_prueba <- predict(modelo, prueba)

# Calcular error cuadrado promedio para el conjunto de prueba.
error_prueba <- prueba[["mpg"]] - predicciones_prueba
mse_prueba <- mean(error_prueba ** 2)
cat("MSE para el conjunto de prueba:", mse_prueba)