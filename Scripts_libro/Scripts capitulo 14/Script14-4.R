# Cargar los datos.
datos <- mtcars

# Crear conjuntos de entrenamiento y prueba.
set.seed(101)
n <- nrow(datos)
n_entrenamiento <- floor(0.7 * n)
muestra <- sample.int(n = n, size = n_entrenamiento, replace = FALSE)
entrenamiento <- datos[muestra, ]
prueba  <- datos[-muestra, ]

# Ajustar modelo con el conjunto de entrenamiento.
modelo <- lm(mpg ~ wt, data = entrenamiento)
print(summary(modelo))

# Calcular error cuadrado promedio para el conjunto de entrenamiento.
mse_entrenamiento <- mean(modelo$residuals ** 2)
cat("MSE para el conjunto de entrenamiento:", mse_entrenamiento, "\n")

# Hacer predicciones para el conjunto de prueba.
predicciones <- predict(modelo, prueba)

# Calcular error cuadrado promedio para el conjunto de prueba.
error <- prueba[["mpg"]] - predicciones
mse_prueba <- mean(error ** 2)
cat("MSE para el conjunto de prueba:", mse_prueba)