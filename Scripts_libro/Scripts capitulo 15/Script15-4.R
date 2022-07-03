# Cargar datos.
datos <- mtcars

# Ajustar modelo inicial con la variable wt como predictor.
modelo <- lm(mpg ~ wt, data = datos)
cat("=== Modelo inicial ===\n")
print(modelo)

# Incorporar el predictor cyl.
modelo <- update(modelo, . ~ . + cyl)
cat("=== Modelo con predictores wt y cyl ===\n")
print(modelo)

# Quitar el predictor wt.
modelo <- update(modelo, . ~ . - wt)
cat("=== Modelo con predictor cyl ===\n")
print(modelo)

# Agregar predictores wt y drat, y quitar predictor cyl.
modelo <- update(modelo, . ~ . + wt + drat - cyl)
cat("=== Modelo con predictores wt y drat ===\n")
print(modelo)