# Cargar datos.
datos <- mtcars

# Ajustar modelo nulo.
nulo <- lm(mpg ~ 1, data = datos)
# cat("=== Modelo nulo ===\n")
# print(summary(nulo))

# Ajustar modelo completo.
completo <- lm(mpg ~ ., data = datos)
# cat("=== Modelo completo ===\n")
# print(summary(completo))

# Evaluar variables para incorporar.
print(add1(nulo, scope = completo))
cat("\n\n")

# Agregar la variable con menor AIC.
modelo <- update(nulo, . ~ . + wt)

# Evaluar variables para incorporar.
print(add1(modelo, scope = completo))
cat("\n\n")

# Agregar la variable con menor AIC.
modelo <- update(modelo, . ~ . + cyl)

# Evaluar variables para eliminar.
print(drop1(completo, scope = completo))
cat("\n\n")

# Eliminar la variable con menor AIC.
modelo <- update(modelo, . ~ . - cyl)
