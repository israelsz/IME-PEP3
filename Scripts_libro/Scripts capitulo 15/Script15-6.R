library(leaps)

# Cargar datos.
datos <- mtcars

# Ajustar modelo nulo.
nulo <- lm(mpg ~ 1, data = datos)
cat("=== Modelo nulo ===\n")
print(summary(nulo))

# Ajustar modelo completo.
completo <- lm(mpg ~ ., data = datos)
cat("=== Modelo completo ===\n")
print(summary(completo))

# Ajustar modelo con selección hacia adelante.
adelante <- step(nulo, scope = list(upper = completo), direction = "forward",
                 trace = 0)

cat("=== Modelo con selección hacia adelante ===\n")
print(summary(adelante))
cat("AIC =", AIC(adelante), "\n\n")

# Ajustar modelo con eliminación hacia atrás.
atras <- step(completo, scope = list(lower = nulo), direction = "backward",
              trace = 0)

cat("=== Modelo con eliminación hacia atrás ===\n")
print(summary(atras))
cat("AIC =", AIC(atras), "\n\n")

# Ajustar modelo con regresión escalonada.
escalonado <- step(nulo, scope = list(lower = nulo, upper = completo),
                   direction = "both", trace = 0)

cat("=== Modelo con regresión escalonada ===\n")
print(summary(escalonado))
cat("AIC =", AIC(escalonado), "\n\n")

# Ajustar modelo con todos los subconjuntos.
modelos <- regsubsets(mpg ~ ., data = datos, method = "exhaustive",
                      nbest = 1, nvmax = 10)

print(plot(modelos))