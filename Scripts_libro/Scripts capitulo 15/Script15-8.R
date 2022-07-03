library(car)

# Cargar datos.
datos <- mtcars

# Ajustar modelo.
modelo <- lm(mpg ~ wt + qsec + am, data = datos)

# Comprobar independencia de los residuos.
cat("Prueba de Durbin-Watson para autocorrelaciones ")
cat("entre errores:\n")
print(durbinWatsonTest(modelo))

# Comprobar normalidad de los residuos.
cat("\nPrueba de normalidad para los residuos:\n")
print(shapiro.test(modelo$residuals))

# Comprobar homocedasticidad de los residuos.
cat("Prueba de homocedasticidad para los residuos:\n")
print(ncvTest(modelo))

# Comprobar la multicolinealidad.
vifs <- vif(modelo)
cat("\nVerificar la multicolinealidad:\n")
cat("- VIFs:\n")
print(vifs)
cat("- Tolerancias:\n")
print(1 / vifs)
cat("- VIF medio:", mean(vifs), "\n")