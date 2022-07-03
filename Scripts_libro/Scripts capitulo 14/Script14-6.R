library(ggpubr)

# Crear los datos originales.
requisitos <- c(11, 10, 12, 14, 8, 13, 18, 15, 20, 16, 21, 13, 10, 9, 21)
stakeholders <- c(8, 8, 6, 6, 8, 7, 3, 1, 3, 4, 5, 4, 4, 9, 2)
datos <- data.frame(requisitos, stakeholders)

# Ajustar modelo.
modelo <- lm(requisitos ~ stakeholders, data = datos)
print(summary(modelo))

# Graficar el modelo.
p <- ggscatter(
  datos, x = "stakeholders", y = "requisitos", color = "blue", fill = "blue",
  xlab = "Stakeholders", ylab = "Requisitos funcionales")

p <- p + geom_smooth(method = lm, se = FALSE, colour = "red")

# Graficar los residuos.
b_1 <- modelo$coefficients[2]
b_0 <- modelo$coefficients[1]
residuos <- datos[["requisitos"]] - (b_1 * datos[["stakeholders"]] + b_0)
datos <- data.frame(datos, residuos)

r <- ggscatter(datos, x = "stakeholders", y = "residuos", color = "blue",
               fill = "blue", xlab = "Stakeholders", ylab = "Residuo")

r <- r + geom_hline(yintercept = 0, colour = "red")

g <- ggarrange(p, r, ncol = 2, nrow = 1)
print(g)

# Verificar normalidad de los residuos.
cat("Prueba de normalidad para los residuos\n")
print(shapiro.test(datos$residuos))