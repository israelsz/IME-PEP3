library(ggpubr)

# Cargar los datos.
datos <- mtcars

# Ajustar modelo con R.
modelo <- lm(mpg ~ vs, data = datos)
print(summary(modelo))

# Graficar el modelo.
p <- ggscatter(datos, x = "vs", y = "mpg", color = "blue", fill = "blue",
               xlab = "Forma del motor", ylab = "Rendimiento [millas/galón]",
               xticks.by = 1)

p <- p + geom_smooth(method = lm, se = FALSE, colour = "red")
print(p)

# Crear gráficos para evaluar el modelo.
plot(modelo)

#Graficar residuos.
residuos <- modelo$residuals
datos <- cbind(datos, residuos)
datos[["vs"]] <- factor(datos[["vs"]])

r <- ggqqplot(datos, x = "residuos", facet.by = "vs", color = "vs", 
              palette = c("blue", "red"))

print(r)