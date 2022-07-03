library(scatterplot3d)

# Cargar los datos.
datos <- mtcars

# Ajustar modelo usando validación cruzada de 5 pliegues.
modelo <- lm(mpg ~ wt + qsec, data = datos)
print(summary(modelo))

# Graficar modelo ajustado.
g <- scatterplot3d(datos$wt, datos$qsec, datos$mpg, type = "p",
                   highlight.3d = TRUE, pch = 20, xlab = "Peso [lb x 1000]",
                   ylab = "Rendimiento [millas/galón]",
                   zlab = "1/4 de milla [s]")

g$plane3d(modelo ,draw_polygon = TRUE, draw_lines = TRUE)
print(g)