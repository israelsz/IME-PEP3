library(ggpubr)

# Cargar los datos.
datos <- mtcars

# Ajustar modelo con R.
modelo <- lm(mpg ~ wt, data = datos)
print(summary(modelo))

# Graficar el modelo.
p <- ggscatter(datos, x = "wt", y = "mpg", color = "blue", fill = "blue",
               xlab = "Peso [lb x 1000]", ylab = "Rendimiento [millas/galón]")

p <- p + geom_smooth(method = lm, se = FALSE, colour = "red")
print(p)

# Crear gráficos para evaluar el modelo.
plot(modelo)

# Ingresar algunas instancias artificiales.
mpg <- c(23.714, 19.691, 19.242, 12.430, 10.090, 9.565, 18.171, 26.492, 7.054,
         24.447, 15.683, 17.403, 13.465, 18.850, 29.493)

wt <- c(2.973, 4.532, 2.332, 3.016, 4.220, 4.286, 2.580, 3.084, 3.816, 2.775,
        3.251, 3.013, 4.951, 2.644, 2.218)

nuevos <- data.frame(mpg, wt)

# Usar el modelo para predecir el rendimiento de los nuevos  y ver los 
# residuos resultantes.
predicciones <- predict(modelo, nuevos)
residuos <- nuevos$mpg - predicciones
nuevos <- data.frame(nuevos, residuos)

r <- ggscatter( nuevos, x = "wt", y = "residuos", color = "blue",
                fill = "blue", xlab = "Peso [lb * 1000]", ylab = "Residuo")

r <- r + geom_hline(yintercept = 0, colour = "red")
print(r)