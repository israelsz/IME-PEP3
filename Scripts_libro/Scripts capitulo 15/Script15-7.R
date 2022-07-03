# Cargar datos.
datos <- mtcars

# Ajustar modelo.
modelo <- lm(mpg ~ wt + qsec + am, data = datos)
plot(modelo)

# Reducir matriz de datos para que solo contenga los predictores
# empleados y la respuesta.
predictores <- names(coef(modelo))[-1]
datos <- datos[, c(predictores, "mpg")]

# Construir una matriz de datos con la respuesta predicha, los
# residuos y algunas estadísticas para evaluar la influencia de
# cada observación.
resultados <- data.frame(respuesta_predicha = fitted(modelo))
resultados[["residuos_estandarizados"]] <- rstandard(modelo)
resultados[["residuos_estudiantizados"]] <-rstudent(modelo)
resultados[["distancia_Cook"]] <- cooks.distance(modelo)
resultados[["dfbeta"]] <- dfbeta(modelo)
resultados[["dffit"]] <- dffits(modelo)
resultados[["apalancamiento"]] <- hatvalues(modelo)
resultados[["covratio"]] <- covratio(modelo)

cat("Identificación de valores atípicos:\n")
# Observaciones con residuos estandarizados fuera del 95% esperado.
sospechosos1 <- which(abs(
  resultados[["residuos_estandarizados"]]) > 1.96)

cat("- Residuos estandarizados fuera del 95% esperado:",
    sospechosos1, "\n")

# Observaciones con distancia de Cook mayor a uno.
sospechosos2 <- which(resultados[["cooks.distance"]] > 1)

cat("- Residuos con una distancia de Cook alta:",
    sospechosos2, "\n")

# Observaciones con apalancamiento mayor igual al doble del
# apalancamiento promedio.

apal_medio <- (ncol(datos) + 1) / nrow(datos)
sospechosos3 <- which(resultados[["apalancamiento"]] > 2 * apal_medio)

cat("- Residuos con apalancamiento fuera de rango:",
    sospechosos3, "\n")

# Observaciones con DFBeta mayor o igual a 1.
sospechosos4 <- which(apply(resultados[["dfbeta"]] >= 1, 1, any))
names(sospechosos4) <- NULL

cat("- Residuos con DFBeta >= 1:",
    sospechosos4, "\n")

# Observaciones con razón de covarianza fuera de rango.
inferior <- 1 - 3 * apal_medio
superior <- 1 + 3 * apal_medio
sospechosos5 <- which(resultados[["covratio"]] < inferior |
                        resultados[["covratio"]] > superior)

cat("- Residuos con razón de covarianza fuera de rango:",
    sospechosos5, "\n")

# Resumen de valores sospechosos.
sospechosos <- c(sospechosos1, sospechosos2, sospechosos3,
                 sospechosos4, sospechosos5)

sospechosos <- sort(unique(sospechosos))

cat("\nResumen de valores sospechosos:\n")
cat("Apalancamiento promedio:", apal_medio, "\n")

cat("Intervalo razón de covarianza: [", inferior, "; ",
    superior, "]\n\n", sep = "")

print(round(resultados[sospechosos, c("distancia_Cook", "apalancamiento",
                                      "covratio")], 3))