library(dummies)

# Crear una matrz de datos.
sujeto <- 1:10
sexo <- c("F", "F", "M", "M", "M", "M", "F", "M", "F", "F")
tipo <- c("B", "D", "A", "B", "A", "C", "D", "D", "D", "A")
valor <- c(1.68, 2.79, 1.92, 2.26, 2.1, 2.63, 2.19, 3.62, 2.76, 1.26)
datos <- data.frame(sujeto , sexo , tipo, valor)

# Crear variables artificiales.
datos.dummy <- dummy.data.frame(datos , drop = TRUE)
datos.dummy[["sexoF"]] <- NULL
datos.dummy[["tipoA"]] <- NULL

# Crear modelos lineales.
m1 <- lm(valor ~ sexo + tipo, datos)
print(m1)

m2 <- lm(valor ~ sexoM + tipoB + tipoC + tipoD, datos.dummy)
print(m2)
