# Crear un data frame con una variable dicotómica.
alumno <- 1:5
sexo <- factor(c("F", "M", "F","F", "M"))
datos <- data.frame(alumno, sexo)

# Crear una variable indicadora para sexo, con valor 0
# para hombres y 1, para mujeres.
es_mujer <- rep(1, length(sexo))
es_mujer[sexo == "M"] <- 0

# Reemplazar la variable sexo por lavariable indicadora.
datos <- cbind(datos, es_mujer)
datos[["sexo"]] <- NULL