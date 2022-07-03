# Ejercicio práctico N°14
# Grupo N°2
# Integrantes:
# Christofer Rodriguez - Christian Méndez  - Israel Arias

#Importación de paquetes
if(!require(tidyverse)){
  install.packages("tidyverse",dependencies = TRUE)
  require(tidyverse)
}

if(!require(ggpubr)){
  install.packages("ggpubr",dependencies = TRUE)
  require(ggpubr)
}

if(!require(car)){
  install.packages("car",dependencies = TRUE)
  require(car)
}

if(!require(pROC)){
  install.packages("pROC",dependencies = TRUE)
  require(pROC)
}


# El siguiente paquete solo es ocupado para generar
# las matrices de confusión, lo cual es permitido
if(!require(caret)){
  install.packages("caret",dependencies = TRUE)
  require(caret)
}


# Se carga el archivo de datos CSV
datos <- read.csv2(file.choose(new = FALSE))

############################################################################
# 1. Definir la semilla a utilizar, que corresponde a los últimos cuatro
#    dígitos del RUN (sin considerar el dígito verificador) del integrante de
#     mayor edad del equipo.
############################################################################
semilla <- 0122
# Se fija la seed
set.seed(semilla)
# Nota: esta seed debió ser modificada a lo largo del código, debido
# a que más adelante al elegir variables predictoras, creaba
# modelos perfectos que no cumplían las condiciones


############################################################################
# 2. Seleccionar una muestra de 120 mujeres (si la semilla es un número par) o
#   120 hombres (si la semilla es impar), asegurando que la mitad tenga estado
#   nutricional “sobrepeso” y la otra mitad “no sobrepeso”. Dividir esta
#   muestra en dos conjuntos: los datos de 80 personas (40 con EN “sobrepeso”)
#   para utilizar en la construcción de los modelos y 40 personas
#   (20 con EN “sobrepeso”) para poder evaluarlos.
############################################################################
datos_mujeres <- datos %>% filter(Gender == 0)
# Calculo del IMC para todas las mujeres
imc_datos_mujeres <- datos_mujeres$Weight/((datos_mujeres$Height / 100) ^ 2)
# Asignación según estado nutricional
EN <- vector()
for(k in 1:260){
  if(imc_datos_mujeres[k] >= 23){ # Se ajusta el estado nutricional a 23
    EN[k] <- 1 # Sobrepeso
  }
  else{
    EN[k] <- 0 # No sobrepeso
  }
}
EN <- as.factor(EN)
datos_mujeres[["EN"]]<- EN

# Se toma muestra de 60 mujeres con sobrepeso
datos_mujeres_sobrepeso <- datos_mujeres %>% filter(EN == 1)
muestra_mujeres_sobrepeso <- sample_n(datos_mujeres_sobrepeso, 60)
# Se toma muestra de 60 mujeres sin sobrepeso
datos_mujeres_sin_sobrepeso <- datos_mujeres %>% filter(EN == 0)
# Se vuelve a setear la seed para el nuevo sample
set.seed(semilla)
muestra_mujeres_sin_sobrepeso <- sample_n(datos_mujeres_sin_sobrepeso, 60)
# Se crean los datasets de entrenamiento y prueba
entrenamiento_1 <- muestra_mujeres_sobrepeso[1:40, ]
entrenamiento_2 <- muestra_mujeres_sin_sobrepeso[1:40, ]
datos_entrenamiento <- rbind(entrenamiento_1, entrenamiento_2)

prueba_1 <- muestra_mujeres_sobrepeso[41:60, ]
prueba_2 <- muestra_mujeres_sin_sobrepeso[41:60, ]
datos_prueba <- rbind(prueba_1, prueba_2)


############################################################################
# 3. Recordar las ocho posibles variables predictoras seleccionadas de forma
#    aleatoria en el ejercicio anterior.
############################################################################
indicesValidosTodos <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                    24)
# Se usa la semilla del ep anterior para poder recuperar las 8 variables
# predictoras utilizadas
set.seed(25)
indicesValidos <- sample(indicesValidosTodos,8)
#Se agrega la variable predictora EN
indicesValidos <- c(indicesValidos,26)
# Se filtran los datos para dejar solo los predictores
variables_aleatorias_entrenamiento <- datos_entrenamiento[,indicesValidos]
variables_aleatorias_prueba <- datos_prueba[,indicesValidos]
set.seed(semilla)

############################################################################
# 4. Seleccionar, de las otras variables, una que el equipo considere que
#   podría ser útil para predecir la clase EN, justificando bien esta selección.
############################################################################
# https://scielo.isciii.es/scielo.php?script=sci_arttext&pid=S1135-57272003000300005
# Considerando el estudio adjunto, es posible considerar que el diámetro 
# del antebrazo es un buen indicador respecto a la acumulación de grasa
# que este presenta, es por esto por lo que se elige como variable 
# predictora para poder predecir la clase EN.


############################################################################
# 5. Usando el entorno R y paquetes estándares, construir un modelo de regresión
#   logística con el predictor seleccionado en el paso anterior y utilizando 
#   de la muestra obtenida.
############################################################################

# Ajustar modelo con el diametro de cintura como predictor
cat("Modelo con diametro de cintura como predictor\n")
cat("--------------------------------------\n")
modelo_simple <- glm(EN ~ Forearm.Girth, family = binomial(link = "logit"),
                   data = variables_aleatorias_entrenamiento)

print(summary(modelo_simple))


############################################################################
# 6. Usando herramientas estándares para la exploración de modelos del
#   entorno R, buscar entre dos y cinco predictores de entre las variables
#   seleccionadas al azar, recordadas en el punto 3, para agregar al modelo
#   obtenido en el paso 5.
############################################################################

# Ajustar modelo completo.
cat("\n\n")
completo <- glm(EN ~ ., family = binomial(link = "logit"),
                data = variables_aleatorias_entrenamiento)

# Ajustar modelo con regresión escalonada.
cat("Modelo con regresión escalonada\n")
cat("--------------------------------------\n")
modelo_multiple <- step(modelo_simple, scope = list(lower = modelo_simple, upper = completo),
              direction = "both", trace = 0)

print(summary(modelo_multiple))

############################################################################
# 7. Evaluar la confiabilidad de los modelos (i.e. que tengan un buen nivel de 
#   ajuste y son generalizables) y “arreglarlos” en caso de que tengan
#   algún problema.
############################################################################

# Modelo de regresión logística simple:

# Para que un modelo sea confiable debe de en primera instancia cumplir
# las condiciones para su uso, en el caso de la regresión logística corresponden a:

# 1. Debe existir una relación lineal entre los predictores y la respuesta transformada.

# - Esta condición se comprueba, existe una relación lineal entre los predictores y
# su respuesta, dada por  la función lógica estándar usada en el modelo de regresión logística


# 2. Los residuos deben ser independientes entre sí.

# Independencia de los residuos .
set.seed(296)
cat ( " Verificación de independencia de los residuos \n" )
cat ( " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n" )
print(durbinWatsonTest(modelo_simple))

# Es posible observar que en este caso el modelo de regresión logística de una
# variable no cumple la condición de que los residuos sean independientes entre
# sí, esto debido a que el p-valor de la prueba de Durbin Watson es igual a cero,
# lo que debe ser tomado en consideración ya que reduce la confiabilidad del modelo.


# Modelo de regresión logística múltiple:

# Al igual que para el caso anterior se deben verificar las condiciones:

# 1. Debe existir una relación lineal entre los predictores y la respuesta transformada.
# - Esta condición se comprueba, existe una relación lineal entre los predictores y
# su respuesta, dada por  la función lógica estándar usada en el modelo de regresión logística

# 2. Los residuos deben ser independientes entre sí.

# Comprobar independencia de los residuos.
set.seed(296)
cat("Prueba de Durbin-Watson para autocorrelaciones ")
cat("entre errores:\n")
print(durbinWatsonTest(modelo_multiple))

# Al efectuar la prueba de Durbin Watson se obtiene un p-valor de 0.052, al ser mayor que
# el nivel de significación es posible concluir con 95% de confianza que los residuos son independientes.

# Además de las condiciones anteriores, se verificarán las condiciones
# para verificar que el modelo converja:
#1. Multicolinealidad entre los predictores.


# Comprobar la multicolinealidad.
vifs <- vif(modelo_multiple)
cat("\nVerificar la multicolinealidad:\n")
cat("- VIFs:\n")
print(vifs)
cat("- Tolerancias:\n")
print(1 / vifs)
cat("- VIF medio:", mean(vifs), "\n")

# Tomando en consideración los resultados del factor de inflación de varianza (vif) y
# la tolerancia (1/vif), no hay valores de VIF preocupantes mayor a 5 o 10, como tampoco
# hay valores preocupantes menores a 0.2 para el caso de las tolerancias, sin embargo, el VIF
# medio es mayor a 1, específicamente es 1.3685, esto podría implicar la posibilidad que el
# modelo podría estar sesgado, esto no es posible corregirlo eliminando predictores
# del modelo, ya que como es posible observar, no existe valor a vif menor a 1, por
# ende es un factor a tomar en cuenta sobre el modelo.

#2. Información incompleta.
# El modelo no presenta información incompleta, esto es posible verificarlo observando los
# datos del modelo, no se presenta ningún NA’s o similar.

#3. Separación perfecta.
# El modelo no presenta separación perfecta, la misma función glm al ajustar el modelo
# imprime un warning cuando esto sucede, este no es el caso para el modelo.


############################################################################
# 8. Usando código estándar evaluar el poder predictivo de los modelos con los
#   datos de las 40 personas que no se incluyeron en su construcción en términos
#   de sensibilidad y especificidad.
############################################################################

# Hacer predicciones para el conjunto de prueba  y entrenamiento, tanto para
# el modelo simple como para el modelo.
prediccionesModeloSimple_entrenamiento <- predict(modelo_simple, datos_entrenamiento, type = "response")
prediccionesModeloSimple_prueba <- predict(modelo_simple, datos_prueba, type = "response")
prediccionesModelomodelo_multiple_entrenamiento <- predict(modelo_multiple, datos_entrenamiento, type = "response")
prediccionesModelomodelo_multiple_prueba <- predict(modelo_multiple, datos_prueba, type = "response")

# Umbral para predecir la ocurrencia del suceso
umbral <- 0.5

# Modelo simple datos de entrenamiento
preds_simple_entrenamiento <- sapply(prediccionesModeloSimple_entrenamiento , function(p) ifelse(p >= umbral, "1", "0"))
preds_simple_entrenamiento <- factor(preds_simple_entrenamiento , levels = levels(datos_entrenamiento[["EN"]]))
ROC_simple_entrenamiento <- roc(datos_entrenamiento[["EN"]], prediccionesModeloSimple_entrenamiento)
plot(ROC_simple_entrenamiento)
# Matriz de confusión modelo simple datos de entrenamiento
matriz_simple_entrenamiento <- confusionMatrix(preds_simple_entrenamiento, datos_entrenamiento[["EN"]])
cat ( " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n" )
cat("Matriz de confusión para el modelo simple con datos de entrenamiento: \n")
print(matriz_simple_entrenamiento)

# Modelo simple datos de prueba
preds_simple_prueba <- sapply(prediccionesModeloSimple_prueba , function(p) ifelse(p >= umbral, "1", "0"))
preds_simple_prueba <- factor(preds_simple_prueba , levels = levels(datos_prueba[["EN"]]))
ROC_simple_prueba <- roc(datos_prueba[["EN"]], prediccionesModeloSimple_prueba)
plot(ROC_simple_prueba)
# Matriz de confusión modelo simple datos de prueba
matriz_simple_prueba <- confusionMatrix(preds_simple_prueba, datos_prueba[["EN"]])
cat ( " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n" )
cat("Matriz de confusión para el modelo simple con datos de prueba: \n")
print(matriz_simple_prueba)

# Modelo multiple datos de entrenamiento
preds_multiple_entrenamiento <- sapply(prediccionesModelomodelo_multiple_entrenamiento , function(p) ifelse(p >= umbral, "1", "0"))
preds_multiple_entrenamiento <- factor(preds_multiple_entrenamiento , levels = levels(datos_entrenamiento[["EN"]]))
ROC_multiple_entrenamiento <- roc(datos_entrenamiento[["EN"]], prediccionesModelomodelo_multiple_entrenamiento)
plot(ROC_multiple_entrenamiento)
# Matriz de confusión modelo multiple datos de entrenamiento
matriz_multiple_entrenamiento <- confusionMatrix(preds_multiple_entrenamiento, datos_entrenamiento[["EN"]])
cat ( " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n" )
cat("Matriz de confusión para el modelo multiple con datos de entrenamiento: \n")
print(matriz_multiple_entrenamiento)

# Modelo multiple datos de prueba
preds_multiple_prueba <- sapply(prediccionesModelomodelo_multiple_prueba , function(p) ifelse(p >= umbral, "1", "0"))
preds_multiple_prueba <- factor(preds_multiple_prueba , levels = levels(datos_prueba[["EN"]]))
ROC_multiple_prueba <- roc(datos_prueba[["EN"]], prediccionesModelomodelo_multiple_prueba)
plot(ROC_multiple_prueba)
# Matriz de confusión modelo multiple datos de prueba
matriz_multiple_prueba <- confusionMatrix(preds_multiple_prueba, datos_prueba[["EN"]])
cat ( " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n" )
cat("Matriz de confusión para el modelo multiple con datos de prueba: \n")
print(matriz_multiple_prueba)

# Observando las curvas ROC construidas para cada uno de los modelos, tanto con
# el conjunto de datos de entrenamiento como el conjunto de datos de prueba,
# es posible observar que la curva resultante se aleja bastante de la diagonal
# lo que nos indica a primera vista que estamos en presencia de un buen modelo.

# Además, para continuar evaluando el poder predictivo de los modelos, se obtuvo 
# la matriz de confusión para cada uno de los casos. Observando la matriz
# tanto del modelo simple como la del modelo múltiple con los datos de entrenamiento
# podemos notar que estos tienen una exactitud de 82,5% y 92,5%; sensibilidad 
# de 85% y 95%; especificidad de 80% y 90% respectivamente. Lo anterior nos indica
# que ambos modelos son buenos para predecir correctamente los sucesos cuando
# estamos en presencia de sucesos positivos(sobrepeso) como negativos (no sobrepeso),
# ambos poseen un alto nivel de exactitud, aunque naturalmente el modelo múltiple es 
# mejor al simple debido a que posee más predictores.

# A su vez, se obtuvieron las matrices de ambos modelos con los datos de prueba,
# obtuvo una exactitud de 75% y 77,5%; sensibilidad de 70% y 65%; especificidad de 
# 80% y 90% respectivamente. En esta ocasión podemos observar que la exactitud, 
# sensibilidad y especificidad disminuyeron tanto para el modelo simple como 
# para el modelo múltiple, en especial disminuyó la sensibilidad del modelo múltiple
# en comparación con el conjunto de datos de entrenamiento, esto puede indicar que
# el modelo podría estar un poco sobreajustado para el conjunto de entrenamiento,
# pero también de que el conjunto de prueba puede ser muy pequeño para obtener una 
# evaluación confiable, poseyendo este la mitad del tamaño que el conjunto de entrenamiento.
