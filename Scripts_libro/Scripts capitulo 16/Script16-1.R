library(pROC)
library(caret)

set.seed(1313)

# Cargar los datos.
datos <- mtcars
datos$am <- factor(datos$am)

# Separar conjuntos de entrenamiento y prueba.
n <- nrow(datos)
n_entrenamiento <- floor(0.8 * n)
muestra <- sample.int(n = n, size = n_entrenamiento, replace = FALSE)
entrenamiento <- datos[muestra, ]
prueba  <- datos[-muestra, ]

# Ajustar modelo.
modelo <- glm(am ~ wt, family = binomial(link = "logit"), data = entrenamiento)
print(summary(modelo))

# Evaluar el modelo con el conjunto de entrenamiento.
cat("Evaluación del modelo a partir del conjunto de entrenamiento:\n")
probs_e <- predict(modelo, entrenamiento, type = "response")

umbral <- 0.5
preds_e <- sapply(probs_e, function(p) ifelse(p >= umbral, "1", "0"))
preds_e <- factor(preds_e, levels = levels(datos[["am"]]))

ROC_e <- roc(entrenamiento[["am"]], probs_e)
plot(ROC_e)

matriz_e <- confusionMatrix(preds_e, entrenamiento[["am"]])
print(matriz_e)

# Evaluar el modelo con el conjunto de prueba.
cat("Evaluación del modelo a partir del conjunto de prueba:\n")
probs_p <- predict(modelo, prueba, type = "response")

preds_p <- sapply(probs_p, function(p) ifelse(p >= umbral, "1", "0"))
preds_p <- factor(preds_p, levels = levels(datos[["am"]]))

ROC_p <- roc(prueba[["am"]], probs_p)
plot(ROC_p)

matriz_p <- confusionMatrix(preds_p, prueba[["am"]])
print(matriz_p)