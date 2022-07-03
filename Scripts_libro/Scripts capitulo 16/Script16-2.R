library(caret)

set.seed(1313)

# Cargar los datos.
datos <- mtcars
datos$am <- factor(datos$am)

# Ajustar modelo usando validación cruzada de 5 pliegues.
modelo <- train(am ~ wt, data = entrenamiento, method = "glm",
                family = binomial(link = "logit"),
                trControl = trainControl(method = "cv", number = 5, 
                                         savePredictions = TRUE))

print(summary(modelo))

# Evaluar el modelo 
cat("Evaluación del modelo basada en validación cruzada:\n")
matriz <- confusionMatrix(modelo$pred$pred, modelo$pred$obs)
print(matriz)