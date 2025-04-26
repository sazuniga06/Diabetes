
install.packages(c("mlbench", "caret", "rpart", "rpart.plot", "pROC", "ROCR"))
library(mlbench)
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)
library(ROCR)

data(PimaIndiansDiabetes)
dataSet <- PimaIndiansDiabetes

print(table(dataSet$diabetes))
print(prop.table(table(dataSet$diabetes)))

set.seed(123)
trainIndex <- createDataPartition(dataSet$diabetes, p = 0.7, list = FALSE)
dataTrain <- dataSet[trainIndex, ]
dataTest  <- dataSet[-trainIndex, ]

modelo_logit <- glm(diabetes ~ ., data = dataTrain, family = "binomial")
summary(modelo_logit)

prob_pred_logit <- predict(modelo_logit, newdata = dataTest, type = "response")
class_pred_logit <- ifelse(prob_pred_logit > 0.5, "pos", "neg")
class_pred_logit <- as.factor(class_pred_logit)


modelo_tree <- rpart(diabetes ~ ., data = dataTrain, method = "class")
rpart.plot(modelo_tree)


prob_pred_tree <- predict(modelo_tree, newdata = dataTest, type = "prob")[,2]
class_pred_tree <- predict(modelo_tree, newdata = dataTest, type = "class")


roc_logit <- roc(dataTest$diabetes, prob_pred_logit)
plot(roc_logit, main = "ROC - Regresión Logística", col = "blue", legacy.axes = TRUE)
auc_logit <- auc(roc_logit)
print(paste("AUC Logit:", round(auc_logit, 4)))

roc_tree <- roc(dataTest$diabetes, prob_pred_tree)
plot(roc_tree, main = "ROC - Árbol de Decisión", col = "darkgreen", legacy.axes = TRUE)
auc_tree <- auc(roc_tree)
print(paste("AUC Árbol:", round(auc_tree, 4)))

pred_logit <- prediction(prob_pred_logit, dataTest$diabetes)
perf_logit <- performance(pred_logit, "lift", "rpp")
plot(perf_logit, main = "Curva LIFT - Regresión Logística", col = "blue")


pred_tree <- prediction(prob_pred_tree, dataTest$diabetes)
perf_tree <- performance(pred_tree, "lift", "rpp")
plot(perf_tree, main = "Curva LIFT - Árbol de Decisión", col = "darkgreen")


plot(roc_logit, col = "blue", main = "Comparación ROC", legacy.axes = TRUE)
plot(roc_tree, col = "darkgreen", add = TRUE)
legend("bottomright", legend = c("Logística", "Árbol"), col = c("blue", "darkgreen"), lwd = 2)
