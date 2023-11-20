library("e1071")
library("pROC")
library("rpart")
library("rpart.plot")
library("caret")
library("randomForest")
library(gbm)
library(neuralnet)
library(xgboost)
data = read.csv("dataset.csv")

set.seed(42)
data <- data[sample(nrow(data), 1500), ]
data <- data[1:1500,]

str(data)

data <- data[complete.cases(data),]
dim(data)

data = data[,c('sex','age','weight','height','HDL_chole','LDL_chole','triglyceride','hemoglobin','DRK_YN')]
str(data)

colnames(data)[colnames(data) == 'DRK_YN'] <- 'Y'


#Tolygus kintamieji padaromi i diskreciuosius
data$age_group <- ifelse(data$age >= 20 & data$age <= 35, '20-35',
                         ifelse(data$age > 35 & data$age <= 50, '35-50', '50+'))
data <- data[, !(colnames(data) %in% 'age')]

data$weight_group <- ifelse(data$weight >= 25 & data$weight <= 60, '30-60',
                         ifelse(data$weight > 60 & data$weight <= 85, '60-85', '85+'))
data <- data[, !(colnames(data) %in% 'weight')]

data$height_group <- ifelse(data$height >= 130 & data$height <= 160, '130-160',
                            ifelse(data$height > 160 & data$height <= 185, '160-185', '185+'))
data <- data[, !(colnames(data) %in% 'height')]

data$sex <- as.numeric(factor(data$sex))
data$age_group <- as.numeric(factor(data$age_group))
data$weight_group <- as.numeric(factor(data$weight_group))
data$height_group <- as.numeric(factor(data$height_group))
data$Y <- ifelse(data$Y == 'Y', 1, 0)

train_indices <- sample(1:nrow(data), 0.8 * nrow(data))
train_data <- data[train_indices, ]  # Training data
test_data <- data[-train_indices, ]  # Testing data

x_train <- train_data[, -which(names(train_data) == "Y")]  # Features for training
y_train <- train_data$Y  # Target variable for training

# For the testing set
x_test <- test_data[, -which(names(test_data) == "Y")]  # Features for testing
y_test <- test_data$Y 


tune.out=tune(svm ,Y~.,data=train_data ,kernel ="linear", 
              ranges =list(cost=c(0.001,0.01,0.1, 1,5,10,100),gamma=c(0.001,0.01,0.1, 1,5,10,100)))
tune_grid_svm <- expand.grid(cost = c(0.1, 1, 10, 100), gamma = c(0.1, 1, 10))
svm_model <- svm(Y ~ ., data=train_data, kernel="linear",  cost = 0.001,gamma=0.001,epsilon=0.1, scale=FALSE)
svm_model
predictions_svm <- predict(svm_model, x_test)
roc_obj_svm <- roc(test_data$Y, predictions_svm)
auc_score_svm <- auc(roc_obj_svm)
auc_score_svm
predictions_svm =  ifelse(predictions_svm > 0.5, 1, 0)
conf_matrix_svm <- confusionMatrix(data = factor(predictions_svm),
                                   reference = factor(test_data$Y))
sensitivity_svm <- conf_matrix_svm$byClass["Sensitivity"]
specificity_svm <- conf_matrix_svm$byClass["Specificity"]
print(paste("Sensitivity (True Positive Rate):", sensitivity_svm))
print(paste("Specificity (True Negative Rate):", specificity_svm))

numFolds <- trainControl(method = "cv", number = 10)
cpGrid <- expand.grid(.cp = seq(0.01, 0.5, 0.01))
train(Y ~., data = train_data, method = "rpart", trControl = numFolds, tuneGrid = cpGrid)
tree <- rpart(Y ~., data = train_data,cp=0.01)
rpart.plot(tree)
predictions_tree <- predict(tree, x_test)
roc_obj_tree <- roc(test_data$Y, predictions_tree)
auc_score_tree <- auc(roc_obj_tree)
auc_score_tree
predictions_tree =  ifelse(predictions_tree > 0.5, 1, 0)
conf_matrix_tree <- confusionMatrix(data = factor(predictions_tree),
                                   reference = factor(test_data$Y))
sensitivity_tree <- conf_matrix_tree$byClass["Sensitivity"]
specificity_tree <- conf_matrix_tree$byClass["Specificity"]
print(paste("Sensitivity (True Positive Rate):", sensitivity_tree))
print(paste("Specificity (True Negative Rate):", specificity_tree))

cpGrid <- expand.grid(mtry = c(2, 3, 4,5,7,10))
train(Y ~., data = train_data, method = "rf", trControl = numFolds, tuneGrid = cpGrid)
rf_model <- randomForest(Y ~., data = train_data,mtry=2)
predictions_rf <- predict(rf_model, x_test)
roc_obj_rf <- roc(test_data$Y, predictions_rf)
auc_score_rf <- auc(roc_obj_rf)
auc_score_rf
predictions_rf =  ifelse(predictions_rf > 0.5, 1, 0)
conf_matrix_rf <- confusionMatrix(data = factor(predictions_rf),
                                    reference = factor(test_data$Y))
sensitivity_rf <- conf_matrix_rf$byClass["Sensitivity"]
specificity_rf <- conf_matrix_rf$byClass["Specificity"]
print(paste("Sensitivity (True Positive Rate):", sensitivity_rf))
print(paste("Specificity (True Negative Rate):", specificity_rf))

param_grid <- expand.grid(
  nrounds = 100,
  max_depth = c(3, 5, 7),
  eta = c(0.1, 0.01, 0.001),
  gamma = 0,
  colsample_bytree = c(0.7, 0.8, 0.9),
  min_child_weight = 1,
  subsample = c(0.7, 0.8, 0.9)
)

ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
train(
  x = as.matrix(train_data),
  y = y_train,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = param_grid,
  metric = "AUC"
)
train(
  x = as.matrix(train_data),
  y = y_train,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = params,
  metric = "AUC")

params <- list(
  eval_metric = "auc",
  max_depth = 3,
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.9,
  min_child_weight = 1,
  subsample = 0.8,
  nthread = 2
)

# Train the XGBoost model
boosting_model <- xgboost(
  data = as.matrix(train_data),
  label = train_data$Y,
  nrounds = 100,
  params = params
)

predictions_boosting <- predict(boosting_model, as.matrix(test_data))
roc_obj_boosting <- roc(test_data$Y, predictions_boosting)
auc_score_boosting <- auc(roc_obj_boosting)
auc_score_boosting
predictions_boosting =  ifelse(predictions_boosting > 0.5, 1, 0)
conf_matrix_boosting <- confusionMatrix(data = factor(predictions_boosting),
                                  reference = factor(test_data$Y))
sensitivity_boosting <- conf_matrix_boosting$byClass["Sensitivity"]
specificity_boosting <- conf_matrix_boosting$byClass["Specificity"]
print(paste("Sensitivity (True Positive Rate):", sensitivity_boosting))
print(paste("Specificity (True Negative Rate):", specificity_boosting))

nn_model <- neuralnet(Y ~ ., data = train_data,  hidden=3, threshold = 0.05, act.fct = "logistic",linear.output = TRUE)
plot(train_data)
predictions_nn <- predict(nn_model, test_data)
roc_obj_nn <- roc(test_data$Y, predictions_nn)
auc_score_nn <- auc(roc_obj_nn)
auc_score_nn
predictions_nn =  ifelse(predictions_nn > 0.5, 1, 0)
conf_matrix_nn <- confusionMatrix(data = factor(predictions_nn),
                                        reference = factor(test_data$Y))
sensitivity_nn <- conf_matrix_nn$byClass["Sensitivity"]
specificity_nn <- conf_matrix_nn$byClass["Specificity"]
print(paste("Sensitivity (True Positive Rate):", sensitivity_nn))
print(paste("Specificity (True Negative Rate):", specificity_nn))


