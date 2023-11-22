library("e1071")
library("pROC")
library("rpart")
library("rpart.plot")
library("caret")
library("randomForest")
library(gbm)
library(neuralnet)
library(xgboost)
library(nnet)

data = read.csv("dataset.csv")

set.seed(123)
data <- data[sample(nrow(data), 1500), ]

str(data)

data <- data[complete.cases(data),]
dim(data)

data = data[,c('sex','age','weight','height','HDL_chole','LDL_chole','triglyceride','hemoglobin','DRK_YN')]
str(data)
colnames(data)[colnames(data) == "DRK_YN"] <- "Y"

#Tolygus kintamieji padaromi i diskreciuosius
data$age_group <- ifelse(data$age >= 20 & data$age <= 35, '20-35',
                         ifelse(data$age > 35 & data$age <= 50, '35-50', '50+'))
data <- data[, !(colnames(data) %in% 'age')]

data$weight_group <- ifelse(data$weight >= 25 & data$weight <= 60, '30-60',
                         ifelse(data$weight > 60 & data$weight <= 85, '60-85', '85+'))
data <- data[, !(colnames(data) %in% 'weight')]

data$height_group <- ifelse(data$height >= 130 & data$height <= 160, '130-160',
                            ifelse(data$height > 160 & data$height <= 175, '160-175', '175+'))
data <- data[, !(colnames(data) %in% 'height')]

hist(data$HDL_chole)
hist(data$LDL_chole)
hist(data$triglyceride)
hist(data$hemoglobin)
barplot(table(data$sex))
barplot(table(data$age_group))
barplot(table(data$weight_group))
barplot(table(data$height_group))

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
time_taken <- system.time({
  svm_model <- svm(Y ~ ., data = train_data, kernel = "linear",
                   cost = 0.001, gamma = 0.001, epsilon = 0.1, scale = FALSE)
})

time_taken_svm <- time_taken[["elapsed"]]
predictions_svm <- predict(svm_model, x_test)
roc_obj_svm <- roc(test_data$Y, predictions_svm)
auc_score_svm <- auc(roc_obj_svm)
predictions_svm =  ifelse(predictions_svm > 0.5, 1, 0)
conf_matrix_svm <- confusionMatrix(data = factor(predictions_svm),
                                   reference = factor(test_data$Y))
sensitivity_svm <- conf_matrix_svm$byClass["Sensitivity"]
specificity_svm <- conf_matrix_svm$byClass["Specificity"]

numFolds <- trainControl(method = "cv", number = 10)
cpGrid <- expand.grid(.cp = c(0.01,0.05,0.1,0.25, 0.5,0.75,1,2,5,10,100))
train(Y ~., data = train_data, method = "rpart", trControl = numFolds, tuneGrid = cpGrid)
time_taken <- system.time({tree <- rpart(Y ~., data = train_data,cp=0.01)})
time_taken_tree <- time_taken[["elapsed"]]
predictions_tree <- predict(tree, x_test)
roc_obj_tree <- roc(test_data$Y, predictions_tree)
auc_score_tree <- auc(roc_obj_tree)
predictions_tree =  ifelse(predictions_tree > 0.5, 1, 0)
conf_matrix_tree <- confusionMatrix(data = factor(predictions_tree),
                                   reference = factor(test_data$Y))
sensitivity_tree <- conf_matrix_tree$byClass["Sensitivity"]
specificity_tree <- conf_matrix_tree$byClass["Specificity"]

cpGrid <- expand.grid(mtry = c(2, 3, 4,5,7,10))
train(Y ~., data = train_data, method = "rf", trControl = numFolds, tuneGrid = cpGrid)
time_taken <- system.time({rf_model <- randomForest(Y ~., data = train_data,mtry=2)})
time_taken_rf <- time_taken[["elapsed"]]
predictions_rf <- predict(rf_model, x_test)
roc_obj_rf <- roc(test_data$Y, predictions_rf)
auc_score_rf <- auc(roc_obj_rf)
predictions_rf =  ifelse(predictions_rf > 0.5, 1, 0)
conf_matrix_rf <- confusionMatrix(data = factor(predictions_rf),
                                    reference = factor(test_data$Y))
sensitivity_rf <- conf_matrix_rf$byClass["Sensitivity"]
specificity_rf <- conf_matrix_rf$byClass["Specificity"]

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
boosting_model <- train(
  x = as.matrix(train_data),
  y = as.factor(y_train),
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = param_grid,
  eval_metric = "auc"
)

time_taken <- system.time({boosting_model <- xgboost(data = as.matrix(train_data), label = y_train, nrounds = 100, max_depth = 3, 
eta = 0.001, gamma = 0, colsample_bytree = 0.7, min_child_weight = 1, subsample = 0.7, eval_metric = "auc")})
time_taken_boosting <- time_taken[["elapsed"]]
predictions_boosting <- predict(boosting_model, as.matrix(test_data))
roc_obj_boosting <- roc(y_test, as.numeric(predictions_boosting))
auc_score_boosting <- auc(roc_obj_boosting)
predictions_boosting =  ifelse(predictions_boosting > 0.5, 1, 0)
conf_matrix_boosting <- confusionMatrix(data = factor(predictions_boosting),
                                  reference = factor(test_data$Y))
sensitivity_boosting <- conf_matrix_boosting$byClass["Sensitivity"]
specificity_boosting <- conf_matrix_boosting$byClass["Specificity"]

param_grid <- expand.grid(
  size1 = c(5, 10),
  size2 = c(3, 7),
  decay = c(0.1, 0.01),
  learningrate = c(0.01, 0.001)
)

# Create train control
control <- trainControl(method = "cv", number = 5)
grid <- expand.grid(
  size=c(3, 5, 7),
  decay = c(0.1,0.05, 0.01)
)

nn_model <- train(as.factor(Y) ~ ., 
                  data = train_data, 
                  method = "nnet", 
                  trControl = control,
                  tuneGrid = grid,
                  )

time_taken <- system.time({nn_model <- nnet(Y ~., data = train_data, size = 5,decay = 0.01)})
time_taken_nn <- time_taken[["elapsed"]]
predictions_nn <- predict(nn_model, test_data)
roc_obj_nn <- roc(as.factor(test_data$Y), as.numeric(predictions_nn))
auc_score_nn <- auc(roc_obj_nn)
predictions_nn =  ifelse(predictions_nn > 0.5, 1, 0)
conf_matrix_nn <- confusionMatrix(data = factor(predictions_nn),
                                        reference = factor(test_data$Y))
sensitivity_nn <- conf_matrix_nn$byClass["Sensitivity"]
specificity_nn <- conf_matrix_nn$byClass["Specificity"]