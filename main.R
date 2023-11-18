library("e1071")
library("pROC")
library("rpart")
library("rpart.plot")
library("caret")
library("randomForest")
library(gbm)
library(neuralnet)

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

tune_grid_svm <- expand.grid(cost = c(0.1, 1, 10, 100), gamma = c(0.1, 1, 10))
svm_model <- svm(Y ~ ., data=train_data, kernel="linear",  ranges = tune_grid, scale=FALSE)
svm_model
predictions_svm <- predict(svm_model, x_test)
roc_obj_svm <- roc(test_data$Y, predictions_svm)
auc_score_svm <- auc(roc_obj_svm)
auc_score_svm

ctrl <- trainControl(method = "cv",
                     number = 5)

grid <- expand.grid(cp = seq(0.01, 0.1, by = 0.01)) 

tree <- rpart(Y ~., data = train_data)
rpart.plot(tree)
predictions_tree <- predict(tree, x_test)
roc_obj_tree <- roc(test_data$Y, predictions_tree)
auc_score_tree <- auc(roc_obj_tree)
auc_score_tree

rf_model <- randomForest(Y ~., data = train_data)
predictions_rf <- predict(rf_model, x_test)
roc_obj_rf <- roc(test_data$Y, predictions_rf)
auc_score_rf <- auc(roc_obj_rf)
auc_score_rf

boosting_model <- gbm(Y ~ ., data = train_data, distribution = "bernoulli", n.trees = 100, interaction.depth = 4)
predictions_boosting <- predict(boosting_model, x_test)
roc_obj_boosting <- roc(test_data$Y, predictions_boosting)
auc_score_boosting <- auc(roc_obj_boosting)
auc_score_boosting

nn_model <- neuralnet(Y ~ ., data = train_data,  hidden=3, threshold = 0.05, act.fct = "logistic",linear.output = TRUE)
plot(train_data)
predictions_nn <- predict(nn_model, test_data)
roc_obj_nn <- roc(test_data$Y, predictions_nn)
auc_score_nn <- auc(roc_obj_nn)
auc_score_nn



