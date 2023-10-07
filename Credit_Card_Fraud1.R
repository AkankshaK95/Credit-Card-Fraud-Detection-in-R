# CLASSIFICATION PROJECT ON CREDIT CARD FRAUD DETECTION

# Algorithms used :
# 1) C5.0
# 1) Random Forest
# 1) SVM

getwd()
install.packages("readr")
install.packages("dplyr")
install.packages("tidyverse")
install.packages("caret")
install.packages("GGally")
install.packages("stringr")
install.packages("rattle")
install.packages("pROC")
install.packages("ROCR")
install.packages("e1071")
install.packages("randomForest")

library(readr)
library(dplyr)
library(tidyverse)
library(caret)
library(GGally)
library(randomForest)
library(stringr)
library(rattle)
library(pROC)
library(ROCR)
library(e1071)

set.seed(317)

fraud_raw <- read_csv("F:/R Programming/creditcard.csv")
glimpse(fraud_raw)



'------------------------------------------ Data Pre-processing ---------------------------------------'



fraud_df <- fraud_raw %>%
  mutate(name_orig_first = str_sub(nameOrig,1,1)) %>%
  mutate(name_dest_first = str_sub(nameDest, 1, 1)) %>%
  select(-nameOrig, -nameDest)

unique(fraud_df$name_dest_first)

fraud_df$name_dest_first <- as.factor(fraud_df$name_dest_first)
table(fraud_df$name_dest_first)

unique(fraud_df$name_orig_first)

fraud_df2 <- fraud_df %>%
  select(-name_orig_first, -isFlaggedFraud) %>%
  select(isFraud, type, step, everything())
glimpse(fraud_df2)

fraud_df2$type <- as.factor(fraud_df2$type)
fraud_df2$isFraud <- as.factor(fraud_df2$isFraud)

fraud_df2$isFraud <- recode_factor(fraud_df2$isFraud, `0` = "No", `1` = "Yes")

summary(fraud_df2)

fraud_trans <- fraud_df2 %>%
  filter(isFraud == "Yes") 
summary(fraud_trans)

fraud_df3 <- fraud_df2 %>%
  filter(type %in% c("CASH_OUT", "TRANSFER")) %>%
  filter(name_dest_first == "C") %>%
  filter(amount <= 10000000) %>%
  select(-name_dest_first)

summary(fraud_df3)

not_fraud <- fraud_df3 %>%
  filter(isFraud == "No") %>%
  sample_n(8213)

is_fraud <- fraud_df3 %>%
  filter(isFraud == "Yes")

full_sample <- rbind(not_fraud, is_fraud) %>%
  arrange(step)

ggplot(full_sample, aes(x = step, col = isFraud)) + 
  geom_histogram(bins = 743)


ggplot(is_fraud, aes(x = step)) + 
  geom_histogram(bins = 743)

ggpairs(full_sample)

ggplot(full_sample, aes(type, amount, color = isFraud)) +
  geom_point(alpha = 0.01) + 
  geom_jitter()


summary(full_sample)

preproc_model <- preProcess(fraud_df3[, -1], 
                            method = c("center", "scale", "nzv"))


fraud_preproc <- predict(preproc_model, newdata = fraud_df3[, -1])

fraud_pp_w_result <- cbind(isFraud = fraud_df3$isFraud, fraud_preproc)


summary(fraud_pp_w_result)

fraud_numeric <- fraud_pp_w_result %>%
  select(-isFraud, -type)

high_cor_cols <- findCorrelation(cor(fraud_numeric), cutoff = .75, verbose = TRUE, 
                                 names = TRUE, exact = TRUE)

high_cor_removed <- fraud_pp_w_result %>%
  select(-newbalanceDest)

fraud_numeric <- high_cor_removed %>%
  select(-isFraud, -type)
comboInfo <- findLinearCombos(fraud_numeric)
comboInfo

model_df <-high_cor_removed

is_fraud <- model_df %>%
  filter(isFraud == "Yes")

not_fraud <- model_df %>%
  filter(isFraud == "No") %>%
  sample_n(8213)

# To mix up the sample set, arrange by `step`
model_full_sample <- rbind(is_fraud, not_fraud) %>%
  arrange(step)

in_train <- createDataPartition(y = model_full_sample$isFraud, p = .75, 
                                list = FALSE) 
train <- model_full_sample[in_train, ] 
test <- model_full_sample[-in_train, ] 


# creating train and test datasets 
in_train <- createDataPartition(y = model_full_sample$isFraud, p = .75, 
                                list = FALSE) 
train <- model_full_sample[in_train, ] 
test <- model_full_sample[-in_train, ] 




'------------------------------------------------ Classification Models------------------------------------------------------------'



# 1. C5.0 MODEL -> Decision trees and rule-based models for pattern recognition.

grid <- expand.grid( .winnow = c(FALSE), 
                     .trials=c(50, 100, 150, 200), 
                     .model="tree" )


start_time <- Sys.time()

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)


c5_model <- train(isFraud ~ .,
                  data = train, 
                  method = "C5.0",
                  trControl = fitControl, 
                  metric = "ROC",
                  tuneGrid = grid, 
                  verbose = FALSE)
end_time <- Sys.time()
end_time - start_time
##Time difference of 2.277521 mins
print(c5_model)


#Predict on Training set
c5_pred_train <- predict(c5_model, train)
confusionMatrix(train$isFraud, c5_pred_train, positive = "Yes")
'Accuracy : 0.9946; 
Sensitivity : 0.9938
Specificity : 0.9954'


#Predict on Test set
c5_pred_test <- predict(c5_model, test)
confusionMatrix(test$isFraud, c5_pred_test, positive = "Yes")
'Accuracy : 0.983; 
Sensitivity : 0.9801
Specificity : 0.9858'

big_no_sample <- model_df %>%
  filter(isFraud == "No") %>%
  sample_n(100000)

#Predict on Big No-Fraud dataset
start_time <- Sys.time()
c5_pred_big_no <- predict(c5_model, big_no_sample)
end_time <- Sys.time()
end_time - start_time

confusionMatrix(big_no_sample$isFraud, c5_pred_big_no, positive = "Yes")

c5_probs <- predict(c5_model, test, type = "prob")
c5_ROC <- roc(response = test$isFraud, 
              predictor = c5_probs$Yes, 
              levels = levels(test$isFraud))

plot(c5_ROC, col = "red")
auc(c5_ROC)
##Area under the curve: 0.9981
'-------------------------------------------------------------------------------------------------------------------'



#2.   Random Forest model


grid <- expand.grid(.mtry = 5, .ntree = seq(25, 150, by = 25))

start_time <- Sys.time()
trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid")

rf_model <- train(isFraud ~ ., 
                  data = train, 
                  method="rf", 
                  metric = "Accuracy", 
                  TuneGrid = grid, 
                  trControl=trControl)
end_time <- Sys.time()
end_time - start_time
##Time difference of 3.119016 mins

print(rf_model$finalModel)
plot(rf_model$finalModel)

#Predict on Training set
rf_train_pred <- predict(rf_model, train)
confusionMatrix(train$isFraud, rf_train_pred, positive = "Yes")
'Accuracy : 1; Overfit
Sensitivity : 1.0
Specificity : 1.0'

#Predict on Test set
rf_test_pred <- predict(rf_model, test)
confusionMatrix(test$isFraud, rf_test_pred, positive = "Yes")
'Accuracy : 0.9803; not bad.
Sensitivity : 0.9709
Specificity : 0.9901'

#Predict on Big No-Fraud dataset
big_no_sample <- model_df %>%
  filter(isFraud == "No") %>%
  sample_n(100000)

start_time <- Sys.time()
rf_big_no_pred <- predict(rf_model, big_no_sample)
end_time <- Sys.time()
end_time - start_time

'Accuracy : 0.968
Sensitivity : 0.00000
Specificity : 1.00000
False-positives: 3197'

confusionMatrix(big_no_sample$isFraud, rf_big_no_pred, positive = "Yes")

rf_probs <- predict(rf_model, test, type = "prob")
rf_ROC <- roc(response = test$isFraud, 
              predictor = rf_probs$Yes, 
              levels = levels(test$isFraud))

plot(rf_ROC, col = "green")
auc(rf_ROC)
##Area under the curve: 0.9977
'-----------------------------------------------------------------------------------------------------------------------'



#3.   SVM model


start_time <- Sys.time()
trControl <- trainControl(method = "cv",
                          number = 10,
                          classProbs = TRUE)
##fitControl <- trainControl(method = "repeatedcv",
##                           number = 10,
##                         repeats = 10,
                           ## Estimate class probabilities
##                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
##                           summaryFunction = twoClassSummary)
svm_model <- train(isFraud ~ ., 
                   data = train, 
                   method = "svmRadial",   # Radial kernel
                   tuneLength = 3,  # 3 values of the cost function
                   metric="ROC",
                   trControl=trControl)

end_time <- Sys.time()
end_time - start_time
## Time difference of 30.57211 mins

print(svm_model$finalModel)

#Predict on Training set
svm_train_pred <- predict(svm_model, train)
confusionMatrix(train$isFraud, svm_train_pred, positive = "Yes")

"Accuracy : 0.9778
Sensitivity : 0.9570
Specificity : 0.9803"

#Predict on Test set
svm_test_pred <- predict(svm_model, test)
confusionMatrix(test$isFraud, svm_test_pred, positive = "Yes")

"Accuracy : 0.9713
Sensitivity : 0.9160  
Specificity : 0.9783"

big_no_sample <- model_df %>%
  filter(isFraud == "No") %>%
  sample_n(100000)
#Predict on Big No-Fraud dataset
start_time <- Sys.time()
svm_big_no_pred <- predict(svm_model, big_no_sample)
end_time <- Sys.time()
end_time - start_time
## Time difference of 0.2067931 secs
confusionMatrix(big_no_sample$isFraud, svm_big_no_pred, positive = "Yes")
svm_probs <- predict(svm_model, test, type = "prob")
svm_ROC <- roc(response = test$isFraud, 
               predictor = svm_probs$Yes, 
               levels = levels(test$isFraud))

plot(svm_ROC, col = "black")

"Accuracy : 0.9341
Sensitivity : 0.00000
Specificity : 1.00000
False-positives: 6588
Run-time: 49.61945 secs"

auc(svm_ROC)
##Area under the curve: 0.9763

plot(rf_ROC, col = "green")
plot(svm_ROC, col = "black", add = TRUE)
plot(c5_ROC, col = "red", add = TRUE)
'-------------------------------------------------END--------------------------------------------------'