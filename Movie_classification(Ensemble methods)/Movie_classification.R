# Movie classification (Classification trees)
# the purpose of this model is triying to predict if a movie will obtain a Star Tech Oscar
# Importing libraries and dataset
df <- read.csv("Movie_classification.csv", header = TRUE)
View(df)
summary(df)
# Missing values treatment
df$Time_taken[is.na(df$Time_taken)] <- mean(df$Time_taken, na.rm = TRUE)
summary(df)
#Training and testing data split
install.packages("caTools")
library(caTools)
set.seed(0)
split <- sample.split(df, SplitRatio = 0.8)
train <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)

#Training classification tree
install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
clftree <- rpart(formula = Start_Tech_Oscar~., data = df, method = "class", control = rpart.control(maxdepth = 3))
#Plotting the tree
rpart.plot(clftree, box.palette = "RdBu", digits = -3)
test$pred <- predict(clftree, test, type = "class")
confusion_matrix <- table(test$Start_Tech_Oscar, test$pred)
confusion_matrix
acc <- sum(diag(confusion_matrix))/ length(test$pred)
acc
#I will do the same what I did in python file, I am going to do a function to searche the best tree classifier model
SearchBestTree <- function(depths = 3){
  accuracies <- c()
  for(depth in seq(1: depths)){
    set.seed(0)
    clftree <- rpart(formula = Start_Tech_Oscar~., data = df, method = "class", control = rpart.control(maxdepth = depth))
    test$pred <- predict(clftree, test, type = "class")
    confusion_matrix <- table(test$Start_Tech_Oscar, test$pred)
    confusion_matrix
    acc <- sum(diag(confusion_matrix))/ length(test$pred)
    accuracies <- append(accuracies, acc)
  }
  plot(x = seq(1: depths), y = accuracies)
  max_acc <- max(accuracies)
  print(max_acc)
  pos_max_acc <- which.max(accuracies)
  print(paste("position is", pos_max_acc))
}
SearchBestTree(30) #Result of the position is 11

clftree11 <- rpart(formula = Start_Tech_Oscar~., data = df, method = "class", control = rpart.control(maxdepth = 11))
#Plotting the tree
rpart.plot(clftree11, box.palette = "RdBu", digits = -3)
test$pred11 <- predict(clftree11, test, type = "class")
confusion_matrix11 <- table(test$Start_Tech_Oscar, test$pred11)
confusion_matrix11
acc11 <- sum(diag(confusion_matrix11))/ length(test$pred11)
acc11
# In this case, I can obtain an accuracy of 77.57%

#Boosting
#Ada Boost
library(adabag)
train$Start_Tech_Oscar <- as.factor(train$Start_Tech_Oscar)
set.seed(0)
adaboost <- boosting(Start_Tech_Oscar~., data = train, boos = TRUE, mfinal = 1000 )
predada <- predict(adaboost, test)
table(predada$class, test$Start_Tech_Oscar)
View(predada)
t1 <- adaboost$trees[[1]]
t1
rpart.plot(t1)
text(t1, pretty = 100)
(27+40)/107

#XG Boost
install.packages("xgboost")
library(xgboost)
trainY <- train$Start_Tech_Oscar == 1
trainX <- model.matrix(Start_Tech_Oscar~.-1, data = train)
testY <- test$Start_Tech_Oscar == 1
testX <- model.matrix(Start_Tech_Oscar~.-1, data = test)
testX <- testX[,-23]
testX <- testX[,-22]
testX <- testX[,-12]
trainX <- trainX[,-12]
Xmatrix <- xgb.DMatrix(data = trainX, label = trainY)
Xmatrix_t <- xgb.DMatrix(data = testX, label = testY)
XgBosting <- xgboost(data = Xmatrix,
                     nrounds = 10000, #max number of boosting iterations
                     objective = "multi:softmax", eta = 0.01, num_class = 2, max_depth = 1000)
xgpred <- predict(XgBosting, Xmatrix_t)
table(testY, xgpred)
72/113
