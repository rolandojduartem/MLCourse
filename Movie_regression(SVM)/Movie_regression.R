# Movie regression using SVM
#In this analysis, I will predict the collection of the movie using SVM
#Practice

#Importing dataset
df <- read.csv("Movie_regression.csv", header = TRUE)
View(df)
summary(df)

# Missing values tratment
df$Time_taken[is.na(df$Time_taken)] = mean(df$Time_taken, na.rm = TRUE)
summary(df)

#Training and testing data split
install.packages("caTools")
require(caTools)
set.seed(0)
split <- sample.split(df, SplitRatio = 0.8)
train <- subset(df, split == T)
test <- subset(df, split == F)

#Linear SVM
install.packages("e1071")
require(e1071)
set.seed(0)
svmfit <- svm(Collection~., data = train, kernel = "linear",
              cost = 1, scale = T)
summary(svmfit)
#Predicting using kernel linear
y_pred_lin <- predict(svmfit, test)
mean((test$Collection - y_pred_lin)^2) #Mean squared error
#Support vectors
svmfit$index
#Linear hyperparameters (best cost value)
set.seed(0)
tune.out <- tune(svm, Collection~., data = train, kernel = "linear",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100)), scale = T)
bestmod <- tune.out$best.model
summary(bestmod)

y_bestL <- predict(bestmod, test)
MSE2L <- mean((test$Collection - y_bestL)^2)

#Polynomial SVM
set.seed(0)
svmfitP <- svm(Collection~., data = train, kernel = "polynomial",
               cost = 1, degree = 2, scale = T)
#Predicting using kernel polynomial
y_pre_poly <- predict(svmfitP, test)
mean((test$Collection - y_pre_poly)^2)
#Support vectors
svmfitP$index
#Polynomial hyperparameters (best cost value)
set.seed(0)
tune.outP <- tune(svm, Collection~., data = train, cross = 4, 
                  kernel = "polynomial",
                  ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100), degree = c(0.5, 1, 2,3,4,5)),
                  scale = T)
bestmodP <- tune.outP$best.model
summary(bestmodP)

y_bestP <- predict(bestmodP, test)
MSE2P <- mean((test$Collection - y_bestP)^2)

#Radial SVM
set.seed(0)
svmfitR <- svm(Collection~., data = train, kernel = "radial",
               cost = 1, gamma = 1, scale = T)
#Predicting using kernel radial
y_pre_rad <- predict(svmfitR, test)
mean((test$Collection - y_pre_rad)^2)
#Support vectors
svmfitR$index
#Linear hyperparameters (best cost value)
set.seed(0)
tune.outR <- tune(svm, Collection~. ,data = train, cross = 4, 
                  kernel = "radial",
                  ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100), 
                                gamma = c(0.01, 0.1, 0.5, 1, 2,3,4, 10, 50)),
                  scale = T)
bestmodR <- tune.outR$best.model
summary(bestmodR)

y_bestR <- predict(bestmodR, test)
MSE2R <- mean((test$Collection - y_bestR)^2)
