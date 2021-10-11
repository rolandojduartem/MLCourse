#House price for classification model (K nearest neighbors)
#In this analysis, I will train a model if a house could be sold before 3 months (Sold column yes: 1 No: 0)
df <- read.csv("House-price.csv", header = TRUE)
View(df)
str(df)
summary(df)
#From summary, I can see n_hos_beds has 8 missing values and 
#n_hot_rooms and rainfall columns could have some outliers
boxplot(df$n_hot_rooms) #From graph, there are 2 outliers
boxplot(df$rainfall) #the wisker, at the bottom, is longer the up one
plot(df$rainfall, df$Sold) #There is an outlier
#Now, I will check categorical variables
barplot(table(df$airport)) #Two categories
barplot(table(df$waterbody)) #Four categories
barplot(table(df$bus_ter)) #One category
#Preliminar observations
#1. Missing values in n_hos_beds column
#2. Outliers in n_hot_rooms and rainfall columns
#3. Only one category in bus_ter column

#Outlier treatment
lim <- quantile(df$n_hot_rooms, 0.99) #limit
df$n_hot_rooms[df$n_hot_rooms > 3 * lim] <- 3 * lim
df$n_hot_rooms[df$n_hot_rooms > lim] #Cheking changes
lim <- quantile(df$rainfall, 0.01)
df$rainfall[df$rainfall < 0.3 * lim] <- 0.3 * lim
df$rainfall[df$rainfall < lim]

#Missing values treatment
hos_mean <- mean(df$n_hos_beds, na.rm = TRUE)
df$n_hos_beds[is.na(df$n_hos_beds)] <- hos_mean
summary(df)
#Outliers and missing values are fixed now

#Variables transformation and deletion
df[6:9]
df$avg_dist <- (df$dist1 + df$dist2 + df$dist3 + df$dist4)/4
df <- df[,-6:-9] #Deleting dist columns
View(df)
df <- df[,-13] #Deleting constant categorical variable

# Dummy variables for categorical variables
install.packages("dummies")
library(dummies)
df <- dummy.data.frame(df)
df <- df[,-8]
df <- df[,-13]

#K nearest neighbors
#Train data split
#Firstable, I will analize accuracy of Logistic regression and linear discriminant analysis
install.packages("caTools")
library(caTools)
set.seed(0)
split <- sample.split(df, SplitRatio = 0.8)
train_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

#Linear regression
train.fit <- glm(Sold~., data = train_set, family = binomial)
test.probs <- predict(train.fit, test_set, type = "response")
test.pred <- rep("NO", 120)
test.pred[test.probs > 0.5] <- "YES"
confusion_matrix <- table(test.pred, test_set$Sold)
confusion_matrix
accuracy <- (confusion_matrix[1, 1] + confusion_matrix[2, 2]) / length(test_set$Sold)
accuracy
test.probs
#Linear Discriminant Analysis
install.packages("MASS")
library(MASS)
trainlda.fit <- lda(Sold~., data = train_set)
testlda.pred <- predict(trainlda.fit, test_set)
View(testlda.pred)
confusion_matrix_lda <- table(testlda.pred$class, test_set$Sold)
confusion_matrix_lda
accuracy_lda<- (confusion_matrix_lda[1, 1] + confusion_matrix_lda[2, 2]) / length(test_set$Sold)
accuracy_lda

#K-Nearest Neighbors
install.packages("class")
library(class)
trainX <- train_set[,-16]
testX <- test_set[, -16]
trainY <- train_set$Sold
testY <- test_set$Sold
k = 9
trainX_s <- scale(trainX)
testX_s <- scale(testX)
set.seed(0)
knn.pred <- knn(trainX_s, testX_s, trainY, k = 3)
cm <- table(knn.pred, testY)
cm
acc <- (cm[1,1] + cm[2,2]) / length(testY)
acc

#Creating a function to search the best k value depending of accuracy
SearchBestKNN <- function(trainx_values, testx_values, 
                          trainy_values, testy_values, 
                          k, seed = 0){
  acc <- 0
  for (i in 1:k){
    set.seed(seed)
    temp.pred <- knn(trainx_values, testx_values, trainy_values, k = i)
    temp_matrix <- table(temp.pred, testy_values)
    temp_acc <- (temp_matrix[1,1] + temp_matrix[2,2]) / length(testy_values)
    if (temp_acc > acc){
      bestk_value <- i
      bestknn.pred <- temp.pred
      best_confusion_matrix <- temp_matrix
      acc <- temp_acc
    }
  }
  print(paste("Best k values is k =", bestk_value))
  return(list(pred = bestknn.pred, confusion = best_confusion_matrix, accuracy = acc))
}
best_knn <- SearchBestKNN(trainX_s, testX_s, trainY, testY, k = 386, seed = 0)
best_knn$confusion
best_knn$accuracy


