#House price for classification models (Linear discriminant analysis)
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

#LDA
install.packages("MASS")
library(MASS)
lda.fit <- lda(Sold~., data  = df)
lda.pred <- predict(lda.fit, df)
View(lda.pred)
lda.pred$posterior #These values are the percentage to be no or yes
lda.class <- lda.pred$class
confusion_matrix <- table(lda.class, df$Sold)
confusion_matrix

#QDA
qda.fit <- qda(Sold~., data  = df)
qda.pred <- predict(qda.fit, df)
View(qda.pred)
qda.pred$posterior #These values are the percentage to be no or yes
qda.class <- qda.pred$class
confusion_matrix <- table(qda.class, df$Sold)
confusion_matrix

