#House price analysis using R
#Importing data
directory <- paste(getwd(), "/House_Price.csv", sep = "") #Set directory where Home_Price.R is
directory
raw_df <- read.csv(directory, header = TRUE)
str(raw_df)

#Extended data directory (EDD)
summary(raw_df)
hist(raw_df$crime_rate)
pairs(~price + crime_rate + n_hot_rooms + rainfall, data = raw_df)
barplot(table(raw_df$airport))
barplot(table(raw_df$waterbody))
barplot(table(raw_df$bus_ter))
#Observations
#1. missing values in n_hos_beds
#2. non-linear realtionship in crime_rate
#3. Outliers in n_hot_rooms and rainfall
#4. bus_ter has only YES value (ignorable)

#Outliners treatment
max_lim <- quantile(raw_df$n_hot_rooms, 0.99)
raw_df$n_hot_rooms[raw_df$n_hot_rooms > 3 * max_lim] <- 3 * max_lim 
summary(raw_df$n_hot_rooms)
min_lim <- quantile(raw_df$rainfall, 0.01)
raw_df$rainfall[raw_df$rainfall < 0.3 * min_lim] <- 0.3 * min_lim
summary(raw_df$rainfall)
pairs(~price + crime_rate + n_hot_rooms + rainfall, data = raw_df)

#Missing values treatment
View(raw_df[is.na(raw_df$n_hos_beds),])#Verifying missing values location
mean_value <- mean(raw_df$n_hos_beds, na.rm = TRUE)
raw_df$n_hos_beds[is.na(raw_df$n_hos_beds)] <- mean_value #Also it can be used which(is.na()) to find na
which(is.na(raw_df$n_hos_beds)) 
View(raw_df[is.na(raw_df$n_hos_beds),]) #There are no more missig values
df = raw_df # Now, data is cleaner than before
rm(raw_df)

#Data Transformation
plot(df$crime_rate, df$price) #non-linear realtionship
df$crime_rate = log(1 + df$crime_rate)
plot(df$crime_rate, df$price) # more identical to a linear relationship
df$avg_dist = (df$dist1 + df$dist2 + df$dist3 + df$dist4) / 4
View(df)
df2 <- df[, -7 : -10]
View(df2)
df <-df2
rm(df2)
df <- df[, -14]
View(df)

install.packages("dummies")
library("dummies")
df <- dummy.data.frame(df)
View(df)
df <- df[,-9]
df <- df[,-14]
View(df)
str(df) #Verifying df shape

#Correlation matrix
corr <- cor(df)
corr <- round(corr, 2)
View(corr)
#At this moment, The parks and air_qual are high correlated, that is why I decide delete parks column
#to avoid a multiple linearity
df <- df[, -16]
View(df)

#Linear regression
                      #y~x
simple_model <- lm(price~room_num, data = df)
summary(simple_model)
plot(df$room_num, df$price, col = "blue")
abline(simple_model, col = "orange")
View(predict(simple_model))

#Multiple linear regression

multiple_model <- lm(price~., data = df)
summary(multiple_model)

# Test train split
install.packages("caTools")
library("caTools")
set.seed(0)
split <- sample.split(df, SplitRatio = 0.8)
training_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)
lm_a <- lm(price~., data = training_set)
train_a <- predict(lm_a, training_set)
test_a <- predict(lm_a, test_set)
mean((test_set$price - test_a)^2)
mean((training_set$price - train_a)^2)

#subset selection
#best selection
install.packages("leaps")
library("leaps")
lm_best <- regsubsets(price~., data = df, nvmax = 15)
summary(lm_best)
summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)
coef(lm_best, 8)
#forward selection
lm_forward <- regsubsets(price~., data = df, nvmax = 15, method = "forward")
summary(lm_forward)
summary(lm_forward)$adjr2
which.max(summary(lm_forward)$adjr2)
coef(lm_forward, 8)
#backward selection
lm_backward <- regsubsets(price~., data = df, nvmax = 15, method = "backward")
summary(lm_backward)
summary(lm_backward)$adjr2
which.max(summary(lm_backward)$adjr2)
coef(lm_backward, 8)

# Ridge and Lasso
install.packages("glmnet")
install.packages("Matrix")
library("Matrix")
library("glmnet")
x <- model.matrix(price~., data = df)[, -1]
y <- df$price
grid <- 10^seq(10, -2, length = 100)
grid
lm_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(lm_ridge)
cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
summary(cv_fit)
plot(cv_fit)
opt_lambda <- cv_fit$lambda.min
tss <- sum((y - mean(y))^2)
y_a <- predict(lm_ridge, s = opt_lambda, newx = x)
rss <- sum((y_a - y)^2)
rsq <- 1- rss/tss
which(grid == opt_lambda)
coef(lm_ridge, which(grid == opt_lambda))
#Lasso
lm_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(lm_lasso)
cv_fit <- cv.glmnet(x, y, alpha = 1, lambda = grid)
summary(cv_fit)
plot(cv_fit)
opt_lambda <- cv_fit$lambda.min
tss <- sum((y - mean(y))^2)
y_a <- predict(lm_lasso, s = opt_lambda, newx = x)
rss <- sum((y_a - y)^2)
rsq <- 1- rss/tss
which(grid == opt_lambda)
coef(lm_lasso, which(grid == opt_lambda))

