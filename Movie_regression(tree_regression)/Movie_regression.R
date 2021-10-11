#Movie collections (Tree regression)
#Importing and checking data
df <- read.csv("Movie_regression.csv", header = TRUE)
View(df)
summary(df)
#Missing value treatment
df$Time_taken[is.na(df$Time_taken)] <- mean(df$Time_taken, na.rm = TRUE)
str(df)
summary(df)
# Dummy variables creation
library(dummies)
df <- dummy.data.frame(df)
View(df)
df <- df[,-12] 
df<- df[,-15] 
View(df)

#Train and test data split

install.packages("caTools")
library(caTools)
set.seed(0)
split  <- sample.split(df, SplitRatio = 0.8)
train <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)
View(test)

#Training and testing tree regression
install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)

regtree <- rpart(formula = Collection~., data = train, control = rpart.control(maxdepth = 3))
test$pred <- predict(regtree, test, type = "vector")
MSE2 <- mean((test$pred - test$Collection)^2)
print(MSE2)
#plotting decision tree
rpart.plot(regtree, box.palette = "RdBu", digits = -3, tweak = 1.3)

#Pruning Tree

fulltree <- rpart(formula = Collection~., data = train, control = rpart.control(cp = 0))
rpart.plot(fulltree, box.palette = "RdBu", digits = -3)
printcp(fulltree)
plotcp(fulltree)

mincp <- regtree$cptable[which.min(regtree$cptable[, "xerror"]), "CP"]
mincp
prunedtree <- prune(regtree, cp = mincp)
rpart.plot(prunedtree, box.palette = "RdBu", digits = -3)

mincpf <- fulltree$cptable[which.min(fulltree$cptable[, "xerror"]), "CP"]
prunedfulltree <- prune(fulltree, cp = mincpf)
rpart.plot(prunedfulltree, box.palette = "RdBu", digits = -3)


test$fulltree <- predict(fulltree, test, type = "vector")
MSE2fulltree <- mean((test$Collection - test$fulltree)^2)
MSE2fulltree

test$pruned <- predict(prunedtree, test, type = "vector")
MSE2prunedtree <- mean((test$Collection - test$pruned)^2)
MSE2prunedtree

test$prunedfulltree <- predict(prunedfulltree, test, type = "vector")
MSE2prunedfulltree <- mean((test$Collection - test$prunedfulltree)^2)
MSE2prunedfulltree

#The fulltree minmize the mean squared error, so fulltree is the most accurate
percerror <- 100 * abs(test$Collection - test$fulltree) / test$Collection
summary(percerror)
#Using fulltree model, it has a mean error 14.10%
plot(seq(1, 101), percerror)
