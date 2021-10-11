#Fashion Classification using an artificial neural network

#Installing keras
#install.packages("keras")
#installing core keras and tensorflow
library(keras)
#install_keras()
#If I need instal the gpu version, install_keras(tensorflow = "gpu")

#Importing datsaet from keras dataset
fashion_mnist <- dataset_fashion_mnist()

#Train and test data split
#train_images <- fashion_mnist$train$x
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

#Explore data structure
dim(train_images)
str(train_images)

#Plotting images
fobject <- train_images[2,,]
plot(as.raster(fobject, max = 255))

class_names = c("T-shirt/top", "Trouser", "Pullover",
                "Dress", "Coat", "Sandal", "Shirt",
                "Snicker", "Bag", "Anckle boot")
class_names[train_labels[2] + 1] #Tshit plotted before

#Normilizing data Max pixels is 255
train_images = train_images / 255
test_images = test_images / 255

#Train valid data
valid_indexes <- 1:5000
valid_images <- train_images[valid_indexes, ,]
valid_labels <- train_labels[valid_indexes]
train_images <- train_images[-valid_indexes, ,]
train_labels <- train_labels[-valid_indexes]

#Flattening
#xxx
#yyy -> xxxyyyzzz
#zzz

#Building, compiling and training
set.seed(42)
model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28))%>%
  layer_dense(units = 128, activation = "relu")%>%
  layer_dense(units = 10, activation = "softmax")
model %>% compile(loss = "sparse_categorical_crossentropy",
                  optimizer = "sgd",
                  metrics = c("accuracy"))
#sparse_categorical_crossentropy more than 2 classes and each object belongs only one class
#binary_categorical _crossentropy 2 classes
#categorical_crossentropy more than 2 classes and object can balong different classes

model %>% fit(train_images, train_labels, epochs = 10, batch_size = 32,
              validation_data = list(valid_images, valid_labels))

#Model performance
score <- model %>% evaluate(test_images, test_labels)
cat("Test loss:", score[1], "\n")
cat("Test accuray:", score[2], "\n")

#Predicting
predictions <- model %>% predict(test_images)
round(predictions[1,], digits = 2)
which.max(predictions[1, ])
class_names[which.max(predictions[1, ])]
plot(as.raster(test_images[1,,] * 255, max = 255))

class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]

#NeuralNet package
#Installing package
install.packages("neuralnet")
require(neuralnet)

hours <- c(20, 10, 30, 20, 50, 30)
mocktest <- c(90, 20, 20, 10, 50, 80)
Passed <- c(1, 0, 0, 0, 1, 1)
df <- data.frame(hours, mocktest, Passed)
nn = neuralnet(Passed~., data = df, hidden = c(3, 2), 
               act.fct = "logistic", linear.output = F)
plot(nn)

thours <- c(20, 20, 30)
tmocktest <- c(80, 30, 20)
test <- data.frame(thours, tmocktest)
Predict <- compute(nn, test)
Predict$net.result
prob <- Predict$net.result
pred <- ifelse(prob > 0.5, 1, 0)
pred
