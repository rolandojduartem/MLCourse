#Fashion mnist classification using convusional neural network

#Importing library and dataset
require(keras)
fashion_mnist <- dataset_fashion_mnist()

#Data split and normalization
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test
class_names = c("T-shirt/top", "Trouser", "Pullover",
                "Dress", "Coat", "Sandal", "Shirt",
                "Snicker", "Bag", "Anckle boot")
class_names[train_labels[1] + 1 ]
plot(as.raster(train_images[1, ,], max = 255))
train_images_n <- train_images / 255
test_images_n <- test_images / 255 

indexes <- 1:5000
valid_images <- train_images_n[indexes, ,]
part_train_images <- train_images_n[-indexes, ,]
valid_labels <- train_labels[indexes]
part_train_labels <- train_labels[-indexes]

#Reshaping images
part_train_images <- array_reshape(part_train_images, c(nrow(part_train_images), ncol(part_train_images),
                                                        ncol(part_train_images), 1))
valid_images <- array_reshape(valid_images, c(nrow(valid_images), ncol(valid_images),
                                                        ncol(valid_images), 1))
test_images_n <- array_reshape(test_images_n, c(nrow(test_images_n), ncol(test_images_n),
                                                        ncol(test_images_n), 1))

#Creating model
model <- keras_model_sequential()%>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(28, 28, 1))%>%
  layer_max_pooling_2d(pool_size = c(2, 2))
  #layer_conv_2d(filter = 32, kernel_size = c(3, 3), activation = "relu") #If another conv layer is necessary
model <- model %>%
  layer_flatten()%>%
  layer_dense(300, activation = "relu")%>%
  layer_dense(100, activation = "relu")%>%
  layer_dense(10, activation = "softmax")

summary(model)

#Compiling model
model %>% compile(loss = "sparse_categorical_crossentropy",
                  optimizer = "SGD",
                  metrics = c("accuracy"))

model %>% fit(part_train_images, part_train_labels, epochs = 30, batch_size = 64,
              validation_data = list(valid_images, valid_labels))

#Model performance
score <- model %>% evaluate(test_images_n, test_labels)

class_pred <- model %>% predict_classes(test_images_n)
class_names[class_pred[2] + 1]
plot(as.raster(test_images[2,,], max = 255)) #It is a pullover
rm(model)
k_clear_session()

#Poolig vs no pooling

set.seed(42)
model1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "valid", activation = "relu", input_shape = c(28, 28, 1))%>%
  layer_max_pooling_2d(pool_size = c(2, 2))
model1 <- model1 %>%
  layer_flatten()%>%
  layer_dense(300, activation = "relu")%>%
  layer_dense(100, activation = "relu")%>%
  layer_dense(10, activation = "softmax")

model2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "valid", activation = "relu", input_shape = c(28, 28, 1))
model2 <- model2 %>%
  layer_flatten()%>%
  layer_dense(300, activation = "relu")%>%
  layer_dense(100, activation = "relu")%>%
  layer_dense(10, activation = "softmax")

summary(model1)
summary(model2)

model1 %>% compile(loss = "sparse_categorical_crossentropy",
                  optimizer = "SGD",
                  metrics = c("accuracy"))
model2 %>% compile(loss = "sparse_categorical_crossentropy",
                   optimizer = "SGD",
                   metrics = c("accuracy"))

model1 %>% fit(part_train_images, part_train_labels, epochs = 3, batch_size = 64,
               validation_data = list(valid_images, valid_labels))

model2 %>% fit(part_train_images, part_train_labels, epochs = 3, batch_size = 64,
               validation_data = list(valid_images, valid_labels))
rm(model1)
rm(model2)
k_clear_session()
