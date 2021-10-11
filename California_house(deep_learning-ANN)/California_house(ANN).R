#Boston housing predicting price using artificial neural networks
#The main idea is try to predict the price of different Boston's houses

#Importing libraries and data
require(keras)
boston_housing <- dataset_boston_housing()
View(boston_housing)

c(train_data, train_label) %<-% boston_housing$train
c(test_data, test_label) %<-% boston_housing$test

#Normalize trining data 
train_data <- scale(train_data)

#Normalize test data using train_data information
col_mean_train <- attr(train_data, "scaled:center")
col_stddev_train <- attr(train_data, "scaled:scale")
test_data <- scale(test_data, center = col_mean_train, scale = col_stddev_train)

#Functional API has two parts: inputs and outputs
#input layer
inputs <- layer_input(shape = dim(train_data)[2])
#Output is composed by inputs and layers
outputs <- inputs %>%
  layer_dense(units = 64, activation = "relu")%>%
  layer_dense(units = 64, activation = "relu")%>%
  layer_dense(units = 1)
set.seed(42)
#Creating, compiling and training model
model <- keras_model(inputs = inputs, outputs = outputs)
model %>% compile(loss = "mse",
                  optimizer = "rmsprop",
                  metrics = list("mean_absolute_error"))
model %>% fit(train_data, train_label, epochs = 30)

#Test performance
score <- model %>% evaluate(test_data, test_label)
cat("Test loss:", score[1], "\n")
cat("Test mae:", score[2], "\n")
Predict <- model %>% predict(test_data)
pred_some <- Predict[1:3]
test_label[1:3]
pred_some #Values are around of real values

#Different architecture using Functional API
input_func <- layer_input(shape = dim(train_data)[2])
#constructing output
output_func <- input_func %>%
  layer_dense(units = 64, activation = "relu")%>%
  layer_dense(units = 64, activation = "relu")
#Concatening inputs with values obtained after the secon hidden layer
main_output <- layer_concatenate(c(output_func, input_func))%>%
  layer_dense(units = 1)
#Creating, compiling and training second model
model_func <- keras_model(inputs = input_func, outputs = main_output)
model_func %>% compile(loss = "mse",
                       optimizer = "rmsprop",
                       metrics = list("mean_absolute_error"))
summary(model_func)
model_func %>% fit(train_data, train_label, epochs = 30)
score_func <- model %>% evaluate(test_data, test_label)
score
score_func

#Saving and restoring Models
model_func %>% save_model_hdf5("my_first_model_r.h5")
new_model <- load_model_hdf5("my_first_model_r.h5")
model_func %>% summary()
summary(new_model)

#Using callbacks
checkpoints_dir <- "checkpoints"
dir.create(checkpoints_dir, showWarnings = F)
filepath <- file.path(checkpoints_dir, "Epochs-{epoch:02d}.hdf5")

#Creating checkpoint callback
cp_callback <- callback_model_checkpoint(filepath = filepath)
rm(model_func)
k_clear_session()
model_cb <- keras_model(inputs = input_func, outputs = main_output)
model_cb %>% compile(loss = "mse",
                  optimizer = "rmsprop",
                  metrics = list("mean_absolute_error"))
model_cb %>% fit(train_data, train_label, epochs = 30, callbacks = list(cp_callback))
list.dirs(checkpoints_dir)

tenth_model <- load_model_hdf5(file.path(checkpoints_dir, "Epochs-10.hdf5"))
summary(tenth_model)

#Saving the best one
callback_best <- callback_model_checkpoint(filepath = "best_model.h5", monitor = "val_loss",
                                           save_best_only = T)
rm(model_cb)
k_clear_session()

model_cb_best <- keras_model(inputs = input_func, outputs = main_output)
model_cb_best %>% compile(loss = "mse", 
                          optimizer = "rmsprop",
                          metrics = list("mean_absolute_error"))
model_cb_best %>% fit(train_data, train_label, epochs = 30,
                      validation_data = list(test_data, test_label),
                      callbacks = list(callback_best))

best_model <- load_model_hdf5("best_model.h5")
summary(best_model)

#Early stopping
callback_list <- list(callback_early_stopping(monitor = "val_loss",
                                              patience = 15),
                      callback_model_checkpoint("early_model.h5",
                                                monitor = "val_loss",
                                                save_best_only = T))
rm(model_cb_best)
k_clear_session()

model_cb_early <- keras_model(inputs = input_func, outputs = main_output)
model_cb_early %>% compile(loss = "mse",
                           optimizer = "rmsprop",
                           metrics = list("mean_absolute_error"))
model_cb_early %>% fit(train_data, train_label, epochs = 100,
                       validation_data = list(test_data, test_label),
                       callbacks = callback_list)
early_best <- load_model_hdf5("early_model.h5")
score_early <- early_best %>% evaluate(test_data, test_label)
