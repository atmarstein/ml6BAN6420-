### Load required libraries
library(keras)

### Load and preprocess the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
X_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
X_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

### Normalize and reshape the data
X_train <- X_train / 255
X_test <- X_test / 255
X_train <- array_reshape(X_train, c(nrow(X_train), 28, 28, 1))
X_test <- array_reshape(X_test, c(nrow(X_test), 28, 28, 1))

### One-hot encode the labels
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

### Check the dimensions of the data
print("X_train shape:", dim(X_train))
print("X_test shape:", dim(X_test))
print("y_train shape:", dim(y_train))
print("y_test shape:", dim(y_test))

### Build the CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')

### Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

### Train the model
history <- model %>% fit(
  X_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)

### Make predictions for at least two images
predictions <- model %>% predict(X_test[1:2,, , , drop=FALSE])
cat("Predicted labels:", apply(predictions, 1, which.max) - 1, "\n")
cat("True labels:", apply(y_test[1:2,], 1, which.max) - 1, "\n")
