# Fashion MNIST Classification using Convolutional Neural Network

This project implements a Convolutional Neural Network (CNN) to classify the Fashion MNIST dataset using both Python and R.

---

## Description

The objective of this project is to build a CNN model with six layers to classify images from the Fashion MNIST dataset. The model is designed to recognize different categories of clothing items, such as shirts, shoes, and bags.

---

## Dataset

The Fashion MNIST dataset is used for this classification task. It contains:
- 60,000 training images
- 10,000 testing images
- Image size: 28x28 pixels
- Grayscale (1 channel)
- 10 categories (e.g., T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

---

## Model Architecture

The implemented CNN model consists of six layers:
1. **Convolutional Layer**: 32 filters, 3x3 kernel, ReLU activation
2. **Max Pooling Layer**: 2x2 pool size
3. **Convolutional Layer**: 64 filters, 3x3 kernel, ReLU activation
4. **Max Pooling Layer**: 2x2 pool size
5. **Fully Connected Layer**: 128 units, ReLU activation, Dropout (0.5)
6. **Output Layer**: 10 units, Softmax activation

---

## Implementation Details

This project is implemented in both:
1. **Python**: Using TensorFlow/Keras.
2. **R**: Using Keras in R.

The script includes:
- Loading and preprocessing the Fashion MNIST dataset
- Normalizing image pixel values to [0, 1]
- Reshaping the data to include the channel dimension
- One-hot encoding of the labels
- Building, compiling, and training the CNN model
- Making predictions for at least two images from the test set

---

## Requirements

### Python
- TensorFlow
- Keras
- NumPy

### R
- Keras
- TensorFlow

---

## Running the Scripts

### Python
To run the Python version:
Fashion_Mnist_Classification_Python.py

### R
To run the R version:
Fashion_Mnist_Classification.R

### Output
Training and validation accuracy and loss plots
Predicted labels and true labels for at least two test images

### Results
The model is expected to achieve a reasonable accuracy on the Fashion MNIST dataset. Results may vary depending on the number of epochs and batch size.
