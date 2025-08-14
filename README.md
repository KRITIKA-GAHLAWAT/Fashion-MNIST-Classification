# Fashion-MNIST-Classification
# Fashion-MNIST Classification – Neural Network from Scratch

A deep learning model built entirely in NumPy to classify clothing images from the Fashion-MNIST dataset.
No TensorFlow. No PyTorch. Just raw math, forward propagation, and backward propagation coded from scratch 🚀

# Problem Statement

In this assignment, you will implement a feedforward neural network and manually code the backpropagation algorithm for training. You must use NumPy for all matrix and vector operations, without any automatic differentiation packages. The network will be trained on the Fashion-MNIST dataset to classify 28×28 grayscale images into 10 fashion categories. The project also explores various optimization techniques and hyperparameter tuning to enhance model performance. The implementation utilizes NumPy, Pandas, and Matplotlib, with Keras for dataset loading and Scikit-learn's train_test_split for data preprocessing.

# Project Highlights

- From-Scratch Implementation – no auto-differentiation, manual gradient calculations

- Flexible Architecture – custom number of layers, neurons, and activation functions

- Multiple Optimizers – SGD, Momentum, NAG, RMSProp, Adam, Nadam

- Hyperparameter Tuning – automated sweeps with Weights & Biases (W&B)

- Generalization Testing – model evaluated on both Fashion-MNIST and MNIST

# Dataset

Fashion-MNIST contains:

- 70,000 images of clothing (60k train, 10k test)

- 10 classes (e.g., t-shirt, trousers, coat, sneaker)

- 28×28 grayscale images

# Preprocessing

- Flattening – each 28×28 image → 784-dimensional vector

- Normalization – pixel values scaled to [0, 1]

- One-hot encoding – labels converted to 10-dimensional binary vectors

# Neural Network Workflow

# Forward Propagation

- Inputs → hidden layers (activation: ReLU, Sigmoid, or Tanh)

- Output layer → Softmax for class probabilities

- Predictions generated for training & validation sets

# Backward Propagation

- Perform forward pass to compute predictions

- Calculate loss (Cross-Entropy or MSE)

- Compute gradients for weights & biases manually

- Backpropagate errors layer-by-layer

- Update parameters using the chosen optimization algorithm

- Repeat for multiple epochs until convergence

# Optimization Algorithms Implemented

- SGD – Stochastic Gradient Descent

- Momentum – Accelerated gradient updates

- NAG – Nesterov Accelerated Gradient

- RMSProp – Adaptive learning rate

- Adam – Adaptive Moment Estimation

- Nadam – Adam + Nesterov momentum

# Hyperparameter Tuning

- Using Weights & Biases, explored multiple configurations:

- Learning rate: 0.001, 0.0001

- Hidden layers: 3, 4, 5

- Neurons: 32, 64, 128

- Activations: sigmoid, tanh, ReLU

- Batch size: 16, 32, 64

- Epochs: 5, 10

- Weight init: Xavier, Random

# Evaluation

- Confusion Matrix for best-performing model

- Model also tested on MNIST handwritten digits for generalization

# Tech Stack

- Python (NumPy, Pandas, Matplotlib)

- Scikit-learn (train-test split, metrics)

- Keras (dataset loading only)

- Weights & Biases (hyperparameter tuning)
