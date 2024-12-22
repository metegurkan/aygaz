Animal Classification with CNN

This repository contains a convolutional neural network (CNN) model implementation for classifying animals into different categories. The project utilizes a dataset of animal images and demonstrates the process of building, training, and evaluating a deep learning model.

Features

Preprocessing and augmentation of the image dataset.

Construction of a CNN model using TensorFlow/Keras.

Training and validation of the model.

Application of Gray World algorithm for color constancy.

Evaluation of the model's performance using accuracy and loss metrics.

Dataset

The dataset used in this project contains labeled images of different animals. The images are preprocessed to have a uniform size (128x128) and normalized for better model performance.

Data Preprocessing

Resizing all images to 128x128 pixels.

Normalizing pixel values to the range [0, 1].

Performing data augmentation:

Random rotations.

Horizontal flipping.

Width and height shifting.

Zooming and shearing.

Model Architecture

The CNN model consists of:

Convolutional Layers: Extract spatial features using kernels with ReLU activation.

MaxPooling Layers: Downsample feature maps to reduce computational complexity.

Flatten Layer: Convert 2D feature maps to a 1D vector for the dense layers.

Dense Layers: Fully connected layers with a final output layer using softmax activation for classification.

Summary of Layers:

Conv2D (32 filters, 3x3 kernel, ReLU activation)

MaxPooling2D (2x2)

Conv2D (64 filters, 3x3 kernel, ReLU activation)

MaxPooling2D (2x2)

Conv2D (128 filters, 3x3 kernel, ReLU activation)

MaxPooling2D (2x2)

Flatten

Dense (128 units, ReLU activation)

Dropout (50%)

Dense (number of classes, Softmax activation)

Training

Optimizer: Adam

Loss Function: Sparse categorical cross-entropy

Metrics: Accuracy

Training Steps:

Compile the model with the defined optimizer, loss function, and metric.

Train the model using the fit() method with augmented training data.

Validate the model on unseen data to monitor performance.

Evaluation

The model's performance is evaluated on a test set. Additionally, the Gray World algorithm is applied to improve color constancy in the input images.

Key Metrics:

Accuracy: Percentage of correctly classified images.

Loss: The categorical cross-entropy loss.

Results

The final accuracy of the model on the test set is approximately X% (replace with the actual result). After applying the Gray World algorithm, the accuracy improved to Y% (replace with the actual result).

Usage

Clone this repository:

git clone https://github.com/metegurkan/animal-classification-with-cnn.git

Install the required libraries:

pip install -r requirements.txt

Run the Jupyter notebook:

jupyter notebook animal_classification.ipynb

Future Work

Improve the model by adding more layers or adjusting hyperparameters.

Experiment with other color constancy algorithms.

Use transfer learning for better performance on small datasets.

Deploy the model using Flask or TensorFlow.js for real-time classification.
