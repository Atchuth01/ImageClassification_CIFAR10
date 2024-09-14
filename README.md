# ImageClassification_CIFAR10
 Image clasification of 10 class using CIFAR-10 dataset using python.

Here's a `README.md` description for your image classification project using the CIFAR-10 dataset:

---

# CIFAR-10 Image Classification

This repository contains a deep learning project for image classification using the **CIFAR-10** dataset. The project is implemented using **TensorFlow** and **Keras** to build a Convolutional Neural Network (CNN) model. The model is trained on the CIFAR-10 dataset to classify images into 10 distinct categories, achieving accuracy based on test data.

## Project Overview

The CIFAR-10 dataset is a popular dataset used for machine learning and computer vision tasks. It consists of **60,000 32x32 color images** categorized into **10 classes**. These classes include:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is divided into 50,000 training images and 10,000 test images, with each class having 6,000 images. Each image has a resolution of 32x32 pixels and contains RGB color information.

In this project, I used a Convolutional Neural Network (CNN) model to classify the images into one of these 10 classes. The model is designed with multiple convolutional and max-pooling layers to extract important image features, followed by dense layers to perform the classification.

## Model Architecture

The CNN model used in this project has the following architecture:
1. **Input Layer**: Takes a 32x32x3 image (32x32 pixels, 3 color channels).
2. **Convolutional Layer 1**: 32 filters, kernel size (3x3), activation: ReLU
3. **MaxPooling Layer 1**: Pool size (2x2)
4. **Convolutional Layer 2**: 64 filters, kernel size (3x3), activation: ReLU
5. **MaxPooling Layer 2**: Pool size (2x2)
6. **Convolutional Layer 3**: 128 filters, kernel size (3x3), activation: ReLU
7. **MaxPooling Layer 3**: Pool size (2x2)
8. **Flatten Layer**: Flattens the 3D output into a 1D tensor.
9. **Dense Layer**: 128 units, activation: ReLU
10. **Dropout Layer**: Dropout rate of 0.4 to prevent overfitting.
11. **Output Layer**: 10 units, softmax activation for multiclass classification.

## How to Run the Project

### Prerequisites
To run this project, ensure you have the following dependencies installed:
- Python 3.x
- TensorFlow
- Keras
- NumPy

You can install the required libraries using pip:
```bash
pip install tensorflow keras numpy
```

### Running the Code
1. Clone this repository:
   ```bash
   git clone https://github.com/Atchuth01/ImageClassification_CIFAR10.git
   cd ImageClassification_CIFAR10
   ```

2. Run the Python script to train and evaluate the model:
   ```bash
   python ImageClassification.py
   ```

### Model Training
The CNN model is trained on the CIFAR-10 training set for 10 epochs with a batch size of 32. After training, the model is evaluated on the test set.

### Predicting an Image
You can test the trained model on a custom image:
1. Place the image (JPEG format) in the project directory and rename it (e.g., `bird.jpg`).
2. The script processes and resizes the image to 32x32 pixels and normalizes it to the [0, 1] range.
3. The model predicts the class of the image, and the result is printed.

Example prediction:
```bash
Predicted Class: BIRD
```

## CIFAR-10 Dataset

The **CIFAR-10** dataset was created by **Alex Krizhevsky, Geoffrey Hinton, and their collaborators**. It is commonly used for evaluating machine learning algorithms in image recognition. The dataset consists of 60,000 images in 10 categories, with 6,000 images per class. The classes are mutually exclusive, meaning that an image falls into one category only.

More details about the dataset can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Results

The model achieves the following performance on the CIFAR-10 test set:
- **Loss**: 0.67 (Update with actual loss)
- **Accuracy**: 76% (or) 0.76 (Update with actual accuracy)
Note : You can increase accuracy and decrease loss by using methods like hyperparameter tuning etc. 

## Contributing

Feel free to fork this repository and submit pull requests to improve the model or add more features.

## License

This project is licensed under the MIT License.

## Author

Author : Atchuth V