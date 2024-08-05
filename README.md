# BrainTumorDetectionVGG16
Deep learning model for brain tumor detection using MRI images, achieving high accuracy with VGG16 and data augmentation.


### What is a Brain Tumor?

A brain tumor is an abnormal growth of cells within the brain or central spinal canal. These tumors can be either malignant (cancerous) or benign (non-cancerous). They can disrupt normal brain function by pressing on nearby tissues, causing symptoms like headaches, seizures, changes in vision, and problems with motor skills or cognitive function. Early detection and diagnosis are crucial for effective treatment and management.


### Why is it Necessary to Detect Brain Tumors?

1. **Early Diagnosis:** Early detection of brain tumors can lead to timely medical intervention, which can significantly improve the chances of successful treatment and recovery.
2. **Treatment Planning:** Accurate detection helps in planning the appropriate treatment, including surgery, radiation therapy, or chemotherapy.
3. **Prevention of Complications:** Early identification can help prevent complications associated with advanced-stage tumors, such as increased intracranial pressure and neurological deficits.


### How Can This Project be of Help?

This project leverages deep learning techniques to develop a classification system for detecting brain tumors from MRI images. By using a VGG16-based model with data augmentation and regularization techniques, the project aims to achieve high accuracy in distinguishing between images with and without tumors. 

**Key Contributions:**

- **Improved Diagnosis:** The model provides an automated and efficient method for analyzing MRI images, potentially assisting radiologists in making accurate diagnoses faster.
- **Early Detection:** By enhancing the accuracy of brain tumor detection, this project can contribute to earlier diagnosis, improving patient outcomes and treatment success.
- **Research and Development:** This work can serve as a foundation for further research into automated diagnostic systems and the integration of machine learning techniques in medical imaging.


Sure! Here's a more detailed README file for your brain tumor detection classification project:

---

# Brain Tumor Detection Classification System

This project implements a deep learning-based classification system for detecting brain tumors from MRI images. Leveraging the VGG16 architecture, the model is fine-tuned to distinguish between images with tumors and those without, achieving a test accuracy of 86.9%.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Data Augmentation and Training](#data-augmentation-and-training)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

### Objective

The primary goal of this project is to develop an automated system that can accurately classify MRI images of the brain into two categories:
- **Tumor**: Images showing the presence of a brain tumor.
- **No Tumor**: Images with no signs of a tumor.

This automated classification can assist radiologists in identifying potential tumors more quickly and reliably.

### Approach

The project employs a Convolutional Neural Network (CNN) model based on the VGG16 architecture. This pre-trained model, known for its depth and powerful feature extraction capabilities, is fine-tuned with additional layers to tailor it to the specific task of brain tumor classification.

## Dataset

The dataset used for this project is obtained from [Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection). It contains MRI images categorized into two classes:

- **Yes**: Images with a brain tumor.
- **No**: Images without a tumor.

The dataset comprises a total of 253 MRI images. Each image is resized to 224x224 pixels to fit the input size requirements of the VGG16 model.

## Installation

### Prerequisites

Ensure you have Python 3.6 or later installed. You will also need to install the necessary Python libraries, which are listed in the `requirements.txt` file.

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. **Install the required packages:**

   Use pip to install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset from Kaggle:**

   - Create an account on [Kaggle](https://www.kaggle.com) and generate a `kaggle.json` API token from your account settings.
   - Upload the `kaggle.json` file using the Colab interface:

     ```python
     from google.colab import files
     uploaded = files.upload()
     ```

   - Move `kaggle.json` to the appropriate directory:

     ```bash
     !mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
     ```

4. **Download and extract the dataset:**

   Run the following command to download and extract the dataset:

   ```bash
   !kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
   !unzip brain-mri-images-for-brain-tumor-detection.zip -d /content
   ```

5. **Verify the dataset extraction:**

   Ensure the dataset is correctly extracted by listing its contents.

## Model Architecture

### VGG16 Base Model

The VGG16 model is a powerful CNN architecture pre-trained on ImageNet, which provides a solid foundation for feature extraction. The model is used without the top classification layer.

- **Input Shape:** (224, 224, 3)
- **Base Layers:** Convolutional layers with max-pooling

### Custom Layers

To adapt the model for our classification task, additional layers are added on top of the VGG16 base:

- **Global Average Pooling:** Reduces each feature map to a single value by averaging, maintaining spatial information.
- **Dense Layers:** Three dense layers with ReLU activation functions.
  - First Dense Layer: 1024 neurons
  - Second Dense Layer: 1024 neurons
  - Third Dense Layer: 512 neurons
- **Dropout Layers:** Applied after each dense layer to prevent overfitting.
- **Output Layer:** Dense layer with 2 neurons (for binary classification) and softmax activation.

### Model Summary

```plaintext
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
_________________________________________________________________
...
_________________________________________________________________
dense (Dense)                (None, 1024)              102761472
_________________________________________________________________
dropout (Dropout)            (None, 1024)              0
_________________________________________________________________
...
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 1026
=================================================================
Total params: 130,566,210
Trainable params: 115,563,522
Non-trainable params: 15,002,688
_________________________________________________________________
```

## Data Augmentation and Training

### Data Augmentation

To improve model generalization, data augmentation techniques are applied:

- **Rotation:** Randomly rotate images up to 20 degrees.
- **Width and Height Shifts:** Randomly shift images horizontally and vertically by 20% of the dimensions.
- **Shear and Zoom:** Apply random shearing and zooming transformations.
- **Horizontal Flip:** Randomly flip images horizontally.
- **Fill Mode:** Newly created pixels are filled with the nearest available pixel value.

### Training Process

The model is compiled with the Adam optimizer and trained using categorical cross-entropy loss. Several callbacks are used during training:

- **Early Stopping:** Stop training when validation loss does not improve for 5 consecutive epochs.
- **Reduce Learning Rate on Plateau:** Reduce learning rate by a factor of 0.2 if validation loss plateaus for 3 epochs.
- **Model Checkpoint:** Save the model with the best validation loss.

## Results

The model achieves an accuracy of 86.9% on the test dataset. Detailed performance metrics and visualizations are provided below:

### Classification Report

```plaintext
              precision    recall  f1-score   support

    No Tumor       0.90      0.84      0.87       195
       Tumor       0.84      0.90      0.87       168

    accuracy                           0.87       363
   macro avg       0.87      0.87      0.87       363
weighted avg       0.87      0.87      0.87       363
```

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### Training and Validation Accuracy

![Accuracy Plot](accuracy_plot.png)

### Training and Validation Loss

![Loss Plot](loss_plot.png)

## Usage

To use the model for predictions on new data, follow these steps:

1. **Prepare the Image:**
   Ensure the image is in the correct format and size (224x224 pixels).

2. **Load the Model:**
   Load the pre-trained model using Keras:

   ```python
   from keras.models import load_model
   model = load_model('best_model.h5')
   ```

3. **Make Predictions:**
   Use the model to make predictions on new images:

   ```python
   image = ...  # Load your image here
   prediction = model.predict(image)
   predicted_class = np.argmax(prediction, axis=1)
   print("Predicted class:", predicted_class)
   ```

## Contributing

Contributions to this project are welcome. If you have ideas for improvements, bug fixes, or additional features, feel free to open an issue or submit a pull request.

### How to Contribute

1. Fork the repository.
2. Create a new branch for your feature or bugfix: `git checkout -b feature-name`.
3. Make your changes and commit them: `git commit -m 'Add new feature'`.
4. Push to your fork: `git push origin feature-name`.
5. Create a pull request on the original repository.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software as long as the original license terms are respected.
