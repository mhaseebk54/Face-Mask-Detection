# Face-Mask-Detection
This project performs face mask detection using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.
It classifies whether a person in an image is wearing a mask 😷 or not wearing a mask 🚫.

📌 Project Description

This notebook downloads and preprocesses a face mask dataset from Kaggle.
It then builds and trains a CNN model for binary classification — detecting if a face image shows a masked or unmasked person.

🧠 Model Architecture

Model Type: Convolutional Neural Network (CNN)

Layer Type	Description
Conv2D	32 filters, 3×3 kernel, ReLU activation
MaxPooling2D	2×2 pool size
Conv2D	32 filters, 3×3 kernel, ReLU activation
MaxPooling2D	2×2 pool size
Flatten	Converts feature maps into 1D vector
Dense	128 units, ReLU activation
Dropout	0.5 rate
Dense	64 units, ReLU activation
Dropout	0.5 rate
Dense (Output)	2 units, Sigmoid activation (binary output)

Compilation Settings:

🧩 Optimizer: Adam

🎯 Loss Function: sparse_categorical_crossentropy

📈 Metric: Accuracy

Training Configuration:

Epochs: 5

Validation Split: 0.1 (10%)

🧩 Workflow
🔹 1. Dataset Loading

Downloaded dataset from Kaggle:
omkargurav/face-mask-dataset

Extracted and accessed two folders:

data/with_mask/

data/without_mask/

🔹 2. Data Preprocessing

Resized images to 128×128

Converted to RGB format using PIL

Normalized pixel values by dividing by 255

Assigned labels:

1 → With Mask

0 → Without Mask

Split data into training and testing sets using train_test_split

🔹 3. Model Building

Built CNN model using Keras Sequential API

Added convolutional, pooling, dense, and dropout layers

Compiled model with Adam optimizer and cross-entropy loss

🔹 4. Model Training

Trained for 5 epochs

Used 10% validation split

Monitored accuracy across epochs

🔹 5. Prediction

Accepts image path input from user

Preprocesses image (resize + normalize)

Predicts and outputs:

✅ “The person in the image is wearing a mask”

❌ “The person in the image is not wearing a mask”

🛠️ Tech Stack

Language: Python 🐍

Libraries Used:

TensorFlow / Keras

NumPy

Pandas

Matplotlib

OpenCV

Pillow (PIL)

scikit-learn

📊 Files Included

📁 FaceMask (2).ipynb

Main Jupyter Notebook containing dataset download, preprocessing, CNN model building, training, and prediction steps.

💡 Key Highlights

✅ Built a CNN model for binary image classification
✅ Used a real-world Kaggle dataset of masked and unmasked faces
✅ Implemented image preprocessing, normalization, and visualization
✅ Enabled real-time image prediction for mask detection
