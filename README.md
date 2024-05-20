# Emotion Detection Project

## Overview

This project focuses on detecting emotions from facial images using a pre-trained MobileNet model. The model is fine-tuned to classify images into seven emotion categories: angry, disgust, fear, happy, sad, surprise, and neutral. 

## Dataset

The dataset used for this project is the FER-2013 dataset, which can be found on Kaggle: [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013).

## Project Structure

1. **Importing Libraries**: Essential libraries for data manipulation, model creation, and image processing are imported.
2. **Data Preprocessing**: Image data is augmented and preprocessed for training and validation.
3. **Model Building**: A MobileNet model is used as the base, with added layers for classification.
4. **Model Training**: The model is trained using the training data, with early stopping and model checkpoint callbacks.
5. **Model Evaluation**: The model's performance is evaluated using accuracy and loss metrics.
6. **Visualization**: Training and validation accuracy and loss are plotted for analysis.
7. **Prediction**: The trained model is used to predict emotions on test images.
8. **Real-time Emotion Detection**: The model is deployed to detect emotions in real-time using a webcam feed.

## Results

The model was trained and evaluated on the FER-2013 dataset. The training process involved data augmentation techniques to enhance the model's robustness. The evaluation metrics, including accuracy and loss, were plotted to visualize the model's performance.

## Conclusion

This project demonstrated the use of a pre-trained MobileNet model for emotion detection from facial images. The FER-2013 dataset from Kaggle was used to train and evaluate the model, achieving satisfactory results. Future work could involve further fine-tuning, using more advanced models, and exploring different data augmentation techniques to improve accuracy.

## Future Work

- Further fine-tuning of the model and hyperparameters.
- Exploring different architectures and transfer learning approaches.
- Enhancing data augmentation techniques to improve model robustness.
- Implementing the model in real-world applications for real-time emotion detection.
