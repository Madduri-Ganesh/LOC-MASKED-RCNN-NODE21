# Masked R-CNN for Pulmonary Nodule Detection in Chest X-rays

## Overview
This repository contains the implementation of a **Masked R-CNN** model for detecting pulmonary nodules in chest X-ray images. The model is trained and evaluated on the **NODE21** dataset, which combines images from PadChest and ChestX-ray14 datasets, along with a specially curated test set from four Dutch hospitals.

## Dataset: NODE21

- **Dataset Link**: Node21.grand-challenge.org/

### Training Set
- **Image Type**: Posterior-anterior (PA) chest X-rays
- **Composition**:
  - Images with pulmonary nodules (labeled)
  - 1,500 random images without nodules from PadChest and ChestX-ray14

### Test Set
- **Size**: 300 CXR images
- **Source**: Four different hospitals across the Netherlands
- **Positive Case**: Contains a pulmonary nodule, confirmed by CT scan after 3 months
- **Negative Case**: No pulmonary nodule, confirmed by CT scan after 6 months

## Model Architecture: Masked R-CNN
Masked R-CNN is a deep learning model designed for medical image analysis. It processes regions of interest within medical scans through the following steps:
1. **Generating region proposals** via selective search
2. **Extracting features** using a CNN
3. **Classifying** anatomical structures or abnormalities
4. **Refining** locations of detected objects

## Training Details
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: 0.001
- **Momentum**: 0.9
- **Weight Decay**: 0.0005
- **Epochs**: 50

## Evaluation Metrics
- **Mean Average Precision (mAP)**
  - **IoU Threshold**: 0.6
  - Used during both training and testing phases
- **Intersection over Union (IoU)**
  - Directly assesses the quality of generated bounding boxes
