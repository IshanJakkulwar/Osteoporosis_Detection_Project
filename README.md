# Osteoporosis_Detection_Project

Osteoporosis Multi-Regional Classification
A Novel Multifocal Transfer Learning Neural Network Model for Early, Accurate, and Cost-Effective Detection of Osteoporosis via Radiographs
 Overview
This project implements a deep learning-based model to detect osteoporosis and osteopenia from knee and spine X-ray images using transfer learning. The goal is to provide a cheaper, more accessible alternative to DEXA scans, particularly for low-resource settings.

The best-performing model uses EfficientNet-B0 and achieves:

Accuracy: ~87.5%

Recall: 80%

Precision: ~70–75%

F1 Score: 0.719

Developed by Ishan Jakkulwar at Queen Elizabeth’s School Barnet, under the mentorship of Michael Noonan.

 Key Features
 Multi-region classification (knee and spine)

 Transfer learning with EfficientNet-B0

 Custom data augmentation techniques

 Robust evaluation with precision, recall, F1-score

 Designed for medical integration and low-resource settings

 Project Structure
php
Copy
Edit
final.ipynb       # Main Colab Notebook containing the entire ML workflow
README.md         # You're reading it!
 Dataset and Sampling
Public datasets from HuggingFace, Kaggle, and Radiopaedia

Includes normal, osteopenic, and osteoporotic X-rays

Sampling methods:

Stratified Sampling

Random Sampling

Oversampling via augmentation

Optional K-Fold Cross Validation for generalisation

 Technologies Used
Python (Google Colab)

Libraries:

PyTorch

OpenCV

NumPy

Matplotlib

Albumentations (for image augmentation)

 Model Pipeline
Custom 5-Step Machine Learning Process:

Data Preparation: Cleaning, formatting, stratified sampling

Feature Extraction: Identifying diagnostic markers (bone density patterns)

Data Augmentation: Rotation, brightness, flipping, etc.

Model Training: Custom CNN vs. EfficientNet-B0

Model Evaluation: Confusion Matrix, ROC-AUC, and core metrics

 Evaluation Metrics
Metric	Best Score
Accuracy	87.15%
Recall	0.800
Precision	0.700
F1 Score	0.719

Confusion Matrix: Used to understand false positives and false negatives

ROC-AUC: Assesses the model’s ability to distinguish between classes

 Results
The EfficientNet-B0 model outperformed all others, proving highly effective despite a relatively small dataset. Training and test losses indicated minimal overfitting. The model shows promise for clinical integration to support early diagnosis and reduce fracture risks.

 Future Work
Expand the dataset (diversity in age, sex, ethnicity)

Integrate with hospital imaging systems

Explore ensemble methods or larger architectures

Improve real-time diagnostic capabilities

 References
HuggingFace: Spine X-ray Dataset

Kaggle: Knee Osteoporosis Dataset

Radiopaedia and Radiology Masterclass

Sozen et al., "Management of Osteoporosis", Eur J Rheumatol, 2019

 Contact
If you'd like to learn more or collaborate:

Ishan Jakkulwar
Queen Elizabeth’s School Barnet
