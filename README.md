# Auto-PCOS Classification Challenge

## Overview
Polycystic ovary syndrome (PCOS) is a prevalent endocrine and metabolic disorder affecting millions of women worldwide. The Auto-PCOS classification challenge aims to leverage Artificial Intelligence (AI) to improve the diagnosis of PCOS through ultrasound imaging, addressing the need for modern diagnostic methods.

## Developed Pipeline
Here, we provide a brief overview of our developed pipeline. The code presents a comprehensive approach for diagnosing PCOS using a combination of Deep Neural Network (DNN) and multiple Machine Learning (ML) models, including ensemble techniques for improved accuracy. It involves preprocessing and augmenting medical image data, training a modified ResNet model with Squeeze-and-Excitation (SE) blocks for feature recalibration, and extracting various features (e.g., LBP, SIFT, ORB, CLIP embeddings) for classical ML models. The ensemble method combines predictions from the DNN and ML models to make final diagnoses. Additionally, it includes functions for image preprocessing, feature extraction, and interpretability through techniques like Class Activation Mapping (CAM) and SHAP values. The script is structured to train models, evaluate their performance, predict on unseen data, and provide insights into the model's decision-making process.

<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/604ce74a-fd7c-4979-b1b5-ed141b49c3f1" width="600">

## Achieved Results on Validation Dataset
Our model has been rigorously tested on a validation dataset to ensure its effectiveness. Below are the evaluation metrics achieved:

| Metric     | Value |
|------------|-------|
| Accuracy   | XX%   |
| Precision  | XX%   |
| Recall     | XX%   |
| F1-Score   | XX%   |
| AUC-ROC    | XX%   |

### Selected Frames from Validation Dataset
Below are the top 5 frames from the validation dataset, showcasing the model's classification capabilities.

<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/c9b64302-1c0f-4335-ba4c-358563b36ea0" width="200">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/55bbfd63-c4e4-449e-bcff-a978f4397840" width="200">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/427683e0-4c02-4f9c-a712-767b676afd4f" width="200">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/b6463295-ba1b-49e4-8ae6-8c2c136ae5e8" width="200">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/456902ff-99f7-4a19-a978-4201ed7e79fe" width="200">

### Interpretability Plots from Validation Dataset
We also provide interpretability plots for a deeper understanding of our model's decision-making process.

<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/7edae352-3437-4479-b85e-02badb3979be" width="200">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/dd98c6d5-4ee5-4d94-8226-9bc0b42c1372" width="200">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/8e7cac3e-1211-482a-a288-10bd79115e5f" width="200">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/7ee297c0-00d3-4dc3-8098-9ab81dcc3d52" width="200">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/690063d0-1ca6-45d5-be09-b56a68a3821d" width="200">

## Achieved Results on Testing Dataset
The model's performance was further validated on a testing dataset, with the following frames highlighting its classification accuracy.

### Selected Frames from Testing Dataset
<!-- Adjust the paths and sizes as needed -->
<img src="path/to/test_frame1.png" width="200">
<img src="path/to/test_frame2.png" width="200">
<img src="path/to/test_frame3.png" width="200">
<img src="path/to/test_frame4.png" width="200">
<img src="path/to/test_frame5.png" width="200">

### Interpretability Plots from Testing Dataset
<!-- Adjust the paths and sizes as needed -->
<img src="path/to/test_plot1.png" width="200">
<img src="path/to/test_plot2.png" width="200">
<img src="path/to/test_plot3.png" width="200">
<img src="path/to/test_plot4.png" width="200">
<img src="path/to/test_plot5.png" width="200">

## Conclusion
This challenge provided an invaluable opportunity to advance the field of medical imaging through AI, specifically in the diagnosis of PCOS. Our results demonstrate the potential of machine learning algorithms in enhancing diagnostic accuracy and efficiency.

## Acknowledgements
We extend our gratitude to the organizers of the Auto-PCOS classification challenge and all contributors to the PCOSGen dataset. This project would not have been possible without their support and the shared commitment to improving women's health globally.

For more information on the challenge, visit the [challenge page](https://misahub.in/pcos/index.html).
