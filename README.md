# Auto-PCOS Classification

Link to the dataset: https://drive.google.com/drive/u/0/folders/1uyOYRVsa1aVNbpggwRkoc0sYaXLWCALj

## Overview
Polycystic ovary syndrome (PCOS) is a prevalent endocrine and metabolic disorder affecting millions of women worldwide. The Auto-PCOS classification aims to leverage Artificial Intelligence (AI) to improve the diagnosis of PCOS through ultrasound imaging, addressing the need for modern diagnostic methods.

## Developed Pipeline
Here, we provide a brief overview of our developed pipeline. The code presents a comprehensive approach for diagnosing PCOS using a combination of Deep Neural Network (DNN) and multiple Machine Learning (ML) models, including ensemble techniques for improved accuracy. It involves preprocessing and augmenting medical image data, training a modified ResNet model with Squeeze-and-Excitation (SE) blocks for feature recalibration, and extracting various features (e.g., LBP, SIFT, ORB, CLIP embeddings) for classical ML models. The ensemble method combines predictions from the DNN and ML models to make final diagnoses. Additionally, it includes functions for image preprocessing, feature extraction, and interpretability through techniques like Class Activation Mapping (CAM) and SHAP values. The script is structured to train models, evaluate their performance, predict on unseen data, and provide insights into the model's decision-making process.

<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/604ce74a-fd7c-4979-b1b5-ed141b49c3f1" width="600">

## Achieved Results on Validation Dataset
Our model has been rigorously tested on a validation dataset to ensure its effectiveness. Below are the evaluation metrics achieved:

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **Class 0 (Unhealthy)**         | 0.88      | 0.97   | 0.93     | 460     |
| **Class 1 (Healthy)**         | 0.97      | 0.87   | 0.92     | 459     |
| **Accuracy**  |           |        | 0.92     | 919     |
| **Macro Avg** | 0.93      | 0.92   | 0.92     | 919     |
| **Weighted Avg** | 0.93  | 0.92   | 0.92     | 919     |


### Selected Frames from Validation Dataset
Below are the top 5 frames from the validation dataset, showcasing the model's classification capabilities. 

<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/2659585f-9793-4b36-85ce-7f4ef29776bc" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/3469866b-7e23-42bc-acce-edfab6c22704" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/d68f9ebd-be29-4046-bb4d-be914dc5c2bd" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/e2b7c258-7a08-49c8-9fe0-d3e8783dd6a2" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/e9f7d08d-fbbc-4fb4-8941-5bc4699fc71e" width="180">

### Interpretability Plots from Validation Dataset
We also provide interpretability plots for a deeper understanding of our model's decision-making process.

<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/4dc018e0-fdb5-4035-9a7c-bb8dd31a5c7e" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/900a4039-b313-4ed1-8bec-4a3e043deb4f" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/507c698a-e936-4f77-922c-b4d81090e1da" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/169f44c6-19b4-43b4-ba58-38aa258ab89b" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/758f4d74-7025-46cf-8ada-0fdcec485061" width="180">

## Achieved Results on Testing Dataset
The model's performance was further validated on a testing dataset, with the following frames highlighting its classification accuracy.

### Selected Frames from Testing Dataset
Below are the top 5 frames from the testing dataset, showcasing the model's classification capabilities. 

<!-- Adjust the paths and sizes as needed -->
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/16b07680-00a4-4496-baf5-35efa8774247" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/1964e488-468c-45de-be04-915940b813fc" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/32cdc2d5-809e-4f72-9ec4-48c10737c857" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/1c41ead5-9e12-4413-b389-b4c7b9afb038" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/f72e25a5-2b52-47af-acc0-157bb23decc9" width="180">

### Interpretability Plots from Testing Dataset
We also provide interpretability plots for a deeper understanding of our model's decision-making process.

<!-- Adjust the paths and sizes as needed -->
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/fd387c54-923c-4e68-b191-4878345103bb" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/b7e5671a-2bc4-45ef-82b4-18231ce46e2f" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/2213ad18-21d4-4583-8799-0bda219d4d4f" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/fd3f2813-ed76-45dc-a55d-0c0d990a87e4" width="180">
<img src="https://github.com/musk-singhal/auto-pcos-classification/assets/34962939/c63f4e82-a9bf-4018-982c-f74f45dbd3d9" width="180">

## Conclusion
This challenge provided an invaluable opportunity to advance the field of medical imaging through AI, specifically in the diagnosis of PCOS. Our results demonstrate the potential of machine learning algorithms in enhancing diagnostic accuracy and efficiency.

## Acknowledgements
We extend our gratitude to the organizers of the Auto-PCOS classification challenge and all contributors to the PCOSGen dataset. This project would not have been possible without their support and the shared commitment to improving women's health globally.

For more information on the challenge, visit the [challenge page](https://misahub.in/pcos/index.html).

## More info
Polycystic ovary syndrome (PCOS) is one of the most common endocrine and metabolic disorders in premenopausal women. Heterogeneous by nature, PCOS is defined by a combination of signs and symptoms of androgen excess and ovarian dysfunction in the absence of other specific diagnoses. The aetiology of this syndrome remains largely unknown, but mounting evidence in the recent literature suggests that PCOS might be a complex multigenic disorder with strong epigenetic and environmental influences, including diet and lifestyle factors.
The World Health Organization (WHO) estimates that a staggering 116 million women globally grapple with PCOS. Statistics report that about 70% of the women suffering from PCOS remain undiagnosed which highlights the substantial prevalence and under-recognition of this condition. In India, as per the Indian Fertility society, the prevalence of PCOS ranges from 3.7% to 22.5%. This lifestyle-related ailment leads to a spectrum of metabolic and psychological challenges, including irregular menstrual cycles, hirsutism, sudden weight gain, type 2 diabetes, thyroid irregularities, and increased risk of depression and other psychiatric disorders, significantly affecting overall quality of life.
In the last 25 years, several attempts have been made by institution and societies like National Institutes of Health (NIH), European Society of Human Reproduction and Embryology (ESHRE) and American Society for Reproductive Medicine (ASRM) and Androgen Excess Society (AES) to standardize the diagnostic criteria for PCOS. They are based on various combinations of otherwise unexplained hyperandrogenism, anovulation, and the presence of polycystic ovaries observed through ultrasound imaging. This observation is time-consuming, dependent on the sensitivity of the ultrasound equipment, the skill of the operator, the approach (vaginal v/s abdominal) and the weight of the patient.
Considering the underdiagnoses of PCOS, lack of experts in ultrasound imaging, and overall low patient-physician ratio across globe, there arises a need of modern diagnostic methods to fight PCOS. Artificial intelligence (AI) is predicted to have profound effects on the future of ultrasound imaging technology in the context of contrast enhancement, quality assessment, abnormality-based video summarization, annotations, measurements, and its artefact removal. Machine learning-based algorithms may help in automatic classification, detection and segmentation of polycystic ovaries observed in ultrasound imaging. These algorithms may prove to enhance diagnostic accuracy, reduce the manual steps, and overcome operator-independency.
The aim of the Auto-PCOS classification challenge is to provide an opportunity for the development, testing and evaluation of Artificial Intelligence (AI) models for automatic PCOS classification of healthy and un-healthy frames extracted from ultrasound videos. This challenge encompasses diverse training and test datasets, fostering the creation of vendor-agnostic, interpretable, and broadly applicable AI models. The PCOSGen dataset is first of its kind, consists of different training and test datasets which have been collected from multiple internet resources like YouTube, ultrasoundcases.info, and Kaggle. PCOSGen-train and PCOSGen-test consists of 3200 and 1468 healthy and un-healthy instances respectively. Both training and testing datasets have been medically annotated with the help of experienced gynaecologist based in New Delhi, India.
