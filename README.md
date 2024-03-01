# Auto-PCOS Classification Challenge

## Overview
Polycystic ovary syndrome (PCOS) is a prevalent endocrine and metabolic disorder affecting millions of women worldwide. The Auto-PCOS classification challenge aims to leverage Artificial Intelligence (AI) to improve the diagnosis of PCOS through ultrasound imaging, addressing the need for modern diagnostic methods.

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


# Project Setup Guide

This guide will walk you through the steps to set up your Python environment for the project. It includes installing Python 3.11, setting up a virtual environment, and installing the required dependencies.

## Prerequisites

Ensure you have administrative access on your machine, as this may be required for the installation steps.

## Installing Python 3.11

### Windows

1. Download the Python 3.11 installer from the [official Python website](https://www.python.org/downloads/).
2. Run the installer. Make sure to check the box that says "Add Python 3.11 to PATH" during installation.
3. Follow the installation prompts to complete the installation.

### macOS

1. Install [Homebrew](https://brew.sh/) if you haven't already, by running the following command in the Terminal:
   ```sh
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Once Homebrew is installed, you can install Python 3.11 by running:
   ```sh
   brew install python@3.11
   ```
3. After the installation completes, ensure the system uses the Homebrew version of Python by adding it to your `PATH` in your `.bash_profile` or `.zshrc` file:
   ```sh
   echo 'export PATH="/usr/local/opt/python@3.11/bin:$PATH"' >> ~/.bash_profile
   # Or for zsh users
   echo 'export PATH="/usr/local/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
   ```

### Linux

The method to install Python 3.11 on Linux depends on the distribution. For Debian-based distributions (like Ubuntu), you can use the following commands:

1. Update and upgrade your package manager with:
   ```sh
   sudo apt-get update
   sudo apt-get upgrade
   ```
2. Install the prerequisites for building Python:
   ```sh
   sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev
   ```
3. Download the Python 3.11 source code from the [official website](https://www.python.org/downloads/source/) and extract it.
4. Navigate to the extracted directory and run:
   ```sh
   ./configure --enable-optimizations
   make -j 8
   sudo make altinstall
   ```

## Creating a Virtual Environment

Once Python 3.11 is installed, create a virtual environment for the project. Navigate to your project directory in the terminal and run:

```sh
python3.11 -m venv env
```

This command creates a new directory named `env` in your project directory, containing the virtual environment.

## Activating the Virtual Environment

Before installing dependencies, you must activate the virtual environment. The activation command varies depending on your operating system:

### Windows

```sh
.\env\Scripts\activate
```

### macOS and Linux

```sh
source env/bin/activate
```

## Installing Dependencies

With the virtual environment activated, install the project dependencies by running:

```sh
pip install -r requirements.txt
```

Ensure `requirements.txt` is present in your project directory and contains all the necessary packages.

## Update Paths accordng to yours in .py file

```sh
training_dataset_path = '/content/drive/MyDrive/data/input/PCOSGen-train'   # DIR FOR first data provided by misahub
test_dataset_path = '/content/drive/MyDrive/data/input/PCOSGen-test/images/' # test data dir
label_path ='/content/drive/MyDrive/data/input/class_label.xlsx'  # class label for provided data
data_dir = '/content/drive/MyDrive/data/output' # create one dir and put path here for storing intermediet result and final results
train_dir = os.path.join(data_dir, 'train')  # dir for training data after spliting given data into 80%:20% ratio , test data 80%
val_dir = os.path.join(data_dir, 'val') # dir for validation data after spliting given data into 80%:20% ratio , test data 80%
train_labels_path = os.path.join(data_dir, 'train_labels.csv' )  # dir for training data label after spliting given data into 80%:20% ratio , test data 80%
val_labels_path = os.path.join(data_dir, 'val_labels.csv' )  # dir for validation data label after spliting given data into 80%:20% ratio , test data 20%
models_path = os.path.join(data_dir, 'models') # dir to store AI models
plot_dir = os.path.join(data_dir, 'plots') #dir to store plots
os.makedirs(models_path, exist_ok=True) # creating dir for models
os.makedirs(plot_dir, exist_ok=True) # creating dir for plots
result_submission_file =  os.path.join(data_dir, 'result_submission.xlsx' ) # path to store final result for test data

```

## Open terminal and go to the project dir then Run following cmd on terminal to train , eval and test data pred


```sh
python3 auto-pcos-classification.py
```

## Conclusion
This challenge provided an invaluable opportunity to advance the field of medical imaging through AI, specifically in the diagnosis of PCOS. Our results demonstrate the potential of machine learning algorithms in enhancing diagnostic accuracy and efficiency.

## Acknowledgements
We extend our gratitude to the organizers of the Auto-PCOS classification challenge and all contributors to the PCOSGen dataset. This project would not have been possible without their support and the shared commitment to improving women's health globally.

For more information on the challenge, visit the [challenge page](https://misahub.in/pcos/index.html).
