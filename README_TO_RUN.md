
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