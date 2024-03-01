
# pip install tqdm

import os
import random
import shutil
from PIL import Image
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from PIL import Image
# !pip install -U scikit-image
from joblib import dump, load
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")



import os
# BASE_DIR = os.path.dirname(os.path.join(os.path.dirname(__file__), 'pcos'))
# print("Base Dir:", BASE_DIR)
import os
# BASE_DIR = os.path.dirname(os.path.join(os.path.dirname(__file__), 'pcos'))
# print("Base Dir:", BASE_DIR)
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

import random
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to prepare a balanced dataset
def prepare_balanced_dataset():
    # Load images from a folder and filter by category
    def load_images_from_folder(folder, category, image_category_map):
        images = []
        # Iterate through filenames in the given folder
        for filename in os.listdir(folder):
            # Check if the image's category matches the desired category
            if image_category_map.get(filename) == category:
                img_path = os.path.join(folder, filename)
                # Verify the file exists before appending to the list
                if os.path.isfile(img_path):
                    images.append(filename)
        return images

    # Oversample images to achieve desired count for balancing
    def oversample_images(images, desired_count):
        oversampled = images[:]
        # Continue appending random images until reaching the desired count
        while len(oversampled) < desired_count:
            oversampled.append(random.choice(images))
        return oversampled

    # Create a new directory or clear an existing one
    def create_or_clear_dir(dir_path):
        # If the directory exists, remove it and all its contents
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        # Create the directory
        os.makedirs(dir_path, exist_ok=True)

    # Begin data preparation
    print("Start preparing data...")
    directories = [train_dir, val_dir]
    # Create or clear directories for training and validation data
    for directory in directories:
        create_or_clear_dir(directory)

    # Read the Excel file containing image paths and their health status
    df = pd.read_excel(label_path)
    # Create a map of image filenames to their categories
    image_category_map = pd.Series(df.Healthy.values, index=df.imagePath).to_dict()

    # Determine the maximum number of images in any category
    max_count = df['Healthy'].value_counts().max()

    # Prepare a list to hold all data
    all_data = []
    categories = df['Healthy'].unique()
    # Process each category
    for category in categories:
        # Load and oversample images for the current category
        images = load_images_from_folder(training_dataset_path, category, image_category_map)
        oversampled_images = oversample_images(images, max_count)
        # Extend the all_data list with tuples of image filenames and their category
        all_data.extend([(image, category) for image in oversampled_images])

    # Split the combined dataset into training and validation sets
    train_data, val_data = train_test_split(all_data, test_size=0.2, stratify=[label for _, label in all_data])

    # Save the images to their respective directories and return label data for CSV
    def save_images(data, root_dir):
        label_data = []
        for img_filename, label in data:
            # Define source and target paths for each image
            source_path = os.path.join(training_dataset_path, img_filename)
            target_path = os.path.join(root_dir, img_filename)
            # Copy the image from source to target
            shutil.copy(source_path, target_path)
            # Append the image filename and label to the label_data list
            label_data.append([img_filename, label])
        return label_data

    # Save images and labels for training and validation sets
    train_label_data = save_images(train_data, train_dir)
    val_label_data = save_images(val_data, val_dir)

    # Save the label data to CSV files
    pd.DataFrame(train_label_data, columns=['id', 'label']).to_csv(train_labels_path, index=False)
    pd.DataFrame(val_label_data, columns=['id', 'label']).to_csv(val_labels_path, index=False)
    # Indicate completion
    print(f"Balanced Dataset created under {directories}")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from torchvision.transforms.functional import to_pil_image

from PIL import Image



def plot_cam(model, image_tensor, target_layer, filename):
    # Switch the model to evaluation mode
    model.eval()
    # Ensure the tensor is on the same device as the model
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Extract features and model output
        features_fn = torch.nn.Sequential(*list(model.base_model.children())[:-2])
        features = features_fn(image_tensor.unsqueeze(0))
        output = model(image_tensor.unsqueeze(0))

    # Move tensors to CPU for further processing
    features = features.cpu()
    output = output.cpu()

    # Get weights for the target class from the fully connected layer
    params = list(model.fc.parameters())
    weight_softmax = np.squeeze(params[0].cpu().data.numpy())

    # Compute CAM
    class_idx = torch.topk(output, 1)[1].int()
    cam = weight_softmax[class_idx.squeeze().item()]
    cam = np.dot(cam, features.reshape((features.shape[1], -1)))
    cam = cam.reshape(features.shape[2], features.shape[3])
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam_img = np.uint8(255 * cam)

    # Resize CAM to the size of the input image
    cam_img = np.float32(Image.fromarray(cam_img).resize((image_tensor.shape[2], image_tensor.shape[1]), Image.LINEAR))

    # Convert CAM to a heatmap
    heatmap = plt.get_cmap('jet')(cam_img)[:, :, :3]
    heatmap = Image.fromarray(np.uint8(heatmap * 255))

    # Create a matplotlib figure
    fig, ax = plt.subplots()
    ax.imshow(transforms.ToPILImage()(image_tensor.cpu().squeeze()))
    ax.imshow(np.asarray(heatmap), cmap='jet', alpha=0.5, extent=(0, image_tensor.shape[2], image_tensor.shape[1], 0))
    ax.axis('off')

    # Save the figure with 600 DPI
    fig.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



def plot_occlusion(model, image_tensor, filename, occlusion_size=4, stride=1):


    # Ensure the image tensor is on the correct device
    image_tensor = image_tensor.to(device)

    # Calculate the output height and width based on occlusion size and stride
    output_height = int(np.ceil((224 - occlusion_size) / stride) + 1)
    output_width = int(np.ceil((224 - occlusion_size) / stride) + 1)

    # Initialize the heatmap tensor on the correct device
    heatmap = torch.zeros((output_height, output_width), device=device)

    model.eval()
    with torch.no_grad():
        for h in range(0, 224, stride):
            for w in range(0, 224, stride):
                h_start = h
                w_start = w
                h_end = min(224, h + occlusion_size)
                w_end = min(224, w + occlusion_size)

                if (h_end - h_start) != occlusion_size or (w_end - w_start) != occlusion_size:
                    continue

                input_image = image_tensor.clone().detach()
                input_image[:, h_start:h_end, w_start:w_end] = 0
                input_image = input_image.to(device)  # Ensure the modified tensor is on the correct device

                output = model(input_image.unsqueeze(0))
                output = torch.nn.functional.softmax(output, dim=1)
                prob = output.max().item()

                heatmap[int(np.floor(h / stride)), int(np.floor(w / stride))] = prob

    # Plot and save the heatmap
    fig, ax = plt.subplots()
    ax.imshow(image_tensor.cpu().squeeze().permute(1, 2, 0))
    ax.imshow(heatmap.cpu(), cmap='hot', alpha=0.5, extent=(0, 224, 224, 0))
    ax.axis('off')
    plt.savefig(filename, dpi=600)
    plt.close()


# !pip install shap

import shap
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure



# Set up the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to plot SHAP values
def plot_shap(model, image_tensor, filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure the model is in evaluation mode
    model.eval()

    # Check if the image tensor has a batch dimension, and add it if missing
    if image_tensor.dim() == 3:
        # Adds a batch dimension at the start
        image_tensor = image_tensor.unsqueeze(0)

    # Move the image tensor to the same device as the model
    image_tensor = image_tensor.to(device)

    # Define a wrapper function for the model to be compatible with SHAP
    def model_wrapper(input_data):
        # Convert input_data to a PyTorch tensor if not already, ensuring it is on the correct device
        input_data = torch.tensor(input_data, device=device, dtype=torch.float32)
        # Ensure the input data has the correct shape (batch_size, channels, height, width)
        if input_data.dim() == 3:
            input_data = input_data.unsqueeze(0)
        # Forward pass through the model
        with torch.no_grad():
            output = model(input_data)
        # Convert output to numpy array for SHAP
        return output.cpu().numpy()

    # Assuming the image tensor shape is [batch_size, channels, height, width], like [1, 3, 224, 224]
    input_shape = image_tensor.shape[1:]  # This gets the shape as (channels, height, width)

    # Create a SHAP masker specifying the expected input shape
    masker = shap.maskers.Image("input", shape=input_shape)

    # Create a SHAP explainer
    explainer = shap.Explainer(model_wrapper, masker)

    # Generate SHAP values
    shap_values = explainer(image_tensor.cpu().numpy())

    # Plotting the SHAP values
    # Depending on your specific case, you might need to adjust how you index into shap_values
    shap.image_plot(shap_values[0], -image_tensor.cpu().numpy())

    # Save the plot
    plt.savefig(filename)
    plt.close()

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models

# Set the computation device based on CUDA availability.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32  # Define the batch size for training and evaluation.

# Define a custom dataset class for handling image data.
class PCOSDataset(Dataset):
    def __init__(self, data_dir, labels_file_path=None, transform=None):
        # Initialize dataset with images directory and optional labels and transformations.
        self.transform = transform
        # If no labels file path is provided, assume unlabeled data.
        if labels_file_path is None:
            files = os.listdir(data_dir)
            self.df = pd.DataFrame({'id': files, 'label': [0] * len(files)})
        else:
            self.df = pd.read_csv(labels_file_path)
        self.images_dir = data_dir

    def __len__(self):
        # Return the total number of items in the dataset.
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve an image and its label by index, applying any transformations.
        img_id, label = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, f"{img_id}")
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# Define image transformations for data augmentation and normalization.
transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
    ], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a Squeeze-and-Excitation block for channel re-calibration.
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply SE block operations to re-calibrate channel-wise feature responses.
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Define a modified ResNet model incorporating SE blocks.
class ModifiedResNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        # Add SE blocks to the original ResNet model at each stage.
        self.se_block1 = SEBlock(256)
        self.se_block2 = SEBlock(512)
        self.se_block3 = SEBlock(1024)
        self.se_block4 = SEBlock(2048)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Forward pass through the base model with SE blocks and a custom classifier.
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.se_block1(x)
        x = self.base_model.layer2(x)
        x = self.se_block2(x)
        x = self.base_model.layer3(x)
        x = self.se_block3(x)
        x = self.base_model.layer4(x)
        x = self.se_block4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def dnn_model_train_and_eval():
    # Initialize and split the dataset for training and validation.
    train_dataset = PCOSDataset(data_dir=train_dir, labels_file_path=train_labels_path, transform=transformations)
    len_train = int(0.8 * len(train_dataset))
    len_val = len(train_dataset) - len_train
    train_ds, val_ds = random_split(train_dataset, [len_train, len_val])

    # Prepare data loaders for training and validation.
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Train the model and evaluate its performance on the validation set.
    def train_epoch(model, dataloader, loss_fn, optimizer, device):
        model.train()
        total_loss, total_correct = 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == targets).sum().item()
        return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)

    def evaluate(model, dataloader, loss_fn, device):
        model.eval()
        total_loss, total_correct = 0, 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == targets).sum().item()
        return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)

    # Initialize the model, optimizer, loss function, and learning rate scheduler.
    model = ModifiedResNet(num_classes=2).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Train the model for a predefined number of epochs, applying early stopping based on validation loss.
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    early_stopping_patience = 20
    best_val_loss = float('inf')
    patience = 0
    epochs = 150  # Set the number of training epochs.
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_dl, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_dl, loss_fn, device)
        # Logging training and validation metrics.
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Store metrics for later analysis and visualization.
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step()  # Adjust the learning rate based on the scheduler.

        # Check for early stopping condition.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print("Early stopping triggered")
                break

    # Plot training and validation loss and accuracy.
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.show()

    # Save the trained model for future use.
    torch.save(model.state_dict(), os.path.join(models_path, 'dnn_model.pth'))

    return model  # Return the trained model for further use or evaluation.


def dnn_predict(image_dir, labels_file_path=None, loaded_model=None, dtype='val'):
    # Ensure consistency in model architecture before making predictions
    if loaded_model is None:
        loaded_model = ModifiedResNet(num_classes=2).to(device)
        # Load the pre-trained model parameters
        loaded_model.load_state_dict(torch.load(os.path.join(models_path, 'dnn_model.pth')))
    # Prepare the dataset for prediction
    test_dataset = PCOSDataset(data_dir=image_dir, labels_file_path=labels_file_path, transform=transformations)
    # Create a DataLoader for the dataset
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def predict_(model, dataloader, device):
        # Set the model to evaluation mode for predictions
        model.eval()
        predictions = []
        # Disable gradient computation for efficiency
        with torch.no_grad():
            for inputs, _ in dataloader:
                # Transfer inputs to the appropriate device
                inputs = inputs.to(device)
                # Forward pass to compute predictions
                outputs = model(inputs)
                # Get the index of the max log-probability as the prediction
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        return predictions

    # Make predictions using the prepared DataLoader
    predictions = predict_(loaded_model, test_dl, device)
    # Generate and save interpretability plots for the predictions
    plot_interpretability_plots(loaded_model, test_dl, plot_dir, dtype)
    return predictions  # Return the predictions for further analysis

def plot_interpretability_plots(model, dataloader, output_dir, type='val', no_of_plot=20):
    # Set the model to evaluation mode for generating interpretability plots
    model.eval()
    top_images = []  # Initialize a list to store selected images
    with torch.no_grad():
        # Iterate over the DataLoader to fetch images and their predictions
        for images, _ in dataloader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            top_images.extend(images.cpu())
            # Break the loop after collecting the desired number of images
            if len(top_images) >= no_of_plot:
                break

    # Generate and save interpretability plots for each selected image
    for i, image_tensor in enumerate(top_images[:no_of_plot]):
        # Define filenames for different interpretability plots
        cam_filename = f"{output_dir}/{type}_cam_plot_{i + 1}.png"
        occlusion_filename = f"{output_dir}/{type}_occlusion_plot_{i + 1}.png"
        shap_filename = f"{output_dir}/{type}_shap_plot_{i + 1}.png"
        # Generate and save a Class Activation Mapping (CAM) plot
        plot_cam(model, image_tensor, target_layer=model.base_model.layer4[-1], filename=cam_filename)

def get_top_images_and_predictions(model, dataloader, device, top_k=5):
    # Evaluate the model to get predictions along with their confidence scores
    model.eval()
    top_images_info = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            # Apply softmax to compute probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # Get top predictions and their confidence scores
            top_prob, top_pred_classes = torch.topk(probabilities, k=1, dim=1)
            for i in range(images.size(0)):
                # Append image, prediction, and probability to the list
                top_images_info.append((images[i].cpu(), top_pred_classes[i].cpu(), top_prob[i].cpu()))
    # Sort the list by confidence scores in descending order and select top k
    top_images_info.sort(key=lambda x: x[2], reverse=True)
    return top_images_info[:top_k]  # Return top k images with predictions and confidence scores

def save_top_images_with_predictions(top_images_info, output_dir, type='val'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    for i, (image_tensor, pred_class, _) in enumerate(top_images_info):
        # Convert the tensor to a PIL image
        image = transforms.ToPILImage()(image_tensor)
        # Initialize a drawing context
        draw = ImageDraw.Draw(image)
        # Define the text to be drawn on the image
        text = f'Pred: {pred_class.item()}'
        # Draw the text on the image
        draw.text((10, 10), text, fill='red')
        # Save the annotated image
        image.save(os.path.join(output_dir, f"{type}_top_image_{i + 1}.png"))

import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from joblib import dump, load
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm.notebook import tqdm
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and processor from Hugging Face
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def roi_highlighter(img, radious):
    """
    Highlights regions of interest (ROIs) in an image by detecting circles.
    Args:
        img: Input image as a NumPy array.
        radious: The radius for the circle detection, not used in this function but intended for flexibility.
    Returns:
        img: The original image with detected circles highlighted.
    """
    # Convert to grayscale if the image is not already
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image
    gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Detect circles in the blurred image
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                               param1=50, param2=30, minRadius=2, maxRadius=24 * 2)

    # Draw detected circles on the original image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)
            cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 3)
    return img

def extract_lbp_features(image, s=256):
    """
    Extracts Local Binary Pattern (LBP) features from an image.
    Args:
        image: Input image as a NumPy array.
        s: Size of the feature vector (default 256).
    Returns:
        lbp: LBP feature vector of fixed size.
    """
    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply LBP
    lbp = local_binary_pattern(image_gray, P=8, R=1, method="uniform")
    lbp = lbp.flatten()

    # Resize feature vector to a fixed size
    if lbp.size < s:
        lbp = np.concatenate([lbp, np.zeros(s - lbp.size)])
    else:
        lbp = lbp[:s]
    return lbp

def extract_orb_features(image, s):
    """
    Extracts ORB (Oriented FAST and Rotated BRIEF) features from an image.
    Args:
        image: Input image as a NumPy array.
        s: Size of the feature vector.
    Returns:
        descriptors: ORB feature vector of fixed size.
    """
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # Standardizing to a fixed size
    if descriptors is None:
        return np.zeros((1, s))
    descriptors = descriptors.flatten()[:s]
    if descriptors.size < s:
        descriptors = np.concatenate([descriptors, np.zeros(s - descriptors.size)])
    return descriptors

def extract_sift_features(img, size=128):
    """
    Extracts SIFT (Scale-Invariant Feature Transform) features from an image.
    Args:
        img: Input image as a NumPy array.
        size: Size of the feature vector (default 128).
    Returns:
        descriptors: SIFT feature vector or a zero vector if no features are detected.
    """
    # Convert to grayscale if necessary
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    return descriptors if descriptors is not None else np.zeros((1, size))

def extract_clip_features(img):
    """
    Extracts features from an image using the CLIP model.
    Args:
        img: Input image as a NumPy array.
    Returns:
        features: CLIP feature vector as a NumPy array.
    """
    # Convert cv2 BGR image to RGB and to PIL Image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img_rgb)
    processed = processor_clip(images=image_pil, return_tensors="pt")

    # Extract features without gradient calculation
    with torch.no_grad():
        features = model_clip.get_image_features(processed["pixel_values"]).squeeze(0)
    return features.cpu().numpy()

def circularity_descriptor(img, max_features=10):
    """
    Calculates and returns the circularity descriptor for contours found in an image.
    Args:
        img: Input image as a NumPy array.
        max_features: Maximum number of features to return.
    Returns:
        descriptor: Circularity descriptor as a NumPy array.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate circularity for each contour
    circularities = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        circularities.append(circularity)

    # Sort and pad circularity descriptor
    circularities = sorted(circularities, reverse=True)[:max_features]
    descriptor = np.pad(circularities, (0, max_features - len(circularities)), 'constant')
    return descriptor

def extract_glcm_features(img):
    """
    Extracts Gray Level Co-occurrence Matrix (GLCM) features from an image.
    Args:
        img: Input image as a NumPy array.
    Returns:
        features: GLCM feature vector as a NumPy array.
    """
    # Convert to grayscale if necessary
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Calculate GLCM and its properties
    glcm = graycomatrix(gray_img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    features = [graycoprops(glcm, prop)[0, 0] for prop in
                ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]
    return np.array(features)

def extract_resnet50_features(img):
    """
    Extracts features from an image using the ResNet50 model.
    Args:
        img: Input image as a NumPy array.
    Returns:
        features: ResNet50 feature vector as a NumPy array.
    """
    # Load ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    base_model.trainable = False  # Freeze the model

    # Resize and preprocess the image
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    img_array = np.expand_dims(img_rgb, axis=0)
    preprocessed_img = preprocess_input(img_array)

    # Predict features
    features = base_model.predict(preprocessed_img)
    return features.flatten()

def combine_features(image_path, name, dt):
    """
    Combines various features extracted from an image into a single feature vector.
    Args:
        image_path: Path to the input image.
        name: Name of the image (unused in the function but could be used for logging or labeling).
        dt: A descriptor or label for the data type or source (unused but could be used for filtering or separation).
    Returns:
        combined_features: Combined feature vector as a NumPy array.
    """
    # Extract features using different methods
    sift_features = extract_sift_features(image_path, 128).flatten()[:128]
    clip_features = extract_clip_features(image_path).flatten()[:512]
    glcm_features = extract_glcm_features(image_path).flatten()

    # Combine all extracted features into a single vector
    combined_features = np.hstack((sift_features, clip_features, glcm_features))
    return combined_features

def ml_features_extraction():
    """
    Extracts machine learning features from training and validation datasets, standardizes them, and returns the datasets along with labels.
    Returns:
        X_train, y_train, X_val, y_val: Feature matrices and labels for training and validation sets.
    """
    # Load datasets
    train_df = pd.read_csv(train_labels_path)
    val_df = pd.read_csv(val_labels_path)

    # Extract and combine features for each image in the training and validation datasets
    X_train = np.array([combine_features(roi_highlighter(cv2.imread(f'{train_dir}/{row.id}'), 24), row.id, "train") for _, row in tqdm(train_df.iterrows(), total=train_df.shape[0])])
    y_train = train_df['label'].values
    X_val = np.array([combine_features(roi_highlighter(cv2.imread(f'{val_dir}/{row.id}'), 24), row.id, "val") for _, row in tqdm(val_df.iterrows(), total=val_df.shape[0])])
    y_val = val_df['label'].values

    # Standardize the feature matrices
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Save the scaler model for later use
    dump(scaler, os.path.join(models_path, 'scaler_model.joblib'))
    return X_train, y_train, X_val, y_val

def test_features_extraction(scaler=None, test_dir=test_dataset_path):
    """
    Extracts features from the test dataset and standardizes them using a given scaler.
    Args:
        scaler: Scaler model for standardizing features. If None, loads the scaler model from file.
        test_dir: Directory containing the test dataset images.
    Returns:
        X_test: Standardized feature matrix for the test dataset.
    """
    files = os.listdir(test_dir)
    files = [os.path.join(test_dir, i) for i in files]

    # Extract and combine features for each image in the test dataset
    X_test = np.array([combine_features(roi_highlighter(cv2.imread(row), 24), row, "test") for row in tqdm(files, total=len(files))])

    # Load the scaler model if not provided
    if scaler is None:
        scaler = load(os.path.join(models_path, 'scaler_model.joblib'))

    # Standardize the feature matrix
    X_test = scaler.transform(X_test)
    return X_test

import numpy as np
import itertools
from sklearn.metrics import f1_score
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier)
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import NuSVC, SVC
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, Perceptron, SGDClassifier, PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
# Import necessary libraries for machine learning models and metrics.

def ml_model_training_and_prediction(X_train, y_train, X_val):
    """
    Trains multiple machine learning models on training data and predicts on validation data.
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
    Returns:
        predictions: A dictionary with model names as keys and predicted labels as values.
        models: A list of tuples containing model names and the trained model instances.
    """
    # Initialize models with specific parameters
    models = [
        ("ExtraTreesClassifier", ExtraTreesClassifier(**{'max_depth': 25, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 250})),
        ("LabelPropagation", LabelPropagation(**{'gamma': 100, 'kernel': 'rbf', 'n_neighbors': 17})),
        ("LabelSpreading", LabelSpreading()),
        ("LGBMClassifier", LGBMClassifier(**{'boosting_type': 'gbdt', 'learning_rate': 0.35, 'max_depth': 10, 'n_estimators': 300, 'num_leaves': 93})),
        ("AdaBoostClassifier", AdaBoostClassifier()),
        ("QuadraticDiscriminantAnalysis", QuadraticDiscriminantAnalysis(**{'reg_param': 0.0})),
        ("DecisionTreeClassifier", DecisionTreeClassifier(**{'criterion': 'entropy', 'max_depth': 30, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 6, 'splitter': 'best'})),
        ("SGDClassifier", SGDClassifier()),
    ]

    # Train models and make predictions on the validation set
    predictions = {}
    for name, model in models:
        model.fit(X_train, y_train)  # Train the model on the training set
        y_pred = model.predict(X_val)  # Predict on the validation set
        predictions[name] = y_pred  # Store predictions
        print(f"Training: {name}")  # Print the name of the model being trained

    return predictions, models  # Return the predictions and the trained models

def ml_model_pred(models, X_test):
    """
    Predicts labels for test data using pre-trained models.
    Args:
        models: A list of tuples containing model names and trained model instances.
        X_test: Test features.
    Returns:
        predictions: A dictionary with model names as keys and predicted labels as values.
    """
    predictions = {}
    for name, model in models:
        y_pred = model.predict(X_test)  # Predict on the test set
        predictions[name] = y_pred  # Store predictions
    return predictions  # Return the predictions

import itertools
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from PIL import Image, ImageDraw, ImageFont

class PCOSTraning:
    def __init__(self):
        # Initializes the PCOSTraining class with empty models and DNN model placeholders.
        self.models = None
        self.dnn_model = None

    def put_text_on_image(self, image_path, text, output_path):
        """
        Adds text to an image and saves the result.
        Args:
            image_path: Path to the source image.
            text: Text to be added to the image.
            output_path: Path to save the modified image.
        """
        # Load the image from the given path.
        image = Image.open(image_path)
        # Prepare the drawing context to add text.
        draw = ImageDraw.Draw(image)
        # Define the position for the text.
        text_position = (25, 25)
        # Define the text color (RGBA) as semi-transparent white.
        text_color = (255, 255, 255, 128)
        # Draw the text onto the image.
        draw.text(text_position, text, fill=text_color)
        # Save the modified image to the specified output path.
        image.save(output_path, dpi=(600, 600))
        print("File saved successfully:", output_path)

    def plot_top_images(self, image_dir, y_pred, val_labels_path=None, dtype='val'):
        """
        Selects and annotates top images based on prediction results for visualization.
        Args:
            image_dir: Directory containing images to be annotated.
            y_pred: Array of prediction results corresponding to images.
            val_labels_path: Path to the validation labels file (optional).
            dtype: Type of dataset (e.g., 'val' for validation).
        """
        # List all files in the specified directory.
        files = os.listdir(image_dir)
        # Initialize lists to hold selected images for each class.
        df_1, df_0 = [], []
        # If a validation labels path is provided, use it to filter and select images.
        if val_labels_path:
            df = pd.read_csv(val_labels_path)
            df['y_pred'] = y_pred
            df['label'] = df['label'].apply(str)
            df['y_pred'] = df['y_pred'].apply(str)
            # Filter to include only correct predictions.
            df = df[df['label'] == df['y_pred']]
            # Select top images for each class.
            df_1 = df[df['y_pred'] == "1"]['id'].to_list()[:2]
            df_0 = df[df['y_pred'] == "0"]['id'].to_list()[:3]
        else:
            # If no validation labels are provided, use predictions directly to select images.
            df = pd.DataFrame({'id': files, 'label': y_pred})
            df['label'] = df['label'].apply(str)
            df_1 = df[df['label'] == "1"]['id'].to_list()[:2]
            df_0 = df[df['label'] == "0"]['id'].to_list()[:3]
        # Organize selected images by class.
        z = {'class_1': df_1, 'class_0': df_0}
        # Annotate and save selected images for each class.
        for c in z:
            for img in z[c]:
                self.put_text_on_image(
                    os.path.join(image_dir, img),
                    c, os.path.join(plot_dir, f'{dtype}_sample' + img)
                )

    def dnn_and_ml_ensemble_eval(self, predictions, y_val, models):
        """
        Evaluates ensemble predictions combining DNN and ML models and prints classification metrics.
        Args:
            predictions: Dictionary of predictions from various models.
            y_val: Ground truth labels for validation data.
            models: List of trained models for ensemble.
        Returns:
            ensemble_pred: Combined ensemble predictions.
        """
        ensemble_pred = []
        # Iterate through combinations of models for ensemble.
        for j in range(len(models), len(models) + 1):
            comb_scores = {}
            for combo in itertools.combinations([model[0] for model in models], j - 1):
                combo = combo + ("dnn",)  # Include DNN model in the combination.
                # Generate ensemble predictions by majority vote.
                ensemble_pred = np.array([np.bincount(
                    [predictions[model_name][i] for model_name in combo]).argmax() for i in range(len(y_val))])
                f1 = f1_score(y_val, ensemble_pred, average='macro')
                comb_scores[combo] = f1
            # Identify the best combination based on F1 score.
            sorted_comb_scores = sorted(comb_scores.items(), key=lambda x: x[1], reverse=True)
            best_combo, best_score = sorted_comb_scores[0]
            print(f"F1 score: {best_score} with Best combination: {best_combo}")
        # Print classification report for the best ensemble.
        print(classification_report(y_val, ensemble_pred))
        return ensemble_pred

    def dnn_and_ml_ensemble_pred(self, predictions, models):
        """
        Generates ensemble predictions for test data combining DNN and ML models.
        Args:
            predictions: Dictionary of predictions from various models for test data.
            models: List of trained models for ensemble.
        Returns:
            ensemble_pred: Combined ensemble predictions for test data.
        """
        ensemble_pred = []
        # Similar to evaluation, but for generating predictions on test data.
        for j in range(len(models), len(models) + 1):
            for combo in itertools.combinations([model[0] for model in models], j - 1):
                combo = combo + ("dnn",)
                ensemble_pred = np.array([np.bincount(
                    [predictions[model_name][i] for model_name in combo]).argmax() for i in range(len(predictions['dnn']))])
        return ensemble_pred

    def main(self):
        """
        Main function to execute the training and evaluation pipeline for PCOS image classification.
        """
        # Placeholder functions for dataset preparation and DNN training/evaluation.
        prepare_balanced_dataset()
        print("Start Deep Neural Network Model Training....")
        dnn_model = dnn_model_train_and_eval()
        dnn_model = None  # Placeholder for DNN model training.
        dnn_y_pred = dnn_predict(val_dir, val_labels_path, dnn_model, dtype='val')
        print("Start feature extraction for ML models....")
        # Placeholder function for feature extraction.
        X_train, y_train, X_val, y_val = ml_features_extraction()
        print("Start ML models training....")
        # Placeholder function for ML model training and prediction.
        predictions, models = ml_model_training_and_prediction(X_train, y_train, X_val)
        predictions['dnn'] = dnn_y_pred
        # Evaluate ensemble of DNN and ML models.
        ensemble_pred = self.dnn_and_ml_ensemble_eval(predictions, y_val, models)
        self.models = models
        self.dnn_model = dnn_model
        # Visualize top images based on ensemble predictions.
        self.plot_top_images(val_dir, ensemble_pred, val_labels_path, dtype='val')

    def test_label_prediction(self, test_dir=test_dataset_path):
        """
        Predicts labels for test dataset images using the trained ensemble of DNN and ML models.
        Args:
            test_dir: Directory containing test dataset images.
        """
        # Predict labels using the DNN model.
        dnn_test_prediction = dnn_predict(test_dir, labels_file_path=None, loaded_model=self.dnn_model, dtype='test')
        # Extract features for test dataset images.
        X_test = test_features_extraction(scaler=None, test_dir=test_dataset_path)
        # Predict labels using ML models.
        ml_test_predictions = ml_model_pred(self.models, X_test)
        ml_test_predictions['dnn'] = dnn_test_prediction
        # Generate ensemble predictions for test dataset.
        predictions = self.dnn_and_ml_ensemble_pred(ml_test_predictions, self.models)
        # Save predictions to an Excel file.
        df = pd.DataFrame({"S.No.": list(range(1, len(predictions) + 1)), "Image Path": os.listdir(test_dir), "Predicted Class Label": predictions})
        df.to_excel(result_submission_file, index=False)
        # Visualize top images based on test dataset predictions.
        self.plot_top_images(test_dir, predictions, None, dtype='test')

# Instantiate and run the PCOSTraining class.
obj = PCOSTraning()
obj.main()
obj.test_label_prediction(test_dataset_path)



