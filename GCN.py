# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Extract dataset (ignoring _MACOSX and .DS_Store)
import zipfile
import os

zip_path = "/content/drive/MyDrive/augmented_db.zip"
extract_path = "/content/dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Remove unwanted files (_MACOSX and .DS_Store)
!rm -rf /content/dataset/__MACOSX
!find /content/dataset -name ".DS_Store" -delete

# Define dataset path
dataset_path = "/content/dataset/augmented_db"
print("Dataset extracted successfully!")

# Step 3: Load MRI images and extract histogram features
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Categories and image paths
categories = ['glioma', 'meningioma', 'pituitary', 'healthy']
image_size = (128, 128)  # Resize images to 128x128 for uniformity

def extract_histogram_features(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])  # Histogram for grayscale
    hist = cv2.normalize(hist, hist).flatten()  # Normalize
    return hist

data = []
labels = []

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    label = categories.index(category)  # Assign label (0,1,2,3)

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

        if image is not None:
            image = cv2.resize(image, image_size)  # Resize
            features = extract_histogram_features(image)  # Extract features
            data.append(features)
            labels.append(label)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

print(f"Feature extraction completed. Dataset size: {data.shape}")


# Step 4: Split the dataset into Training (80%) and Validation (20%)
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

print(f"Training set size: {X_train.shape}, Validation set size: {X_val.shape}")


# Step 5: Install required dependencies
!pip install torch-geometric torch-sparse torch-scatter torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html



# Step 6: Construct Graph from Features using k-NN (Fixed Version)
import torch
import networkx as nx
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp

# Construct adjacency matrix using k-NN
k = 5  # Number of neighbors
adjacency_matrix = kneighbors_graph(X_train, k, mode='connectivity', include_self=True)

# Convert to NetworkX Graph using the updated method
graph = nx.from_scipy_sparse_array(adjacency_matrix)

# Convert edges to PyTorch format
edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

# Convert features and labels to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Define PyG Data object
train_data = Data(x=X_train_tensor, edge_index=edge_index, y=y_train_tensor)
val_data = Data(x=X_val_tensor, edge_index=edge_index, y=y_val_tensor)

print("Graph construction completed successfully!")



import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define the GCN Model
class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.dropout = nn.Dropout(0.3)  # Dropout rate to prevent overfitting

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize Model
num_features = X_train.shape[1]  # Number of extracted features
num_classes = len(set(y_train))  # Number of classes
model = GCN(num_features, num_classes)

print("GCN Model Defined!")

import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp

# Function to create graph data and return adjacency matrix
def create_graph(X, y, k=5):
    # Construct adjacency matrix using k-NN
    adjacency_matrix = kneighbors_graph(X, k, mode='connectivity', include_self=True)

    # Convert adjacency matrix to NetworkX graph
    graph = nx.from_scipy_sparse_array(adjacency_matrix)

    # Convert edges to PyTorch format
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Convert features and labels to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Create PyG Data object
    graph_data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)

    return graph_data, adjacency_matrix.toarray()  # Convert sparse matrix to dense array for printing

# Create separate graphs and adjacency matrices for train and validation
train_data, train_adj_matrix = create_graph(X_train, y_train, k=5)
val_data, val_adj_matrix = create_graph(X_val, y_val, k=5)

# Print adjacency matrices
print("Train Adjacency Matrix:\n", train_adj_matrix)
print("\nValidation Adjacency Matrix:\n", val_adj_matrix)

print("Graph construction completed successfully!")


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define the GCN Model
class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.dropout = nn.Dropout(0.3)  # Dropout to prevent overfitting

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize Model
num_features = train_data.x.shape[1]  # Number of extracted features
num_classes = len(set(y_train))  # Number of MRI categories
model = GCN(num_features, num_classes)

print("GCN Model Defined!")



import torch.optim as optim

# Move model & data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
train_data = train_data.to(device)
val_data = val_data.to(device)

# Define Optimizer & Loss Function
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# Manually set number of epochs
num_epochs = 600 # Change this as needed

# Store training history
train_losses = []
val_losses = []
train_accs = []
val_accs = []

# Training Loop
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    loss = criterion(out, train_data.y)
    loss.backward()
    optimizer.step()

    # Calculate training accuracy
    pred = out.argmax(dim=1)
    train_acc = (pred == train_data.y).sum().item() / train_data.y.size(0)

    # Validation Step
    model.eval()
    with torch.no_grad():
        val_out = model(val_data)
        val_loss = criterion(val_out, val_data.y)
        val_pred = val_out.argmax(dim=1)
        val_acc = (val_pred == val_data.y).sum().item() / val_data.y.size(0)

    # Store values
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Print progress
    print(f"Epoch [{epoch}/{num_epochs}] - Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

print("Training Completed!")


from sklearn.metrics import confusion_matrix, classification_report

# Convert predictions and ground truths to numpy arrays
y_true = val_data.y.cpu().numpy()
y_pred = model(val_data).argmax(dim=1).cpu().numpy()

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Compute precision, recall, and F1-score
print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))


import matplotlib.pyplot as plt

# Plot Accuracy Graph
plt.figure(figsize=(10, 4))
plt.plot(range(1, num_epochs + 1), train_accs, label="Train Accuracy", marker="o")
plt.plot(range(1, num_epochs + 1), val_accs, label="Validation Accuracy", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Epoch vs Accuracy")
plt.legend()
plt.grid()
plt.show()

# Plot Loss Graph
plt.figure(figsize=(10, 4))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epoch vs Loss")
plt.legend()
plt.grid()
plt.show()


# Compute overall accuracy
correct_predictions = (y_pred == y_true).sum()
total_samples = y_true.shape[0]
overall_accuracy = correct_predictions / total_samples * 100

print(f"Overall Model Accuracy: {overall_accuracy:.2f}%")




import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Define class labels
class_labels = ["Glioma", "Meningioma", "Pituitary", "Healthy"]

# Plot confusion matrix as heatmap
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - GCN MRI Classification")
plt.show()
