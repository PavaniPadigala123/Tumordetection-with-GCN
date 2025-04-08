from google.colab import drive
import zipfile
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
zip_path = "/content/drive/MyDrive/augmented_db.zip"
extract_path = "/content/augmented_db"

# Extract dataset (excluding _MACOSX and .DS_Store)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for file in zip_ref.namelist():
        if not file.startswith("__MACOSX") and not file.endswith(".DS_Store"):
            zip_ref.extract(file, extract_path)

print("Dataset extracted successfully!")
from google.colab import drive
drive.mount('/content/drive')
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),



    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
dataset_path = "/content/augmented_db/augmented_db"
dataset = ImageFolder(root=dataset_path, transform=transform)

# Split dataset (80% training, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Dataset loaded successfully!")
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn

# Enable Faster Computation
torch.backends.cudnn.benchmark = True  # Speeds up convolutions

# Load Pretrained ResNet-18
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18 = nn.Sequential(*list(resnet18.children())[:-1])  # Remove classification layer
resnet18.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# Use Torch Compile for Speedup (if PyTorch 2.0+)
if hasattr(torch, 'compile'):
    resnet18 = torch.compile(resnet18)

# Function for Fast Feature Extraction
def extract_features(dataloader):
    features, labels = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):  # Mixed Precision for Speed
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            output = resnet18(images).squeeze(-1).squeeze(-1)  # Flatten
            features.append(output.cpu())
            labels.append(targets.cpu())

    return torch.cat(features), torch.cat(labels)

# Extract Features for Training & Validation
train_features, train_labels = extract_features(train_loader)
val_features, val_labels = extract_features(val_loader)

print("🚀 Fast Feature Extraction Completed Successfully!")
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
import scipy.sparse as sp
import networkx as nx

# ✅ Step 1: Reduce Feature Dimensions to Prevent Memory Overflow
pca = PCA(n_components=128)  # Reduce feature size to 128 dimensions
train_features_np = pca.fit_transform(train_features.numpy())
val_features_np = pca.transform(val_features.numpy())

print(f"🔹 Reduced Feature Matrix Shape (Train): {train_features_np.shape}")
print(f"🔹 Reduced Feature Matrix Shape (Validation): {val_features_np.shape}")

# ✅ Step 2: Construct Sparse Adjacency Matrices using KNN (k=5)
k = 5
adj_train_sparse = kneighbors_graph(train_features_np, k, mode="connectivity", include_self=True)
adj_val_sparse = kneighbors_graph(val_features_np, k, mode="connectivity", include_self=True)

# ✅ Step 3: Convert Sparse Matrices to Efficient PyTorch Sparse Tensors
adj_train_tensor = torch.sparse.FloatTensor(
    torch.tensor(adj_train_sparse.nonzero(), dtype=torch.long),
    torch.tensor(adj_train_sparse.data, dtype=torch.float32),
    torch.Size(adj_train_sparse.shape)
)

adj_val_tensor = torch.sparse.FloatTensor(
    torch.tensor(adj_val_sparse.nonzero(), dtype=torch.long),
    torch.tensor(adj_val_sparse.data, dtype=torch.float32),
    torch.Size(adj_val_sparse.shape)
)

# ✅ Step 4: Convert Sparse Matrices to NetworkX Graphs
graph_train = nx.from_scipy_sparse_array(adj_train_sparse)
graph_val = nx.from_scipy_sparse_array(adj_val_sparse)

# ✅ Step 5: Print Adjacency & Feature Matrices
print("\n🔹 Adjacency Matrix (Train):")
print(adj_train_sparse.toarray())  # Convert sparse to dense ONLY for small preview

print("\n🔹 Adjacency Matrix (Validation):")
print(adj_val_sparse.toarray())  # Convert sparse to dense ONLY for small preview

print("\n🔹 Feature Matrix (Train):")
print(train_features_np)

print("\n🔹 Feature Matrix (Validation):")
print(val_features_np)

print("\n✅ Graph Construction Completed! Adjacency & Feature Matrices Printed Without Crashing.")
# Install PyTorch Geometric and dependencies
!pip install torch-geometric
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.optim as optim

# ✅ GCN Model Definition
class GCNModel(nn.Module):
    def _init_(self, in_features, out_features):
        super(GCNModel, self)._init_()

        # Define layers
        self.conv1 = pyg_nn.GCNConv(in_features, 64)  # GCN layer 1
        self.conv2 = pyg_nn.GCNConv(64, 32)  # GCN layer 2
        self.fc = nn.Linear(32, out_features)  # Fully connected layer for final classification

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x

# ✅ Initialize the model
in_features = 128  # PCA reduced feature size
out_features = 4   # 4 classes: glioma, meningioma, pituitary, healthy
model = GCNModel(in_features, out_features)
from torch_geometric.data import Data

# Convert train features to PyTorch tensors
train_features_tensor = torch.tensor(train_features_np, dtype=torch.float32)
val_features_tensor = torch.tensor(val_features_np, dtype=torch.float32)

# Convert adjacency matrices to edge_index format for PyG
def adjacency_to_edge_index(adj_tensor):
    row, col = adj_tensor._indices()
    return torch.stack([row, col], dim=0)

train_edge_index = adjacency_to_edge_index(adj_train_tensor)
val_edge_index = adjacency_to_edge_index(adj_val_tensor)

# Create Data objects for PyTorch Geometric
train_data = Data(x=train_features_tensor, edge_index=train_edge_index, y=train_labels)
val_data = Data(x=val_features_tensor, edge_index=val_edge_index, y=val_labels)

# Display data object structure
print(train_data)
import torch
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

class GCNModel(torch.nn.Module):
    def _init_(self, in_channels, out_channels):
        super(GCNModel, self)._init_()
        # First Graph Convolutional Layer
        self.conv1 = pyg_nn.GCNConv(in_channels, 64)
        # Second Graph Convolutional Layer
        self.conv2 = pyg_nn.GCNConv(64, out_channels)

    def forward(self, data):
        # Get features and edge index
        x, edge_index = data.x, data.edge_index
        # First Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # Second Convolution Layer
        x = self.conv2(x, edge_index)
        return x
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Hyperparameters
learning_rate = 0.01
epochs = 100  # You can set this dynamically
in_channels = train_features_tensor.shape[1]  # Number of features per node
out_channels = len(torch.unique(train_labels))  # Number of classes (labels)

# Initialize the model, optimizer, and loss function
model = GCNModel(in_channels, out_channels)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Move model to device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
train_data.to(device)
val_data.to(device)
# Tracking variables for plotting
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass for training
    train_out = model(train_data)
    train_loss = criterion(train_out, train_data.y)
    train_loss.backward()
    optimizer.step()

    # Evaluate training accuracy
    _, train_pred = torch.max(train_out, dim=1)
    train_acc = accuracy_score(train_data.y.cpu(), train_pred.cpu())

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_out = model(val_data)
        val_loss = criterion(val_out, val_data.y)
        _, val_pred = torch.max(val_out, dim=1)
        val_acc = accuracy_score(val_data.y.cpu(), val_pred.cpu())

    # Track metrics
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    # Print progress
    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {train_loss.item():.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")
import matplotlib.pyplot as plt

# Plot the training and validation accuracy and loss
plt.figure(figsize=(12, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
plt.plot(range(epochs), val_accuracies, label='Validation Accuracy')
plt.title('Epoch vs Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_losses, label='Train Loss')
plt.plot(range(epochs), val_losses, label='Validation Loss')
plt.title('Epoch vs Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
from sklearn.metrics import accuracy_score

# Set model to evaluation mode
model.eval()
with torch.no_grad():
    val_out = model(val_data)
    _, val_pred = torch.max(val_out, dim=1)

# Convert to CPU
val_pred = val_pred.cpu()
val_true = val_data.y.cpu()

# Compute Overall Accuracy
overall_accuracy = accuracy_score(val_true.numpy(), val_pred.numpy())
print(f"Overall Model Accuracy: {overall_accuracy:.4f}")
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Confusion Matrix
model.eval()
with torch.no_grad():
    val_out = model(val_data)
    _, val_pred = torch.max(val_out, dim=1)

# Convert predictions and labels to CPU for metrics
val_pred = val_pred.cpu()
val_true = val_data.y.cpu()

# Compute confusion matrix
cm = confusion_matrix(val_true.numpy(), val_pred.numpy())

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Glioma', 'Meningioma', 'Pituitary'], yticklabels=['Healthy', 'Glioma', 'Meningioma', 'Pituitary'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification report
print(classification_report(val_true.numpy(), val_pred.numpy(), target_names=['Healthy', 'Glioma', 'Meningioma', 'Pituitary']))
