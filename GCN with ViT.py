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

print("✅ Dataset extracted successfully!")
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

print("✅ Dataset loaded successfully!")
import torchvision.models as models
import torch.nn as nn

# Load Pretrained ViT
vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
vit_model.heads = nn.Identity()  # Remove classification head
vit_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# Feature Extraction Function
def extract_features(dataloader):
    features, labels = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            output = vit_model(images)  # Extract features
            features.append(output.cpu())
            labels.append(targets.cpu())

    return torch.cat(features), torch.cat(labels)

# Extract Features for Training & Validation
train_features, train_labels = extract_features(train_loader)
val_features, val_labels = extract_features(val_loader)

print("✅ ViT Feature Extraction Completed!")
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
import torch

# Reduce Feature Dimensions
pca = PCA(n_components=128)
train_features_np = pca.fit_transform(train_features.numpy())
val_features_np = pca.transform(val_features.numpy())

print(f"🔹 Reduced Feature Matrix Shape (Train): {train_features_np.shape}")
print(f"🔹 Reduced Feature Matrix Shape (Validation): {val_features_np.shape}")

# Create Adjacency Matrices using KNN
k = 5
adj_train_sparse = kneighbors_graph(train_features_np, k, mode="connectivity", include_self=True)
adj_val_sparse = kneighbors_graph(val_features_np, k, mode="connectivity", include_self=True)

# Convert Sparse Matrices to PyTorch Tensors
adj_train_tensor = torch.tensor(adj_train_sparse.toarray(), dtype=torch.float32)
adj_val_tensor = torch.tensor(adj_val_sparse.toarray(), dtype=torch.float32)

print("✅ Graph Construction Completed!")

# Print adjacency matrices
print("🔹 Adjacency Matrix (Train):")
print(adj_train_tensor)

print("🔹 Adjacency Matrix (Validation):")
print(adj_val_tensor)
from torch_geometric.data import Data

# Convert adjacency matrix to edge index format
def adjacency_to_edge_index(adj_tensor):
    indices = torch.nonzero(adj_tensor, as_tuple=False).t()
    return indices

train_edge_index = adjacency_to_edge_index(adj_train_tensor)
val_edge_index = adjacency_to_edge_index(adj_val_tensor)

# Create PyTorch Geometric Data objects
train_data = Data(x=torch.tensor(train_features_np, dtype=torch.float32), edge_index=train_edge_index, y=train_labels)
val_data = Data(x=torch.tensor(val_features_np, dtype=torch.float32), edge_index=val_edge_index, y=val_labels)

print("✅ GCN Data Objects Created!")
!pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

class GCNModel(torch.nn.Module):
    def _init_(self, in_channels, out_channels, dropout_rate=0.3):
        super(GCNModel, self)._init_()
        self.conv1 = pyg_nn.GCNConv(in_channels, 64)
        self.conv2 = pyg_nn.GCNConv(64, out_channels)
        self.dropout = dropout_rate  # Dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout
        x = self.conv2(x, edge_index)
        return x
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Model Setup
learning_rate = 0.01
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCNModel(in_channels=128, out_channels=4, dropout_rate=0.3).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Added L2 Regularization
criterion = torch.nn.CrossEntropyLoss()

train_data.to(device)
val_data.to(device)

# Training Loop
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    train_out = model(train_data)
    train_loss = criterion(train_out, train_data.y)
    train_loss.backward()
    optimizer.step()

    _, train_pred = torch.max(train_out, dim=1)
    train_acc = accuracy_score(train_data.y.cpu(), train_pred.cpu())

    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(val_data)
        val_loss = criterion(val_out, val_data.y)
        _, val_pred = torch.max(val_out, dim=1)
        val_acc = accuracy_score(val_data.y.cpu(), val_pred.cpu())

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss={train_loss.item():.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss.item():.4f}, Val Acc={val_acc:.4f}")
# @title Default title text
import matplotlib.pyplot as plt

# Plot Accuracy
plt.figure(figsize=(10,5))
plt.plot(train_accuracies, label='Train Accuracy', marker='o')
plt.plot(val_accuracies, label='Validation Accuracy', marker='s')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Epoch vs Accuracy")
plt.grid()
plt.show()

# Plot Loss
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='s')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Epoch vs Loss")
plt.grid()
plt.show()
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Set model to evaluation mode
model.eval()

# Get predictions for validation data
with torch.no_grad():
    val_out = model(val_data)
    _, val_pred = torch.max(val_out, dim=1)

# Compute Confusion Matrix
cm = confusion_matrix(val_data.y.cpu(), val_pred.cpu())

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Healthy', 'Glioma', 'Meningioma', 'Pituitary'], yticklabels=['Healthy', 'Glioma', 'Meningioma', 'Pituitary'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
# Print Classification Report
print("\nClassification Report:\n")
print(classification_report(val_data.y.cpu(), val_pred.cpu(), target_names=['Healthy', 'Glioma', 'Meningioma', 'Pituitary']))

# Compute & Print Overall Accuracy
accuracy = accuracy_score(val_data.y.cpu(), val_pred.cpu())
print(f"\n✅ Model Accuracy: {accuracy:.4f}")
