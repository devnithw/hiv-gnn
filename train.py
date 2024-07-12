import torch 
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

import numpy as np
from tqdm import tqdm

from model import GNN
from dataset import HIVDataset


# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variables
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
TRAIN_CSV = 'train.csv'
TEST_CSV = 'test.csv'

# Hyperparameters
BATCH_SIZE = 32
FEATURE_SIZE = 30


# Load datasets
print("Loading datasets...")
train_dataset = HIVDataset(root=TRAIN_DIR, filename=TRAIN_CSV)
print("Train dataset loaded.")
test_dataset = HIVDataset(root=TEST_DIR, filename=TEST_CSV)
print("Test dataset loaded.")

# Prepare Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Dataloaders ready.")

# Ensure feature size is 30
assert train_dataset[0].x.shape[1] == FEATURE_SIZE, "Feature size mismatch."

# Load model
model = GNN(feature_size=FEATURE_SIZE).to(device)
print(f"Model successfully loaded and sent to device: {device}")
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Count number of trainable parameters
print(f"Number of trainable parameters: {n_params}")





