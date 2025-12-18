import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
from sklearn.model_selection import train_test_split
from utils.vit_setup_loader import load_config
import yaml
from models.vit import VisionTransformer
from data_loader.serra_do_cipo.loader import load_sc_data, load_raw_data
from data_loader.serra_do_cipo.loader import create_loaders as sc_create_loaders

from utils.evaluate import evaluate
from utils.create_output_folder import create_folder_if_needed
from utils.metrics import save_metrics
from utils.visualize_serra_cipo import generate_and_save_visualizations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Read data
data = load_raw_data()
X_train, Y_train, X_val, Y_val, X_test, Y_test = load_sc_data(data)
train_loader, val_loader, test_loader  = sc_create_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test)

loaded_config = load_config('./vit_experiments/setups/serra_do_cipo/setup1.yaml')
print(loaded_config)

EXPERIMENT_NAME = loaded_config['experiment_name']
REGION_SIZE = loaded_config['custom_setup']['region']
SEQUENCE_ORDER = loaded_config['custom_setup']['sequence_order']
FEATURE_ARRANGEMENT = loaded_config['custom_setup']['arrangement']
NORMALIZED_RGB = loaded_config['custom_setup']['normalized_rgb']
CHANNELS = loaded_config['model']['channels']

SEQ_LEN = loaded_config['training']['seq_len']
EPOCHS = loaded_config['training']['epochs']
FEATURE_DIM = REGION_SIZE * CHANNELS if SEQUENCE_ORDER == 'TR' else CHANNELS * SEQ_LEN
NUM_CLASSES = loaded_config['training']['num_classes']

NUM_HEADS = loaded_config['model']['num_heads']
EMBED_DIM = loaded_config['model']['embed_dim']
DEPTH = loaded_config['model']['depth']
MLP_DIM = loaded_config['model']['mlp']
DROP_RATE = loaded_config['model']['drop_rate']
LEARNING_RATE = float(loaded_config['training']['learning_rate'])

# Instantiate model
model = VisionTransformer(
    FEATURE_DIM, SEQ_LEN, CHANNELS, NUM_CLASSES,
    EMBED_DIM, DEPTH, NUM_HEADS, MLP_DIM, DROP_RATE,
    REGION_SIZE, FEATURE_ARRANGEMENT, SEQUENCE_ORDER, 
    NORMALIZED_RGB
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# # Training loop
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)

PATH_OUTPUTFILE = '/home/alangomes/data/Dados_serra_cipo/Dados_serra_cipó/output_vit_experiments'

EXPERIMENT_OUTPUT_PATH = create_folder_if_needed(PATH_OUTPUTFILE, EXPERIMENT_NAME)

# @title
# Training
train_accuracies, test_accuracies = [], []
# best_test_acc = 1 # Disables model saving
best_train_acc = 0
best_val_acc = 0
best_test_acc = 0

overall_train_acc = 0
overall_val_acc = 0
overall_best_test_acc = 0

final_train_acc = 0
final_val_acc = 0
final_test_acc = 0

best_model_path = os.path.join(EXPERIMENT_OUTPUT_PATH, f"best_model_vit_{FEATURE_ARRANGEMENT}_{REGION_SIZE}_{SEQUENCE_ORDER}.pth")
best_model_test_path = os.path.join(EXPERIMENT_OUTPUT_PATH, f"best_model_vit_{FEATURE_ARRANGEMENT}_{REGION_SIZE}_{SEQUENCE_ORDER}_test_based.pth")
final_model_path = os.path.join(EXPERIMENT_OUTPUT_PATH, f"final_model_vit_{FEATURE_ARRANGEMENT}_{REGION_SIZE}_{SEQUENCE_ORDER}.pth")

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_acc = evaluate(model, val_loader, device)
    test_acc = evaluate(model, test_loader, device)
    train_accuracies.append(train_acc)
    test_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

    # Save the best model based on val acc
    if val_acc > best_val_acc:
        best_test_acc = test_acc
        best_train_acc = train_acc
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved to {best_model_path}")

    if test_acc > overall_best_test_acc:
        overall_best_test_acc = test_acc
        overall_train_acc = train_acc
        overall_val_acc = val_acc
        torch.save(model.state_dict(), best_model_test_path)
        print(f"Best test model saved to {best_model_test_path}")

final_train_acc = train_acc
final_val_acc = val_acc
final_test_acc = test_acc
torch.save(model.state_dict(), final_model_path)
print(f"Best test model saved to {final_model_path}")

data_best_val_acc_based = {
    "test_acc": best_test_acc,
    "val_acc": best_val_acc,
    "train_acc": best_train_acc
}

data_overall_test_acc_based = {
    "test_acc": overall_best_test_acc,
    "val_acc": overall_val_acc,
    "train_acc": overall_train_acc
}

data_final_accs = {
    "test_acc": final_test_acc,
    "val_acc": final_val_acc,
    "train_acc": final_train_acc
}
save_metrics(EXPERIMENT_OUTPUT_PATH, data_final_accs, data_best_val_acc_based, data_overall_test_acc_based)

input_path = '/home/alangomes/data/Dados_serra_cipo/Dados_serra_cipó/Imagens_Cedro_ePhenoWS'
output_path = EXPERIMENT_OUTPUT_PATH

generate_and_save_visualizations(
    data, 
    Y_test, 
    model, 
    test_loader, 
    device, 
    input_path, 
    output_path
)