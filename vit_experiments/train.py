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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# # Instantiate model
# model = VisionTransformer(
#     FEATURE_DIM, SEQ_LEN, CHANNELS, NUM_CLASSES,
#     EMBED_DIM, DEPTH, NUM_HEADS, MLP_DIM, DROP_RATE,
#     REGION_SIZE, FEATURE_ARRANGEMENT, DATA_STRUCTURE
# ).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# # Training loop
# def train(model, loader, optimizer, criterion):
#     model.train()
#     total_loss, correct = 0, 0
#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         out = model(x)
#         loss = criterion(out, y)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * x.size(0)
#         correct += (out.argmax(1) == y).sum().item()

#     return total_loss / len(loader.dataset), correct / len(loader.dataset)


# # @title
# # Training
# train_accuracies, test_accuracies = [], []
# # best_test_acc = 1 # Disables model saving
# best_test_acc = 0
# really_best = 0
# best_model_path = os.path.join(GOOGLE_DRIVE_PATH_OUTPUTFILE, f"best_model_vit_{FEATURE_ARRANGEMENT}_{REGION_SIZE}_{DATA_STRUCTURE}.pth")
# best_model_test_path = os.path.join(GOOGLE_DRIVE_PATH_OUTPUTFILE, f"best_model_vit_{FEATURE_ARRANGEMENT}_{REGION_SIZE}_{DATA_STRUCTURE}_test_based.pth")

# for epoch in range(EPOCHS):
#     train_loss, train_acc = train(model, train_loader, optimizer, criterion)
#     val_acc = evaluate(model, val_loader)
#     test_acc = evaluate(model, dataloader_test)
#     train_accuracies.append(train_acc)
#     test_accuracies.append(val_acc)
#     print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

#     # Save the best model
#     if val_acc > best_test_acc:
#         best_test_acc = val_acc
#         torch.save(model.state_dict(), best_model_path)
#         print(f"Best model saved to {best_model_path}")

#     if test_acc > really_best:
#         really_best = test_acc
#         torch.save(model.state_dict(), best_model_test_path)
#         print(f"Best test model saved to {best_model_test_path}")