import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class Lightweight1DCNN(nn.Module):
    """Lightweight 1D CNN (~85K params)."""
    def __init__(self, input_dim=64, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * (input_dim // 4), 128)  # Adjust based on pooling
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
#     model = Lightweight1DCNN()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
#     val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size)
    
#     for epoch in range(epochs):
#         model.train()
#         for xb, yb in train_loader:
#             optimizer.zero_grad()
#             out = model(xb)
#             loss = criterion(out, yb)
#             loss.backward()
#             optimizer.step()
        
#         # Val accuracy
#         model.eval()
#         correct = 0
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 out = model(xb)
#                 correct += (out.argmax(1) == yb).sum().item()
#         acc = correct / len(y_val)
#         print(f'Epoch {epoch}: Val Acc {acc:.4f}')
    
#     return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X_test).float())
        acc = (out.argmax(1) == torch.tensor(y_test).long()).float().mean().item()
    return acc


def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
    model = Lightweight1DCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Good default LR
    
    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        # === TRAINING PHASE ===
        model.train()                                      # <--- CRITICAL
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # === VALIDATION PHASE ===
        model.eval()                                       # Switch to eval
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        
        acc = correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

        # === CREATE LOSS HEATMAP ===
    print("\nGenerating training loss heatmap...")
    
    # Create a 2D array: rows = epochs, columns = dummy for heatmap visualization
    loss_matrix = np.array(train_losses).reshape(-1, 1)  # (epochs, 1)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(loss_matrix, aspect='auto', cmap='YlOrRd_r', interpolation='nearest')
    plt.colorbar(label='Training Loss')
    plt.title('Training Loss Heatmap Over Epochs\n(Lower = Better)', fontsize=16)
    plt.xlabel('Epoch Progress')
    plt.ylabel('Epoch')
    plt.yticks(ticks=range(0, epochs, max(1, epochs//10)), labels=range(1, epochs+1, max(1, epochs//10)))
    
    return model