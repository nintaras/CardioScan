import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.dataset import ECGDataset
from src.model import CardioNet1D

def train():
    # 1. Επιλογή Hardware: Χρήση Apple Silicon GPU (mps) αν είναι διαθέσιμο, αλλιώς CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"-> Εκπαίδευση στη συσκευή: {device}")

    # 2. Φόρτωση του Dataset (πραγματικά περιστατικά ασθενών 100 και 106)
    dataset = ECGDataset(records=['100', '106'])
    
    # Διαχωρισμός 80% Train / 20% Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3. Αντιμετώπιση Class Imbalance: Υπολογισμός βαρών κλάσεων
    total_samples = len(dataset.labels)
    n_normal = (dataset.labels == 0).sum().item()
    n_pvc = (dataset.labels == 1).sum().item()
    
    weight_normal = total_samples / (2.0 * n_normal)
    weight_pvc = total_samples / (2.0 * n_pvc)
    class_weights = torch.tensor([weight_normal, weight_pvc], dtype=torch.float32).to(device)

    # 4. Αρχικοποίηση Μοντέλου, Loss & Optimizer
    model = CardioNet1D(in_channels=1, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop
    epochs = 10
    best_val_acc = 0.0

    print("\n-> Έναρξη Εκπαίδευσης...")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        # Phase 1: Train
        model.train()
        train_loss, train_correct = 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()

        epoch_train_loss = train_loss / train_size
        epoch_train_acc = (train_correct / train_size) * 100

        # Phase 2: Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * len(y_batch)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()

        epoch_val_loss = val_loss / val_size
        epoch_val_acc = (val_correct / val_size) * 100

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.1f}%")

        # Αποθήκευση του καλύτερου μοντέλου
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_cardioscan_model.pth')

    print("-" * 55)
    print(f"-> Εκπαίδευση ολοκληρώθηκε! Καλύτερο Validation Accuracy: {best_val_acc:.2f}%")
    print("-> Τα βάρη αποθηκεύτηκαν στο 'best_cardioscan_model.pth'.")

if __name__ == '__main__':
    train()
