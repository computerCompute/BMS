import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, recall_score



class NpySeqDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)   # (N, T, F)
        self.y = np.load(y_path)   # (N,)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model():
    BATCH = 32
    EPOCHS = 100
    LR = 1e-3
    C = 2   # normal / falling

    train_ds = NpySeqDataset("/root/ByteTrack/dataset/zong2/tid1_x.npy", "/root/ByteTrack/dataset/zong2/tid1_y.npy")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    _, T, F = train_ds.X.shape
    model = HighPerformanceEGRU(in_dim=F, num_classes=C, hidden=256, num_layers=3, dropout=0.3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    weights = torch.tensor([0.6925, 1.7987], device=device)  # normal=1.0, falling=2.0
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)          # (B, C)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        acc = correct / total


        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        recall = recall_score(y_true, y_pred, pos_label=1)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss {total_loss/total:.4f} | Acc {acc:.4f} | Recall(falling) {recall:.4f}")


        print(f"Epoch {epoch+1}/{EPOCHS} | Loss {total_loss/total:.4f} | Acc {acc:.4f}")

    return model

if __name__ == "__main__":
    trained_model = train_model()
    torch.save(trained_model.state_dict(), "egru_fall_detection_f4.pth")
