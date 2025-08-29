import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, recall_score


class EGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wz = nn.Linear(input_size, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wh = nn.Linear(input_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev):
        z = torch.sigmoid(self.Wz(x) + self.Uz(h_prev))
        r = torch.sigmoid(self.Wr(x) + self.Ur(h_prev))
        h_tilde = torch.tanh(self.Wh(x) + self.Uh(r * h_prev))
        h = (1 - z) * h_prev + z * h_tilde
        return h


class EGRUBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.cell = EGRUCell(in_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden)
        self.res_proj = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()

    def forward(self, x):
        B, T, D = x.shape
        device = x.device

        res = self.res_proj(x)
        x_norm = self.ln(res)

        h = torch.zeros(B, self.cell.hidden_size, device=device)
        outs = []
        for t in range(T):
            h = self.cell(x_norm[:, t, :], h)
            outs.append(h)

        y_seq = torch.stack(outs, dim=1)   # (B,T,H)
        y = self.dropout(y_seq) + res      # residual
        return y


class MultiHeadTemporalAttentionPool(nn.Module):
    def __init__(self, hidden: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.W = nn.Linear(hidden, hidden * num_heads)
        self.v = nn.Parameter(torch.randn(num_heads, hidden))

    def forward(self, y):
        B, T, H = y.shape
        y_proj = self.W(y).view(B, T, self.num_heads, H)
        scores = torch.einsum("bthd,hd->bth", torch.tanh(y_proj), self.v)
        att = torch.softmax(scores, dim=1)             # (B,T,num_heads)
        ctx = torch.einsum("bth,bthd->bhd", att, y_proj)
        ctx = ctx.reshape(B, self.num_heads * H)       # (B,num_heads*H)
        return ctx, att

class HighPerformanceEGRU(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=256, num_layers=3, dropout=0.3, num_heads=4):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads

        self.in_ln = nn.LayerNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, hidden)

        self.fw_blocks = nn.ModuleList(
            [EGRUBlock(hidden, hidden, dropout=dropout) for _ in range(num_layers)]
        )

        self.attn_pool = MultiHeadTemporalAttentionPool(hidden, num_heads=num_heads)

        fused_dim = hidden * num_heads  # 1024

        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim * 4, num_classes),
        )

    def forward(self, x, return_attention=False):
        x = self.in_proj(self.in_ln(x))   # (B,T,H)

        for blk in self.fw_blocks:
            x = blk(x)                    # (B,T,H)

        ctx, att = self.attn_pool(x)      # ctx=(B, 1024)
        logits = self.head(ctx)           # (B,C)

        return (logits, att) if return_attention else logits

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
