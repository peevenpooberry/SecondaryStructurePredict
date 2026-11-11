import copy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from Model_Development.softmax_ss import EarlyStopping

PAD_LABEL = -100  # ignored by CrossEntropyLoss


class ChainDataset(Dataset):
    """One item per chain: (L, 21) one-hot sequence and (L,) integer labels."""

    def __init__(self, split_df, encode_seq, encode_labels):
        self.x = [encode_seq(s) for s in split_df["sequence"]]
        self.y = [encode_labels(s) for s in split_df["structure"]]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def collate(batch):
    """Pad chains in a batch to the same length. Padded inputs are all-zero rows,
    padded labels are PAD_LABEL so the loss ignores them."""
    xs, ys = zip(*batch)
    x = pad_sequence(xs, batch_first=True)                              # (B, L, 21)
    y = pad_sequence(ys, batch_first=True, padding_value=PAD_LABEL)     # (B, L)
    return x, y


class CNN1D(nn.Module):
    """Three conv layers (kernels 7, 7, 5 -> receptive field of 17 residues),
    then a 1x1 conv that outputs 3 scores per residue."""

    def __init__(self, n_in=21, hidden=64, n_classes=3, dropout=0.3):
        super().__init__()
        self.c1 = nn.Conv1d(n_in, hidden, kernel_size=7, padding=3)
        self.c2 = nn.Conv1d(hidden, hidden, kernel_size=7, padding=3)
        self.c3 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.out = nn.Conv1d(hidden, n_classes, kernel_size=1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # x: (B, L, 21)
        mask = (x.abs().sum(dim=2) > 0).unsqueeze(1).float()  # (B, 1, L); real residues = 1
        h = x.transpose(1, 2)                   # (B, 21, L) for Conv1d
        for conv in (self.c1, self.c2, self.c3):
            h = self.drop(self.act(conv(h))) * mask      # keep padded positions at zero
        return self.out(h)                      # (B, 3, L)


@torch.no_grad()
def evaluate_cnn(model, loader, loss_fn, device="cpu"):
    """Return (mean loss, predictions, targets) over real (non-padded) residues."""
    model.eval()
    total, n = 0.0, 0
    preds, trues = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)                                # (B, 3, L)
        valid = yb != PAD_LABEL
        total += loss_fn(logits, yb).item() * valid.sum().item()
        n += valid.sum().item()
        preds.append(logits.argmax(dim=1)[valid].cpu())
        trues.append(yb[valid].cpu())
    return total / n, torch.cat(preds), torch.cat(trues)


def train_cnn(model, train_loader, val_loader, epochs=30, lr=1e-3,
              class_weight=None, weight_decay=1e-4, patience=5, min_delta=5e-4, 
              device="cpu"):
    """Train with Adam. Keeps the weights from the epoch with the best validation loss
    and stops early if validation loss doesn't improve for `patience` epochs.
    Returns the same history dict as softmax_ss.train, so plot_history works."""
    model.to(device)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight, ignore_index=PAD_LABEL)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_q3": [], "batch_loss": []}
    best_val, best_state, bad_epochs = float("inf"), None, 0

    stopper = EarlyStopping(patience, min_delta)
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

            n_valid = (yb != PAD_LABEL).sum().item()
            history["batch_loss"].append(loss.item())
            running += loss.item() * n_valid
            n += n_valid

        train_loss = running / n
        val_loss, pred, true = evaluate_cnn(model, val_loader, loss_fn, device)
        q3 = (pred == true).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_q3"].append(q3)
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val Q3 {q3:.4f}")

        if stopper.step(val_loss, model):
            print(f"early stopping at epoch {epoch}")
            break

    stopper.restore(model)
    return history

