import torch
import torch.nn as nn
import matplotlib.pyplot as plt

CLASSES = ["H", "E", "C"]


class SoftmaxRegression(nn.Module):
    """One linear layer -> 3 raw scores (logits). CrossEntropyLoss applies the softmax."""

    def __init__(self, n_features=315, n_classes=3):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.linear(x)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device="cpu"):
    """Return (mean loss, predictions, targets) over a loader."""
    model.eval()
    total_loss, n = 0.0, 0
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * len(yb)
        n += len(yb)
        all_pred.append(logits.argmax(dim=1).cpu())
        all_true.append(yb.cpu())
    return total_loss / n, torch.cat(all_pred), torch.cat(all_true)


def confusion_matrix(pred, true, n_classes=3):
    """cm[i, j] = number of residues with true class i predicted as class j."""
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    idx = true * n_classes + pred
    cm += torch.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    return cm


def report(pred, true, classes=CLASSES):
    """Print Q3 plus per-class precision / recall / F1 and the confusion matrix."""
    cm = confusion_matrix(pred, true, len(classes)).float()
    tp = cm.diag()
    precision = tp / cm.sum(dim=0).clamp(min=1)   # of predicted c, how many were right
    recall = tp / cm.sum(dim=1).clamp(min=1)      # of true c, how many were found
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    q3 = tp.sum() / cm.sum()

    print(f"Q3 accuracy: {q3:.4f}")
    print(f"{'class':<6}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for i, c in enumerate(classes):
        print(f"{c:<6}{precision[i]:>10.3f}{recall[i]:>10.3f}{f1[i]:>10.3f}{int(cm[i].sum()):>10d}")
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print("      " + "".join(f"{c:>8}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:<6}" + "".join(f"{int(v):>8d}" for v in cm[i]))
    return q3.item(), precision, recall


def train(model, train_loader, val_loader, epochs=20, lr=1e-2,
          class_weight=None, weight_decay=0.0, device="cpu"):
    """Train with Adam. Returns a history dict with per-epoch and per-batch losses."""
    model.to(device)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_q3": [], "batch_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

            history["batch_loss"].append(loss.item())
            running += loss.item() * len(yb)
            n += len(yb)

        train_loss = running / n
        val_loss, pred, true = evaluate(model, val_loader, loss_fn, device)
        q3 = (pred == true).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_q3"].append(q3)
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val Q3 {q3:.4f}")

    return history


def plot_history(history, smooth=50):
    """Left: per-epoch train/val loss. Right: per-batch loss (smoothed) and val Q3."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="validation")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss per epoch")
    ax[0].legend()

    b = torch.tensor(history["batch_loss"])
    ax[1].plot(b.numpy(), alpha=0.25, label="per batch")
    if len(b) > smooth:
        kernel = torch.ones(1, 1, smooth) / smooth
        ma = torch.nn.functional.conv1d(b.view(1, 1, -1), kernel).flatten()
        ax[1].plot(range(smooth - 1, len(b)), ma.numpy(), label=f"moving avg ({smooth})")
    ax[1].set(xlabel="batch", ylabel="loss", title="Training loss per batch")
    ax[1].legend()

    ax[2].plot(epochs, history["val_q3"])
    ax[2].set(xlabel="epoch", ylabel="Q3", title="Validation accuracy")

    plt.tight_layout()
    plt.show()

