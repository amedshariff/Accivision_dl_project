"""
Evaluate Model Module
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from models.cnn_lstm_model import CNNLSTMModel
from training.dataset_loader import get_data_loader

def evaluate_model(model_path: str, test_csv: str = "data/splits/test.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTMModel(pretrained_cnn=True, freeze_cnn=True).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = get_data_loader(test_csv, batch_size=2, shuffle=False)
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for seqs, labels, _ in loader:
            seqs = seqs.to(device)
            logits = model(seqs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            y_probs.extend(probs)
            y_pred.extend(preds)
            y_true.extend(labels.numpy().flatten())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}
