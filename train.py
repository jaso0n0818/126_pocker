import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import gzip
import json
import argparse
from collections import Counter

from poker44.utils.hand_features import extract_chunk_features, normalize_label

# -----------------------------
# 1️⃣ Dataset 정의
# -----------------------------
class Poker44Dataset(Dataset):
    def __init__(self, chunks, labels):
        self.features = []
        for chunk in chunks:
            self.features.append(extract_chunk_features(chunk))
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# -----------------------------
# 2️⃣ 모델 정의
# -----------------------------
class MinerNet(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=32):
        super(MinerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)

# -----------------------------
# 3️⃣ 학습 함수
# -----------------------------
def load_dataset(filepath):
    """Load poker hands dataset from gzipped JSON file."""
    chunks = []
    labels = []
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    # Benchmark payloads use chunks + groundTruth. Older local files may be
    # plain hand records with an embedded label.
    if isinstance(data, dict) and "chunks" in data and "groundTruth" in data:
        chunks = list(data["chunks"])
        labels = [int(label) for label in data["groundTruth"]]
    else:
        if isinstance(data, list):
            hands = data
        elif isinstance(data, dict) and 'hands' in data:
            hands = data['hands']
        else:
            hands = data.get('data', [])

        for hand in hands:
            chunks.append(hand)
            label = normalize_label(hand.get("label", "human"))
            labels.append(label)

    if len(chunks) != len(labels):
        raise ValueError(
            f"Dataset shape mismatch: {len(chunks)} chunks but {len(labels)} labels"
        )

    label_counts = Counter(labels)
    print(f"Loaded {len(chunks)} examples from {filepath}")
    print(f"Label distribution: human={label_counts.get(0, 0)} bot={label_counts.get(1, 0)}")
    return chunks, labels

# -----------------------------
# 4️⃣ 학習 함数
# -----------------------------
def train_model(chunks, labels, epochs=10, batch_size=32, lr=1e-3, allow_single_class=False):
    label_counts = Counter(labels)
    if len(label_counts) < 2 and not allow_single_class:
        raise ValueError(
            "Training data contains only one class. Add both human and bot chunks, "
            "or rerun with --allow-single-class if you intentionally want a constant model."
        )

    dataset = Poker44Dataset(chunks, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinerNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # binary classification

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_features.size(0)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataset):.4f}")

    return model, device

# -----------------------------
# 5️⃣ 예측 함수
# -----------------------------
def predict_scores(model, device, chunks):
    model.eval()
    dataset = Poker44Dataset(chunks, labels=[0]*len(chunks))  # dummy labels
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    scores = []
    with torch.no_grad():
        for batch_features, _ in loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            scores.extend(outputs.cpu().numpy().tolist())
    return scores

# -----------------------------
# 6️⃣ 사용 예시
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Poker44 miner model.")
    parser.add_argument(
        "--dataset",
        default="hands_generator/human_hands/poker_hands_combined.json.gz",
        help="Path to a gzipped JSON dataset. Benchmark format should contain chunks + groundTruth.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--model-out",
        default="saved_model/miner_model.pt",
        help="Where to save the trained model state_dict.",
    )
    parser.add_argument(
        "--allow-single-class",
        action="store_true",
        help="Allow training on all-human or all-bot data. This usually creates a constant scorer.",
    )
    args = parser.parse_args()

    # Load dataset from gzipped JSON file
    dataset_path = args.dataset
    try:
        train_chunks, train_labels = load_dataset(dataset_path)
    except FileNotFoundError:
        raise SystemExit(
            f"Dataset file not found: {dataset_path}\n"
            "Pass a real benchmark .json.gz path, not the placeholder path/to/benchmark.json.gz."
        )

    if not train_chunks:
        raise SystemExit(f"Dataset is empty: {dataset_path}")

    # 학습
    model, device = train_model(
        train_chunks,
        train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        allow_single_class=args.allow_single_class,
    )

    # 예측
    test_chunks = [
        {"num_actions": 6, "total_bets": 120, "raise_count": 0},
        {"num_actions": 18, "total_bets": 900, "raise_count": 7},
    ]
    scores = predict_scores(model, device, test_chunks)
    for i, score in enumerate(scores):
        print(f"청크 {i} 점수: {score:.2f}")

    # 모델 저장
    model_out = args.model_out
    model_dir = os.path.dirname(model_out)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"Saved model to {model_out}")
