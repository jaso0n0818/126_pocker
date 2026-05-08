import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import gzip
import json
import argparse
from collections import Counter

from poker44.utils.hand_features import (
    CHUNK_FEATURE_NAMES,
    extract_chunk_features,
    normalize_label,
)
from poker44.utils.miner_model import (
    MODEL_ARCHITECTURE_VERSION,
    MinerNet,
    apply_score_shift,
    logit,
)


class Poker44Dataset(Dataset):
    def __init__(self, chunks, labels):
        self.features = [extract_chunk_features(chunk) for chunk in chunks]
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_dataset(filepath, return_metadata=False):
    chunks = []
    labels = []
    metadata = {}

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "chunks" in data and "groundTruth" in data:
        chunks = list(data["chunks"])
        labels = [int(label) for label in data["groundTruth"]]
        metadata["sourceDates"] = list(data.get("sourceDates") or [])
    else:
        if isinstance(data, list):
            hands = data
        elif isinstance(data, dict) and "hands" in data:
            hands = data["hands"]
        else:
            hands = data.get("data", [])

        for hand in hands:
            chunks.append(hand)
            labels.append(normalize_label(hand.get("label", "human")))

    if len(chunks) != len(labels):
        raise ValueError(
            f"Dataset shape mismatch: {len(chunks)} chunks but {len(labels)} labels"
        )

    label_counts = Counter(labels)
    print(f"Loaded {len(chunks)} examples from {filepath}")
    print(
        f"Label distribution: human={label_counts.get(0, 0)} "
        f"bot={label_counts.get(1, 0)}"
    )

    if return_metadata:
        return chunks, labels, metadata
    return chunks, labels


def _split_indices(
    labels,
    validation_split=0.2,
    seed=44,
    source_dates=None,
    validation_source_date=None,
):
    if validation_source_date and source_dates:
        val_indices = [
            i for i, d in enumerate(source_dates)
            if str(d) == str(validation_source_date)
        ]
        val_set = set(val_indices)
        train_indices = [i for i in range(len(labels)) if i not in val_set]
        if val_indices and train_indices:
            return train_indices, val_indices

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    labels_array = np.asarray(labels)

    for label in sorted(set(labels)):
        indices = np.where(labels_array == label)[0]
        rng.shuffle(indices)

        val_count = int(round(len(indices) * validation_split))
        if validation_split > 0 and len(indices) > 1:
            val_count = max(1, min(val_count, len(indices) - 1))

        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _subset(items, indices):
    return [items[i] for i in indices]


def _average_precision(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    order = np.argsort(-y_score)
    sorted_true = y_true[order]

    positives = int(np.sum(sorted_true == 1))
    if positives == 0:
        return 0.0

    tp = 0
    precision_sum = 0.0

    for rank, label in enumerate(sorted_true, start=1):
        if int(label) == 1:
            tp += 1
            precision_sum += tp / rank

    return float(precision_sum / positives)


def _reward(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int64)

    preds = np.round(y_pred).astype(int)

    tn = int(np.sum((y_true == 0) & (preds == 0)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))
    tp = int(np.sum((y_true == 1) & (preds == 1)))

    negative_count = max(tn + fp, 1)
    positive_count = max(tp + fn, 1)

    fpr = fp / negative_count
    bot_recall = tp / positive_count
    ap_score = _average_precision(y_true, y_pred)

    human_safety_penalty = max(0.0, 1.0 - fpr) ** 2
    if fpr >= 0.10:
        human_safety_penalty = 0.0

    base_score = 0.65 * ap_score + 0.35 * bot_recall
    reward_value = base_score * human_safety_penalty

    return reward_value, {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fpr,
        "bot_recall": bot_recall,
        "ap_score": ap_score,
        "human_safety_penalty": human_safety_penalty,
        "base_score": base_score,
        "reward": reward_value,
    }


def _shift_scores_for_threshold(scores, threshold):
    scores = np.asarray(scores, dtype=np.float32)
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)

    logits = np.log(clipped / np.clip(1 - clipped, 1e-6, 1))
    shifted_logits = logits - logit(threshold)

    return 1.0 / (1.0 + np.exp(-shifted_logits))


def _calibrate_human_safe_shift(scores, labels, max_human_fpr=0.02):
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    human_scores = scores[labels == 0]

    candidates = set(float(s) for s in scores)
    candidates.update([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99])

    if human_scores.size:
        for q in [0.90, 0.95, 0.97, 0.98, 0.99, 1.0]:
            candidates.add(float(np.quantile(human_scores, q)))

    best_threshold = 0.90
    best_reward = -1.0
    best_metrics = None

    for threshold in sorted(candidates, reverse=True):
        threshold = min(max(float(threshold), 1e-6), 1 - 1e-6)
        shifted_scores = _shift_scores_for_threshold(scores, threshold)

        reward_value, metrics = _reward(shifted_scores, labels)

        if metrics["fpr"] > max_human_fpr:
            continue

        better = reward_value > best_reward + 1e-9
        safer_tie = (
            abs(reward_value - best_reward) <= 1e-9
            and best_metrics is not None
            and metrics["fpr"] < best_metrics["fpr"]
        )

        if better or safer_tie:
            best_threshold = threshold
            best_reward = reward_value
            best_metrics = metrics

    if best_metrics is None:
        best_threshold = 0.99
        shifted_scores = _shift_scores_for_threshold(scores, best_threshold)
        best_reward, best_metrics = _reward(shifted_scores, labels)

    score_shift = -logit(best_threshold)
    return score_shift, best_threshold, best_reward, best_metrics


def _class_loss_weights(labels, human_loss_weight=1.50):
    counts = Counter(int(label) for label in labels)
    total = max(1, sum(counts.values()))
    class_count = max(1, len(counts))

    human_weight = total / (class_count * max(1, counts.get(0, 0)))
    bot_weight = total / (class_count * max(1, counts.get(1, 0)))

    human_weight *= float(human_loss_weight)

    return {
        0: float(human_weight),
        1: float(bot_weight),
    }


def predict_scores(model, device, chunks, batch_size=128, score_shift=None):
    model.eval()

    dataset = Poker44Dataset(chunks, labels=[0] * len(chunks))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    scores = []

    with torch.no_grad():
        for batch_features, _ in loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)

            shift = getattr(model, "score_shift", 0.0) if score_shift is None else score_shift
            outputs = apply_score_shift(outputs, shift)

            scores.extend(outputs.cpu().numpy().tolist())

    return scores


def train_model(
    chunks,
    labels,
    epochs=20,
    batch_size=32,
    lr=1e-3,
    allow_single_class=False,
    validation_split=0.2,
    source_dates=None,
    validation_source_date=None,
    max_human_fpr=0.02,
    human_loss_weight=1.50,
    human_margin_weight=0.50,
    bot_margin_weight=0.05,
    seed=44,
):
    label_counts = Counter(labels)

    if len(label_counts) < 2 and not allow_single_class:
        raise ValueError(
            "Training data contains only one class. "
            "Add both human and bot chunks, or use --allow-single-class."
        )

    train_indices, val_indices = _split_indices(
        labels,
        validation_split=validation_split,
        seed=seed,
        source_dates=source_dates,
        validation_source_date=validation_source_date,
    )

    train_chunks = _subset(chunks, train_indices)
    train_labels = _subset(labels, train_indices)
    val_chunks = _subset(chunks, val_indices)
    val_labels = _subset(labels, val_indices)

    dataset = Poker44Dataset(train_chunks, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinerNet().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss(reduction="none")

    loss_weights = _class_loss_weights(
        train_labels,
        human_loss_weight=human_loss_weight,
    )

    class_weight_tensor = torch.tensor(
        [loss_weights[0], loss_weights[1]],
        dtype=torch.float32,
        device=device,
    )

    best_state = None
    best_reward = -1.0
    best_score_shift = 0.0
    best_threshold = 0.5
    best_metrics = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(batch_features)

            batch_weights = torch.where(
                batch_labels >= 0.5,
                class_weight_tensor[1],
                class_weight_tensor[0],
            )

            bce_loss = (criterion(outputs, batch_labels) * batch_weights).mean()

            human_scores = outputs[batch_labels < 0.5]
            bot_scores = outputs[batch_labels >= 0.5]

            human_margin_loss = torch.tensor(0.0, device=device)
            bot_margin_loss = torch.tensor(0.0, device=device)

            if human_scores.numel():
                human_margin_loss = torch.relu(human_scores - 0.45).pow(2).mean()

            if bot_scores.numel():
                bot_margin_loss = torch.relu(0.55 - bot_scores).pow(2).mean()

            loss = (
                bce_loss
                + float(human_margin_weight) * human_margin_loss
                + float(bot_margin_weight) * bot_margin_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * batch_features.size(0)

        message = f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(dataset):.4f}"

        if val_chunks:
            raw_val_scores = np.asarray(
                predict_scores(model, device, val_chunks, batch_size=128, score_shift=0.0),
                dtype=np.float32,
            )

            score_shift, threshold, val_reward, val_metrics = _calibrate_human_safe_shift(
                raw_val_scores,
                val_labels,
                max_human_fpr=max_human_fpr,
            )

            if val_reward > best_reward:
                best_reward = val_reward
                best_score_shift = score_shift
                best_threshold = threshold
                best_metrics = val_metrics
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

            message += (
                f" - ValReward: {val_reward:.4f}"
                f" AP: {val_metrics['ap_score']:.4f}"
                f" Recall: {val_metrics['bot_recall']:.4f}"
                f" FPR: {val_metrics['fpr']:.4f}"
                f" Threshold: {threshold:.4f}"
                f" Shift: {score_shift:.4f}"
            )

        print(message)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.score_shift = float(best_score_shift)
        model.decision_threshold = float(best_threshold)

        print(
            f"Loaded best checkpoint: "
            f"reward={best_reward:.4f}, "
            f"threshold={best_threshold:.6f}, "
            f"score_shift={best_score_shift:.6f}, "
            f"FPR={best_metrics['fpr']:.4f}, "
            f"AP={best_metrics['ap_score']:.4f}, "
            f"Recall={best_metrics['bot_recall']:.4f}"
        )
    else:
        model.score_shift = 0.0
        model.decision_threshold = 0.5

    return model, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Poker44 miner model.")

    parser.add_argument(
        "--dataset",
        default="hands_generator/human_hands/poker_hands_combined.json.gz",
        help="Path to gzipped JSON dataset.",
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)

    parser.add_argument(
        "--validation-source-date",
        help="Hold out one benchmark sourceDate for validation.",
    )

    parser.add_argument("--seed", type=int, default=44)

    parser.add_argument(
        "--max-human-fpr",
        type=float,
        default=0.02,
        help="Maximum validation human false-positive rate.",
    )

    parser.add_argument(
        "--human-loss-weight",
        type=float,
        default=1.50,
        help="Extra BCE weight for human examples.",
    )

    parser.add_argument(
        "--human-margin-weight",
        type=float,
        default=0.50,
        help="Penalty for human scores above 0.45.",
    )

    parser.add_argument(
        "--bot-margin-weight",
        type=float,
        default=0.05,
        help="Small penalty for bot scores below 0.55.",
    )

    parser.add_argument(
        "--model-out",
        default="saved_model/miner_model.pt",
        help="Output model path.",
    )

    parser.add_argument(
        "--allow-single-class",
        action="store_true",
        help="Allow all-human or all-bot training data.",
    )

    args = parser.parse_args()

    try:
        train_chunks, train_labels, metadata = load_dataset(
            args.dataset,
            return_metadata=True,
        )
    except FileNotFoundError:
        raise SystemExit(
            f"Dataset file not found: {args.dataset}\n"
            "Please pass a real .json.gz dataset path."
        )

    if not train_chunks:
        raise SystemExit(f"Dataset is empty: {args.dataset}")

    model, device = train_model(
        train_chunks,
        train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        allow_single_class=args.allow_single_class,
        validation_split=args.validation_split,
        source_dates=metadata.get("sourceDates"),
        validation_source_date=args.validation_source_date,
        max_human_fpr=args.max_human_fpr,
        human_loss_weight=args.human_loss_weight,
        human_margin_weight=args.human_margin_weight,
        bot_margin_weight=args.bot_margin_weight,
        seed=args.seed,
    )

    model_out = args.model_out
    model_dir = os.path.dirname(model_out)

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "score_shift": float(getattr(model, "score_shift", 0.0)),
        "decision_threshold": float(getattr(model, "decision_threshold", 0.5)),
        "feature_names": CHUNK_FEATURE_NAMES,
        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "max_human_fpr": args.max_human_fpr,
        "human_loss_weight": args.human_loss_weight,
        "human_margin_weight": args.human_margin_weight,
        "bot_margin_weight": args.bot_margin_weight,
    }

    torch.save(checkpoint, model_out)
    print(f"Saved model to {model_out}")