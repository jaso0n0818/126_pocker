#!/usr/bin/env python3
import torch
import argparse
import numpy as np
from train import load_dataset, predict_scores, _reward
from poker44.utils.miner_model import (
    apply_score_shift,
    build_model_for_state_dict,
    legacy_chunk_features,
    model_input_dim,
    normalize_state_dict,
)


def load_checkpoint(model_path, device):
    cp = torch.load(model_path, map_location=device)
    if isinstance(cp, dict) and "model_state_dict" in cp:
        state = cp["model_state_dict"]
        score_shift = float(cp.get("score_shift", 0.0))
    else:
        state = cp
        score_shift = 0.0

    state = normalize_state_dict(state)
    model = build_model_for_state_dict(state).to(device)
    model.load_state_dict(state)
    model.score_shift = score_shift
    return model, score_shift


def score_model(model, device, chunks, batch_size, score_shift):
    legacy_input_dim = model_input_dim(model)

    if legacy_input_dim is None:
        return predict_scores(
            model,
            device,
            chunks,
            batch_size=batch_size,
            score_shift=score_shift,
        )

    features = torch.tensor(
        [legacy_chunk_features(chunk, legacy_input_dim) for chunk in chunks],
        dtype=torch.float32,
    )
    scores = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size].to(device)
            outputs = model(batch)
            outputs = apply_score_shift(outputs, score_shift)
            scores.extend(outputs.cpu().numpy().tolist())
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="download/poker44_benchmark/poker44_benchmark_all.json.gz",
    )
    parser.add_argument("--model", default="saved_model/miner_model.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print("Loading dataset...", flush=True)
    chunks, labels = load_dataset(args.dataset)
    print(f"Loaded {len(chunks)} examples", flush=True)
    print("Loading model...", flush=True)
    model, score_shift = load_checkpoint(args.model, device)
    model.eval()
    print("Running evaluation...", flush=True)
    scores = score_model(
        model,
        device,
        chunks,
        batch_size=args.batch_size,
        score_shift=score_shift,
    )
    reward_value, metrics = _reward(np.asarray(scores, dtype=np.float32), labels)
    print("EVAL_RESULT")
    print(f"reward={reward_value:.6f}")
    for k, v in metrics.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
