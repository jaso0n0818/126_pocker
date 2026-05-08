#!/usr/bin/env python3
import torch
import argparse
from train import load_dataset, _evaluate_model
from poker44.utils.miner_model import MinerNet


def load_checkpoint(model_path, device):
    cp = torch.load(model_path, map_location=device)
    if isinstance(cp, dict) and "model_state_dict" in cp:
        state = cp["model_state_dict"]
        score_shift = float(cp.get("score_shift", 0.0))
    else:
        state = cp
        score_shift = 0.0
    model = MinerNet().to(device)
    try:
        model.load_state_dict(state)
    except Exception:
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        model.load_state_dict(new_state)
    model.score_shift = score_shift
    return model, score_shift


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
    reward_value, metrics = _evaluate_model(
        model, device, chunks, labels, batch_size=args.batch_size, score_shift=score_shift
    )
    print("EVAL_RESULT")
    print(f"reward={reward_value:.6f}")
    for k, v in metrics.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
