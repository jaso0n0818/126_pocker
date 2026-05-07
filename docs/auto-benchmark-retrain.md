# Auto Benchmark Refresh And Retraining

This repo can refresh the public Poker44 training benchmark, record it in a
local download register, build a flattened training dataset, and retrain the
local Torch miner model.

## Manual Refresh

Download all released benchmark dates and write the register:

```bash
venv/bin/python scripts/download_poker44_benchmark.py
```

Outputs:

```text
download/poker44_benchmark/register.json
download/poker44_benchmark/poker44_benchmark_all.json.gz
```

Train manually:

```bash
venv/bin/python train.py \
  --dataset download/poker44_benchmark/poker44_benchmark_all.json.gz \
  --epochs 20 \
  --batch-size 64 \
  --model-out saved_model/miner_model.pt
```

## Auto Refresh After Requests

The automatic path is opt-in. It always responds to the validator first, then
runs benchmark refresh and retraining in a background thread.

For the local socket server in `use.py`:

```bash
POKER44_AUTO_RETRAIN=1 \
POKER44_AUTO_RETRAIN_EPOCHS=5 \
POKER44_AUTO_RETRAIN_BATCH_SIZE=64 \
venv/bin/python use.py
```

For the Bittensor miner in `neurons/miner.py`:

```bash
POKER44_AUTO_RETRAIN=1 \
POKER44_USE_TRAINED_MODEL=1 \
POKER44_AUTO_RETRAIN_EPOCHS=5 \
POKER44_AUTO_RETRAIN_BATCH_SIZE=64 \
python neurons/miner.py
```

## Environment Variables

- `POKER44_AUTO_RETRAIN=1`: enable background benchmark refresh and retraining.
- `POKER44_USE_TRAINED_MODEL=1`: make `neurons/miner.py` use `saved_model/miner_model.pt` instead of the heuristic scorer when possible.
- `POKER44_MODEL_PATH`: override the model path. Default: `saved_model/miner_model.pt`.
- `POKER44_BENCHMARK_DATASET`: override the flattened dataset path. Default: `download/poker44_benchmark/poker44_benchmark_all.json.gz`.
- `POKER44_BENCHMARK_REGISTER`: override the register path for `use.py`.
- `POKER44_AUTO_RETRAIN_EPOCHS`: retraining epochs. Default: `5` for auto mode.
- `POKER44_AUTO_RETRAIN_BATCH_SIZE`: retraining batch size. Default: `64`.
- `POKER44_AUTO_RETRAIN_LR`: retraining learning rate. Default: `0.001`.
- `POKER44_AUTO_RETRAIN_MIN_INTERVAL_SECONDS`: only used by `use.py`; prevents retraining too frequently.

## Operational Notes

Automatic retraining needs outbound HTTPS access to:

```text
https://api.poker44.net/api/v1/benchmark
```

If the API or network is unavailable, the miner keeps serving scores with the
currently loaded model or the heuristic fallback.
