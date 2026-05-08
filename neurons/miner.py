"""Reference Poker44 miner with simple chunk-level behavioral heuristics."""

# from __future__ import annotations

import time
import os
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.utils.hand_features import FEATURE_NAMES, extract_chunk_features, extract_hand_features
from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    """
    Reference heuristic miner.

    It aggregates simple behavior signals over each chunk and returns a bot-risk
    score per chunk. The goal is not SOTA accuracy, but a deterministic and
    explainable baseline that is meaningfully better than random.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Heuristic Poker44 Miner started")
        repo_root = Path(__file__).resolve().parents[1]
        self.repo_root = repo_root
        self.auto_retrain_enabled = _env_bool("POKER44_AUTO_RETRAIN", False)
        self.use_trained_model = _env_bool("POKER44_USE_TRAINED_MODEL", False)
        self.model_path = Path(
            os.getenv("POKER44_MODEL_PATH", str(repo_root / "saved_model" / "miner_model.pt"))
        )
        self.register_path = Path(
            os.getenv(
                "POKER44_BENCHMARK_REGISTER",
                str(repo_root / "download" / "poker44_benchmark" / "register.json"),
            )
        )
        self.auto_retrain_state_path = self.register_path.parent / "auto_retrain_state.json"
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[Path(__file__).resolve()],
            defaults=self._manifest_defaults(),
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        self._auto_retrain_lock = threading.Lock()
        self._auto_retrain_running = False
        self._model_lock = threading.Lock()
        self._torch_model = None
        self._torch_device = None
        self._score_shift = 0.0
        if self.use_trained_model:
            self._load_torch_model()
        
        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    def _manifest_defaults(self) -> dict:
        common = {
            "license": "MIT",
            "repo_url": "https://github.com/Poker44/Poker44-subnet",
            "open_source": True,
            "inference_mode": "remote",
            "private_data_attestation": (
                "This miner trains only on public Poker44 benchmark releases and does not train on "
                "validator-only private evaluation data."
            ),
        }
        if self.use_trained_model:
            return {
                **common,
                "model_name": "poker44-benchmark-minernet",
                "model_version": "1",
                "framework": "pytorch",
                "notes": "Chunk-level neural miner trained from public Poker44 benchmark releases.",
                "training_data_statement": (
                    "Trained on public released Poker44 benchmark chunks recorded in "
                    f"{self.register_path}."
                ),
                "training_data_sources": [
                    "https://api.poker44.net/api/v1/benchmark",
                    str(self.register_path),
                ],
            }
        return {
            **common,
            "model_name": "poker44-reference-heuristic",
            "model_version": "1",
            "framework": "python-heuristic",
            "notes": "Reference heuristic miner shipped with the Poker44 subnet.",
            "training_data_statement": (
                "Reference heuristic miner. No training step. Uses only runtime chunk features."
            ),
            "training_data_sources": ["none"],
        }

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one deterministic bot-risk score per chunk."""
        chunks = synapse.chunks or []
        scores = self._score_chunks(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Miner Predctions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks with heuristic risks.")
        self._trigger_auto_retrain()
        return synapse

    def _score_chunks(self, chunks: list[list[dict]]) -> list[float]:
        if self.use_trained_model and self._torch_model is not None:
            model_scores = self._score_chunks_with_torch_model(chunks)
            if model_scores is not None:
                return model_scores
        return [self.score_chunk(chunk) for chunk in chunks]

    def _load_torch_model(self) -> bool:
        try:
            import torch
            from poker44.utils.miner_model import MinerNet

            if not self.model_path.exists():
                bt.logging.warning(f"Trained model file not found: {self.model_path}")
                return False
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MinerNet().to(device)
            checkpoint = torch.load(self.model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                score_shift = float(checkpoint.get("score_shift", 0.0) or 0.0)
            else:
                state_dict = checkpoint
                score_shift = 0.0
            model.load_state_dict(state_dict)
            model.eval()
            with self._model_lock:
                self._torch_device = device
                self._torch_model = model
                self._score_shift = score_shift
            bt.logging.info(f"Loaded trained Poker44 model: {self.model_path}")
            return True
        except Exception as exc:
            bt.logging.warning(f"Could not load trained Poker44 model: {exc}")
            return False

    def _score_chunks_with_torch_model(self, chunks: list[list[dict]]) -> list[float] | None:
        try:
            import torch
            from poker44.utils.miner_model import apply_score_shift

            features = [extract_chunk_features(chunk) for chunk in chunks]
            if not features:
                return []
            with self._model_lock:
                if self._torch_model is None or self._torch_device is None:
                    return None
                batch = torch.tensor(features, dtype=torch.float32).to(self._torch_device)
                with torch.no_grad():
                    outputs = self._torch_model(batch)
                    outputs = apply_score_shift(outputs, self._score_shift)
                return [round(float(score), 6) for score in outputs.cpu().numpy().tolist()]
        except Exception as exc:
            bt.logging.warning(f"Trained model scoring failed; falling back to heuristic: {exc}")
            return None

    def _trigger_auto_retrain(self) -> None:
        if not self.auto_retrain_enabled:
            return
        with self._auto_retrain_lock:
            if self._auto_retrain_running:
                return
            self._auto_retrain_running = True
        threading.Thread(target=self._run_auto_retrain, daemon=True).start()

    def _run_auto_retrain(self) -> None:
        try:
            downloader = self.repo_root / "scripts" / "download_poker44_benchmark.py"
            trainer = self.repo_root / "train.py"
            dataset = Path(
                os.getenv(
                    "POKER44_BENCHMARK_DATASET",
                    str(self.repo_root / "download" / "poker44_benchmark" / "poker44_benchmark_all.json.gz"),
                )
            )
            epochs = os.getenv("POKER44_AUTO_RETRAIN_EPOCHS", "5")
            batch_size = os.getenv("POKER44_AUTO_RETRAIN_BATCH_SIZE", "64")
            lr = os.getenv("POKER44_AUTO_RETRAIN_LR", "0.001")
            max_human_fpr = os.getenv("POKER44_AUTO_RETRAIN_MAX_HUMAN_FPR", "0.03")
            human_loss_weight = os.getenv("POKER44_AUTO_RETRAIN_HUMAN_LOSS_WEIGHT", "1.35")
            human_margin_weight = os.getenv("POKER44_AUTO_RETRAIN_HUMAN_MARGIN_WEIGHT", "0.35")
            bot_margin_weight = os.getenv("POKER44_AUTO_RETRAIN_BOT_MARGIN_WEIGHT", "0.10")
            validation_source_date = os.getenv("POKER44_AUTO_RETRAIN_VALIDATION_SOURCE_DATE", "")
            subprocess.run([sys.executable, str(downloader)], cwd=self.repo_root, check=True)
            dataset_sha = self._current_benchmark_dataset_sha()
            state = self._load_auto_retrain_state()
            if (
                dataset_sha
                and state.get("last_trained_dataset_sha256") == dataset_sha
                and self.model_path.exists()
            ):
                bt.logging.info("Poker44 benchmark unchanged; skipping auto retrain.")
                return

            subprocess.run(
                [
                    sys.executable,
                    str(trainer),
                    "--dataset",
                    str(dataset),
                    "--epochs",
                    epochs,
                    "--batch-size",
                    batch_size,
                    "--lr",
                    lr,
                    "--max-human-fpr",
                    max_human_fpr,
                    "--human-loss-weight",
                    human_loss_weight,
                    "--human-margin-weight",
                    human_margin_weight,
                    "--bot-margin-weight",
                    bot_margin_weight,
                    "--model-out",
                    str(self.model_path),
                ]
                + (
                    ["--validation-source-date", validation_source_date]
                    if validation_source_date
                    else []
                ),
                cwd=self.repo_root,
                check=True,
            )
            state.update(
                {
                    "last_trained_dataset_sha256": dataset_sha,
                    "last_trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "dataset_path": str(dataset),
                    "model_path": str(self.model_path),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "max_human_fpr": max_human_fpr,
                    "human_loss_weight": human_loss_weight,
                    "human_margin_weight": human_margin_weight,
                    "bot_margin_weight": bot_margin_weight,
                    "validation_source_date": validation_source_date,
                }
            )
            self._save_auto_retrain_state(state)
            if self.use_trained_model:
                self._load_torch_model()
            bt.logging.info("Poker44 benchmark auto refresh/retrain completed.")
        except Exception as exc:
            bt.logging.warning(f"Poker44 benchmark auto refresh/retrain failed: {exc}")
        finally:
            with self._auto_retrain_lock:
                self._auto_retrain_running = False

    def _current_benchmark_dataset_sha(self) -> str:
        try:
            register = json.loads(self.register_path.read_text(encoding="utf-8"))
            return str((register.get("combined") or {}).get("sha256") or "")
        except Exception:
            return ""

    def _load_auto_retrain_state(self) -> dict:
        try:
            return json.loads(self.auto_retrain_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_auto_retrain_state(self, state: dict) -> None:
        self.auto_retrain_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.auto_retrain_state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        # Use richer poker behavioral features to reward sustained aggression,
        # normalized stake sizing, and showdown behavior while penalizing passive or
        # overly defensive play. This mirrors the idea of behavioral signatures
        # used to separate human and AI play patterns.
        features = dict(zip(FEATURE_NAMES, extract_hand_features(hand)))

        score = 0.0
        score += 0.14 * features.get("street_depth", 0.0)
        score += 0.16 * features.get("aggression_ratio", 0.0)
        score += 0.12 * features.get("raise_ratio", 0.0)
        score += 0.07 * features.get("max_normalized_amount_bb", 0.0)
        score += 0.06 * features.get("avg_normalized_amount_bb", 0.0)
        score += 0.08 * features.get("showdown", 0.0)
        score += 0.05 * features.get("pot_growth_bb", 0.0)
        score += 0.04 * features.get("hero_stack_bb", 0.0)
        score += 0.03 * features.get("hero_profit_bb", 0.0)
        score += 0.08 * features.get("aggressive_street_coverage", 0.0)
        score += 0.05 * features.get("hero_aggression_ratio", 0.0)
        score += 0.04 * features.get("large_bet_ratio", 0.0)
        score -= 0.16 * features.get("fold_ratio", 0.0)
        score -= 0.08 * features.get("call_ratio", 0.0)
        score -= 0.05 * features.get("check_ratio", 0.0)
        score -= 0.07 * features.get("passive_ratio", 0.0)

        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5

        hand_scores = [cls._score_hand(hand) for hand in chunk]
        avg_score = sum(hand_scores) / len(hand_scores)

        return round(cls._clamp01(avg_score), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Random miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
