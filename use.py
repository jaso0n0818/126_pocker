import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from poker44.utils.hand_features import extract_chunk_features


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTER_PATH = REPO_ROOT / "download" / "poker44_benchmark" / "register.json"
DEFAULT_DATASET_PATH = REPO_ROOT / "download" / "poker44_benchmark" / "poker44_benchmark_all.json.gz"
DEFAULT_MODEL_PATH = REPO_ROOT / "saved_model" / "miner_model.pt"

# -----------------------------
# 1️⃣ Dataset 정의
# -----------------------------
class Poker44Dataset(Dataset):
    def __init__(self, chunks):
        self.features = []
        for chunk in chunks:
            self.features.append(extract_chunk_features(chunk))
        self.features = torch.tensor(self.features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

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
        x = torch.sigmoid(self.fc2(x))  # 0~1 점수
        return x.squeeze(1)

# -----------------------------
# 3️⃣ Miner 클래스 정의
# -----------------------------
class Poker44Miner:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MinerNet().to(self.device)
        self.model_path = Path(model_path) if model_path else None
        self._model_lock = threading.Lock()
        self.reload_model()
        self.model.eval()
        self.reward_window = 20  # validator 기준

    def reload_model(self):
        if not self.model_path or not self.model_path.exists():
            return False
        with self._model_lock:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
        print(f"[INFO] 모델 로드 완료: {self.model_path}")
        return True

    def forward(self, chunks):
        dataset = Poker44Dataset(chunks)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        scores = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                with self._model_lock:
                    preds = self.model(batch)
                scores.extend(preds.cpu().numpy().tolist())
        return scores


class BenchmarkAutoTrainer:
    def __init__(self, miner, enabled=False):
        self.miner = miner
        self.enabled = enabled
        self.lock = threading.Lock()
        self.running = False
        self.last_started_at = 0.0
        self.min_interval_seconds = int(os.getenv("POKER44_AUTO_RETRAIN_MIN_INTERVAL_SECONDS", "0"))
        self.epochs = int(os.getenv("POKER44_AUTO_RETRAIN_EPOCHS", "5"))
        self.batch_size = int(os.getenv("POKER44_AUTO_RETRAIN_BATCH_SIZE", "64"))
        self.lr = float(os.getenv("POKER44_AUTO_RETRAIN_LR", "0.001"))
        self.register_path = Path(os.getenv("POKER44_BENCHMARK_REGISTER", str(DEFAULT_REGISTER_PATH)))
        self.dataset_path = Path(os.getenv("POKER44_BENCHMARK_DATASET", str(DEFAULT_DATASET_PATH)))
        self.model_path = Path(os.getenv("POKER44_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        self.state_path = self.register_path.parent / "auto_retrain_state.json"

    def trigger(self):
        if not self.enabled:
            return
        now = time.time()
        with self.lock:
            if self.running:
                print("[AUTO] benchmark 갱신/재학습이 이미 실행 중입니다.")
                return
            if self.min_interval_seconds and now - self.last_started_at < self.min_interval_seconds:
                print("[AUTO] 재학습 최소 간격 안이라 이번 요청에서는 건너뜁니다.")
                return
            self.running = True
            self.last_started_at = now
        threading.Thread(target=self._run, daemon=True).start()

    def _load_state(self):
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_state(self, state):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _run_command(self, cmd):
        print(f"[AUTO] 실행: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    def _run(self):
        try:
            downloader = REPO_ROOT / "scripts" / "download_poker44_benchmark.py"
            trainer = REPO_ROOT / "train.py"
            self._run_command([sys.executable, str(downloader)])

            register = json.loads(self.register_path.read_text(encoding="utf-8"))
            combined = register.get("combined") or {}
            dataset_sha = combined.get("sha256")
            if not dataset_sha:
                print("[AUTO] register에 combined sha256이 없어 재학습을 건너뜁니다.")
                return

            state = self._load_state()
            model_exists = self.model_path.exists()
            if state.get("last_trained_dataset_sha256") == dataset_sha and model_exists:
                print("[AUTO] 새 benchmark release가 없어 재학습을 건너뜁니다.")
                return

            self._run_command(
                [
                    sys.executable,
                    str(trainer),
                    "--dataset",
                    str(self.dataset_path),
                    "--epochs",
                    str(self.epochs),
                    "--batch-size",
                    str(self.batch_size),
                    "--lr",
                    str(self.lr),
                    "--model-out",
                    str(self.model_path),
                ]
            )
            state.update(
                {
                    "last_trained_dataset_sha256": dataset_sha,
                    "last_trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "dataset_path": str(self.dataset_path),
                    "model_path": str(self.model_path),
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                }
            )
            self._save_state(state)
            self.miner.reload_model()
            print("[AUTO] benchmark 갱신, register 기록, 재학습, 모델 reload 완료")
        except Exception as exc:
            print(f"[AUTO][ERROR] 자동 benchmark 재학습 실패: {exc}")
        finally:
            with self.lock:
                self.running = False

# -----------------------------
# 4️⃣ 실시간 서버 설정 (Validator 요청 처리)
# -----------------------------
class MinerServer:
    def __init__(self, miner, host="0.0.0.0", port=9999, auto_trainer=None):
        self.miner = miner
        self.auto_trainer = auto_trainer
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        print(f"[INFO] Miner 서버 시작: {self.host}:{self.port}")

    def handle_client(self, conn, addr):
        try:
            data = conn.recv(65536)  # 청크 데이터 수신
            if not data:
                return
            chunks = json.loads(data.decode())
            scores = self.miner.forward(chunks)
            response = json.dumps(scores).encode()
            conn.sendall(response)
            if self.auto_trainer:
                self.auto_trainer.trigger()
        except Exception as e:
            print(f"[ERROR] 클라이언트 처리 중 오류: {e}")
        finally:
            conn.close()

    def start(self):
        print("[INFO] Miner 서버 대기중...")
        while True:
            conn, addr = self.sock.accept()
            threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()

# -----------------------------
# 5️⃣ 실행 예시
# -----------------------------
if __name__ == "__main__":
    MODEL_PATH = os.getenv("POKER44_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    miner = Poker44Miner(model_path=MODEL_PATH)
    auto_trainer = BenchmarkAutoTrainer(
        miner,
        enabled=os.getenv("POKER44_AUTO_RETRAIN", "0").strip().lower() in {"1", "true", "yes", "on"},
    )

    server = MinerServer(miner, host="0.0.0.0", port=9999, auto_trainer=auto_trainer)
    server.start()
