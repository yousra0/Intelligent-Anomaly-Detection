from pathlib import Path
import hashlib

import joblib
import torch

from src.models.autoencoder import FraudAutoEncoder
from src.utils.anomaly_utils import validate_saved_score_file


def test_saved_autoencoder_scores_are_valid_artifacts():
	repo_root = Path(__file__).resolve().parents[1]
	models_dir = repo_root / "outputs" / "models"

	for file_name in ("ae_scores_test.npy", "ae_scores_val.npy"):
		summary = validate_saved_score_file(models_dir / file_name, name=file_name)
		assert summary["shape"][0] > 0
		assert summary["min"] >= 0.0
		assert summary["max"] >= summary["min"]


def test_autoencoder_save_records_checksum_and_rng_state(tmp_path):
	ae = FraudAutoEncoder(epochs=1, patience=1)
	ae.build(n_features=3)
	ae.history = {"loss": [0.1], "val_loss": [0.2], "mae": [0.3]}
	ae.threshold = 0.42
	ae.train_time = 1.5
	ae.train_mse_stats = {"mean": 0.1, "std": 0.01, "p95": 0.2, "p99": 0.3, "max": 0.4}
	ae.training_seed = 1234
	ae.torch_rng_state = torch.arange(16, dtype=torch.uint8)
	ae.cuda_rng_state_all = []

	ae.save(tmp_path)

	meta = joblib.load(tmp_path / "autoencoder_meta.pkl")
	weights_path = tmp_path / "autoencoder_weights.pt"

	assert meta["architecture"]["total_params"] == ae.count_params()
	assert meta["weights_sha256"] == hashlib.sha256(weights_path.read_bytes()).hexdigest()
	assert meta["training_seed"] == 1234
	assert torch.equal(meta["torch_rng_state"], torch.arange(16, dtype=torch.uint8))
	assert meta["cuda_rng_state_all"] == []
