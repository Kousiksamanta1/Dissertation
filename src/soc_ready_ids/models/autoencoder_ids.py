"""PyTorch autoencoder anomaly detector with a shared inference interface."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from soc_ready_ids.config import load_config
from soc_ready_ids.data.loader import SUPPORTED_DATASETS, normalize_dataset_name
from soc_ready_ids.data.preprocessor import load_processed_arrays
from soc_ready_ids.models.common import evaluate_predictions, save_model_artifact
from soc_ready_ids.utils.io import ensure_dir
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)


class TabularAutoencoder(nn.Module):
    """Small fully connected autoencoder for scaled network-flow features."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        """Build encoder and decoder layers."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Reconstruct one input batch."""
        return self.decoder(self.encoder(batch))


class ReconstructionErrorModule(nn.Module):
    """Expose autoencoder reconstruction error for SHAP DeepExplainer."""

    def __init__(self, autoencoder: TabularAutoencoder) -> None:
        """Wrap a fitted autoencoder."""
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Return mean squared reconstruction error per row."""
        reconstructed = self.autoencoder(batch)
        return torch.mean((batch - reconstructed) ** 2, dim=1, keepdim=True)


class AutoencoderIDSModel:
    """Scikit-learn-like binary anomaly detector backed by PyTorch."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        threshold: float = 0.0,
    ) -> None:
        """Initialize the network and anomaly threshold."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.threshold = float(threshold)
        self.network = TabularAutoencoder(
            input_dim, hidden_dim, latent_dim
        ).cpu()

    def reconstruction_error(
        self, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """Compute row-level mean squared reconstruction error."""
        values = (
            X.to_numpy(dtype=np.float32)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X, dtype=np.float32)
        )
        self.network.eval()
        with torch.no_grad():
            tensor = torch.tensor(values, dtype=torch.float32)
            reconstructed = self.network(tensor).cpu().numpy()
        return np.mean(np.square(values - reconstructed), axis=1)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict 0 for benign and 1 for anomalous traffic."""
        return (self.reconstruction_error(X) > self.threshold).astype(int)

    def predict_proba(
        self, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """Return calibrated-like benign and anomaly confidence scores."""
        errors = self.reconstruction_error(X)
        scale = max(self.threshold, float(np.std(errors)), 1e-8)
        anomaly_probability = 1.0 / (
            1.0 + np.exp(-(errors - self.threshold) / scale)
        )
        return np.column_stack(
            [1.0 - anomaly_probability, anomaly_probability]
        )

    def deep_explainer_module(self) -> ReconstructionErrorModule:
        """Return the differentiable anomaly-score module."""
        self.network.eval()
        return ReconstructionErrorModule(self.network)


@dataclass
class AutoencoderResult:
    """Artifacts produced by autoencoder training."""

    model: AutoencoderIDSModel
    model_path: Path
    threshold: float
    metrics: dict[str, Any]


def _benign_class_index(class_names: list[str]) -> int:
    """Return the encoded class index used for benign traffic."""
    normalized = [name.upper() for name in class_names]
    return normalized.index("BENIGN") if "BENIGN" in normalized else 0


def train_autoencoder(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: list[str],
    feature_columns: list[str],
    config_values: dict[str, Any],
    model_dir: str,
    metrics_dir: str,
) -> AutoencoderResult:
    """Train, evaluate, and persist a reconstruction-error detector."""
    random_state = int(config_values["project"].get("random_state", 42))
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    ae_config = config_values["models"]["autoencoder"]
    input_dim = X_train.shape[1]
    hidden_dim = int(ae_config.get("hidden_dim", 64))
    latent_dim = min(int(ae_config.get("latent_dim", 16)), hidden_dim)

    benign_index = _benign_class_index(class_names)
    benign_train = X_train.iloc[y_train == benign_index]
    if benign_train.empty:
        LOGGER.warning(
            "No benign training rows found; fitting autoencoder on all rows"
        )
        benign_train = X_train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = AutoencoderIDSModel(input_dim, hidden_dim, latent_dim)
    detector.network = detector.network.to(device)
    optimizer = torch.optim.Adam(
        detector.network.parameters(),
        lr=float(ae_config.get("learning_rate", 0.001)),
    )
    loss_function = nn.MSELoss()
    dataset = TensorDataset(
        torch.tensor(benign_train.to_numpy(dtype=np.float32))
    )
    loader = DataLoader(
        dataset,
        batch_size=int(ae_config.get("batch_size", 256)),
        shuffle=True,
    )

    detector.network.train()
    for epoch in range(int(ae_config.get("epochs", 30))):
        losses: list[float] = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = detector.network(batch)
            loss = loss_function(reconstructed, batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        LOGGER.info(
            "Autoencoder epoch %s loss=%.6f",
            epoch + 1,
            float(np.mean(losses)) if losses else 0.0,
        )

    detector.network = detector.network.cpu()
    training_errors = detector.reconstruction_error(benign_train)
    detector.threshold = float(
        np.percentile(
            training_errors,
            float(ae_config.get("threshold_percentile", 95)),
        )
    )
    y_true_binary = np.where(y_test == benign_index, 0, 1)
    y_pred_binary = detector.predict(X_test)
    y_probability = detector.predict_proba(X_test)
    metrics = evaluate_predictions(
        y_true_binary,
        y_pred_binary,
        ["BENIGN", "ATTACK"],
        metrics_dir,
        "autoencoder_ids",
        y_probability,
        task="binary_anomaly",
    )

    model_directory = ensure_dir(model_dir)
    state_path = model_directory / "autoencoder_ids.pt"
    torch.save(detector.network.state_dict(), state_path)
    save_model_artifact(
        detector,
        "autoencoder_ids",
        model_directory,
        ["BENIGN", "ATTACK"],
        feature_columns,
        metrics,
        {
            "explainability_type": "deep",
            "model_path": str(state_path),
            "threshold": detector.threshold,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "background_data": benign_train.head(100).to_numpy(dtype=float),
        },
    )
    LOGGER.info("Saved autoencoder detector to %s", state_path)
    return AutoencoderResult(
        model=detector,
        model_path=state_path,
        threshold=detector.threshold,
        metrics=metrics,
    )


def main(argv: Iterable[str] | None = None) -> None:
    """Train the PyTorch autoencoder IDS baseline."""
    parser = argparse.ArgumentParser(
        description="Train PyTorch autoencoder IDS baseline."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", choices=list(SUPPORTED_DATASETS))
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    dataset = normalize_dataset_name(
        args.dataset or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(dataset)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        feature_columns,
    ) = load_processed_arrays(config.path("paths.processed_data_dir"))
    train_autoencoder(
        X_train,
        y_train,
        X_test,
        y_test,
        list(label_encoder.classes_),
        feature_columns,
        config.values,
        str(config.path("paths.model_dir")),
        str(config.path("paths.metrics_dir")),
    )


if __name__ == "__main__":
    main()
