"""Score calibration and uncertainty quantification for audio deepfake detectors.

Implements Platt scaling (logistic regression), isotonic regression, and temperature
scaling to map raw detector scores/logits to well-calibrated posterior probabilities
P(fake | x) in [0.0, 1.0]. Also supports uncertainty band classification and ECE computation.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, log_loss
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE partitions probability predictions into equal-width bins and calculates
    the weighted average difference between predicted confidence and observed accuracy.
    """
    probs = np.asarray(probs, dtype=np.float64).flatten()
    labels = np.asarray(labels, dtype=np.int64).flatten()

    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length")
    if len(probs) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(probs)

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        # Include right edge in last bin
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """Compute points for reliability diagram / calibration curve."""
    probs = np.asarray(probs, dtype=np.float64).flatten()
    labels = np.asarray(labels, dtype=np.int64).flatten()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        bin_centers.append(float((bin_lower + bin_upper) / 2.0))

        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        count = int(np.sum(in_bin))
        bin_counts.append(count)
        if count > 0:
            bin_accuracies.append(float(np.mean(labels[in_bin])))
            bin_confidences.append(float(np.mean(probs[in_bin])))
        else:
            bin_accuracies.append(None)
            bin_confidences.append(None)

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "ece": compute_ece(probs, labels, n_bins),
    }


class ScoreCalibrator:
    """Calibrator mapping raw model scores to posterior fake probabilities."""

    SUPPORTED_METHODS = ("platt", "isotonic", "temperature")

    def __init__(
        self,
        method: str = "platt",
        model_id: Optional[str] = None,
        version: str = "1.0.0",
        score_orientation: str = "higher_is_fake",
        operating_threshold: float = 0.5,
        uncertain_band: Optional[Tuple[float, float]] = None,
    ):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown calibration method: '{method}'. Supported: {self.SUPPORTED_METHODS}"
            )
        self.method = method
        self.model_id = model_id
        self.version = version
        self.score_orientation = score_orientation
        self.operating_threshold = operating_threshold
        self.uncertain_band = uncertain_band  # e.g. (0.40, 0.60)

        self.fitted = False
        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}
        self.created_at: Optional[str] = None
        self.fit_data: Dict[str, Any] = {}
        self.held_out_evaluation: Optional[Dict[str, Any]] = None

    def fit(
        self,
        scores: Union[np.ndarray, List[float]],
        labels: Union[np.ndarray, List[int]],
    ) -> ScoreCalibrator:
        """Fit calibration mapping on development set scores and binary labels (0=real, 1=fake)."""
        scores = np.asarray(scores, dtype=np.float64).flatten()
        labels = np.asarray(labels, dtype=np.int64).flatten()

        if len(scores) != len(labels):
            raise ValueError("scores and labels must have identical length")
        if len(scores) < 4:
            raise ValueError("At least 4 samples required to fit calibration")
        if len(np.unique(labels)) < 2:
            raise ValueError("Both classes (0 and 1) must be present in labels")

        if self.method == "platt":
            self._fit_platt(scores, labels)
        elif self.method == "isotonic":
            self._fit_isotonic(scores, labels)
        elif self.method == "temperature":
            self._fit_temperature(scores, labels)

        self.fitted = True
        self.created_at = datetime.now(timezone.utc).isoformat()

        # Compute calibration quality metrics on development set
        cal_probs = self.predict_proba(scores)
        self.metrics = self.evaluate(scores, labels)
        return self

    def _fit_platt(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit Platt scaling (logistic sigmoid: 1 / (1 + exp(-(a * score + b))))."""
        X = scores.reshape(-1, 1)
        if HAS_SKLEARN:
            clf = LogisticRegression(solver="lbfgs", C=1e5, max_iter=1000)
            clf.fit(X, labels)
            a = float(clf.coef_[0][0])
            b = float(clf.intercept_[0])
        else:
            # Simple fallback gradient descent if sklearn not present
            a = 1.0 if self.score_orientation == "higher_is_fake" else -1.0
            b = 0.0
            lr = 0.05
            for _ in range(500):
                logits = a * scores + b
                p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
                err = p - labels
                grad_a = np.mean(err * scores)
                grad_b = np.mean(err)
                a -= lr * grad_a
                b -= lr * grad_b

        self.params = {"a": a, "b": b}

    def _fit_isotonic(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit non-parametric monotonic isotonic regression."""
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn is required for isotonic calibration")

        # For higher_is_bonafide, higher score means lower fake prob, so increasing=False
        increasing = self.score_orientation != "higher_is_bonafide"
        iso = IsotonicRegression(out_of_bounds="clip", increasing=increasing)
        iso.fit(scores, labels)

        # Store piecewise points so it can be serialized to JSON without pickle
        self.params = {
            "x_thresholds": iso.X_thresholds_.tolist(),
            "y_thresholds": iso.y_thresholds_.tolist(),
            "increasing": increasing,
        }

    def _fit_temperature(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit single temperature scaling parameter T for logits: sigma(score / T)."""
        # Grid search over T in [0.05, 10.0]
        best_t = 1.0
        best_nll = float("inf")
        sign = 1.0 if self.score_orientation == "higher_is_fake" else -1.0

        for t in np.linspace(0.05, 5.0, 200):
            scaled = (sign * scores) / t
            probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -30.0, 30.0)))
            probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
            nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)

        self.params = {"temperature": best_t, "sign": sign}

    def predict_proba(
        self, scores: Union[np.ndarray, List[float], float]
    ) -> Union[np.ndarray, float]:
        """Convert raw scores to calibrated fake probabilities in [0.0, 1.0]."""
        if not self.fitted:
            raise RuntimeError("ScoreCalibrator must be fitted before predict_proba()")

        is_scalar = np.isscalar(scores) or (isinstance(scores, np.ndarray) and scores.ndim == 0)
        s = np.asarray([scores] if is_scalar else scores, dtype=np.float64)

        if self.method == "platt":
            a = self.params["a"]
            b = self.params["b"]
            logits = a * s + b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        elif self.method == "isotonic":
            # Piecewise linear interpolation from serialized thresholds
            x_th = np.array(self.params["x_thresholds"])
            y_th = np.array(self.params["y_thresholds"])
            if self.params.get("increasing", True):
                probs = np.interp(s, x_th, y_th, left=y_th[0], right=y_th[-1])
            else:
                # np.interp requires monotonically increasing x
                probs = np.interp(s, x_th, y_th, left=y_th[0], right=y_th[-1])
        elif self.method == "temperature":
            t = self.params["temperature"]
            sign = self.params.get("sign", 1.0)
            scaled = (sign * s) / t
            probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -60.0, 60.0)))
        else:
            raise ValueError(f"Unknown method: {self.method}")

        probs = np.clip(probs, 0.0, 1.0)
        return float(probs[0]) if is_scalar else probs

    def predict_verdict(
        self, score: float, uncertain_band: Optional[Tuple[float, float]] = None
    ) -> str:
        """Classify a raw score into 'fake', 'real', or 'uncertain'."""
        prob = float(self.predict_proba(score))
        band = uncertain_band if uncertain_band is not None else self.uncertain_band

        if band is not None:
            lower, upper = band
            if lower <= prob <= upper:
                return "uncertain"

        return "fake" if prob >= self.operating_threshold else "real"

    def evaluate(
        self,
        scores: Union[np.ndarray, List[float]],
        labels: Union[np.ndarray, List[int]],
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """Evaluate calibration performance on a validation or test set."""
        scores = np.asarray(scores, dtype=np.float64).flatten()
        labels = np.asarray(labels, dtype=np.int64).flatten()
        probs = self.predict_proba(scores)

        ece = compute_ece(probs, labels, n_bins=n_bins)
        brier = float(np.mean((probs - labels) ** 2))

        clipped_probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        nll = float(-np.mean(labels * np.log(clipped_probs) + (1 - labels) * np.log(1 - clipped_probs)))

        # Accuracy at operating threshold
        preds = (probs >= self.operating_threshold).astype(int)
        accuracy = float(np.mean(preds == labels))

        return {
            "ece": round(ece, 5),
            "brier_score": round(brier, 5),
            "log_loss": round(nll, 5),
            "accuracy": round(accuracy, 4),
            "sample_count": len(labels),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save calibrator configuration and fitted parameters to human-readable JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "format_version": "1.0",
            "model_id": self.model_id,
            "version": self.version,
            "method": self.method,
            "score_orientation": self.score_orientation,
            "operating_threshold": self.operating_threshold,
            "uncertain_band": list(self.uncertain_band) if self.uncertain_band else None,
            "fitted": self.fitted,
            "params": self.params,
            "metrics": self.metrics,
            "fit_data": self.fit_data,
            "held_out_evaluation": self.held_out_evaluation,
            "created_at": self.created_at,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> ScoreCalibrator:
        """Load calibrator from JSON artifact."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        band = tuple(data["uncertain_band"]) if data.get("uncertain_band") else None
        cal = cls(
            method=data["method"],
            model_id=data.get("model_id"),
            version=data.get("version", "1.0.0"),
            score_orientation=data.get("score_orientation", "higher_is_fake"),
            operating_threshold=float(data.get("operating_threshold", 0.5)),
            uncertain_band=band,
        )
        cal.fitted = data.get("fitted", True)
        cal.params = data.get("params", {})
        cal.metrics = data.get("metrics", {})
        cal.fit_data = data.get("fit_data", {})
        cal.held_out_evaluation = data.get("held_out_evaluation")
        cal.created_at = data.get("created_at")
        return cal
