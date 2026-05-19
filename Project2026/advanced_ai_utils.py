"""Utility helpers for the Advanced AI supplement notebooks.

This module is intentionally lightweight and notebook-friendly.
It groups together reusable helpers for:

* tabular security / manipulation analysis,
* fairness / group-metric analysis,
* OCR perturbation sweeps,
* small explanation-oriented data summaries.

The functions are designed to work with the existing notebooks in this
repository and to stay flexible about the underlying model interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


def _as_series(sample: Any, feature_names: Sequence[str] | None = None) -> pd.Series:
    """Return a 1-row sample as a pandas Series."""
    if isinstance(sample, pd.Series):
        return sample.copy()
    if isinstance(sample, pd.DataFrame):
        if len(sample) != 1:
            raise ValueError("Expected a single-row DataFrame.")
        return sample.iloc[0].copy()
    if isinstance(sample, Mapping):
        return pd.Series(dict(sample))
    if isinstance(sample, np.ndarray):
        if sample.ndim == 1:
            if feature_names is None:
                feature_names = [f"x{i}" for i in range(sample.shape[0])]
            return pd.Series(sample, index=list(feature_names))
        if sample.ndim == 2 and sample.shape[0] == 1:
            if feature_names is None:
                feature_names = [f"x{i}" for i in range(sample.shape[1])]
            return pd.Series(sample[0], index=list(feature_names))
    raise TypeError(f"Unsupported sample type: {type(sample)!r}")


def _call_score_fn(score_fn: Callable[[Any], Any], sample: Any) -> float:
    """Call a score function and coerce the output to a scalar float."""
    out = score_fn(sample)
    if isinstance(out, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
        arr = np.asarray(out)
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1:
            if arr.size == 1:
                return float(arr[0])
            return float(arr[-1])
        if arr.ndim >= 2:
            return float(arr.reshape(-1)[-1])
    return float(out)


# ---------------------------------------------------------------------
# Threat modeling / feature manipulation
# ---------------------------------------------------------------------


def build_feature_risk_table(
    feature_names: Sequence[str],
    easy_to_fake: Iterable[str] = (),
    hard_to_fake: Iterable[str] = (),
    sensitive: Iterable[str] = (),
) -> pd.DataFrame:
    """Create a compact risk table for tabular features.

    The notebook should define the feature lists based on domain knowledge.
    This helper simply structures the result.
    """
    easy = set(easy_to_fake)
    hard = set(hard_to_fake)
    sens = set(sensitive)

    rows = []
    for feat in feature_names:
        if feat in easy:
            risk = "easy-to-fake"
        elif feat in hard:
            risk = "hard-to-fake"
        else:
            risk = "unclear"
        rows.append(
            {
                "feature": feat,
                "risk_class": risk,
                "sensitive_proxy": feat in sens,
            }
        )
    return pd.DataFrame(rows)


def perturb_numeric(value: float, bounds: tuple[float, float] | None = None, direction: str = "down") -> float:
    """Move a numeric value toward a realistic bound."""
    if bounds is None:
        return float(value)
    low, high = bounds
    return float(low if direction == "down" else high)


def candidate_values_for_feature(
    feature_name: str,
    sample: pd.Series,
    feature_specs: Mapping[str, Mapping[str, Any]],
) -> list[Any]:
    """Return candidate replacement values for a single feature.

    Expected feature spec format:
        {
            "kind": "numeric" | "binary" | "categorical" | "ordinal",
            "values": [...],            # for categorical / ordinal
            "bounds": (low, high),      # for numeric
        }
    """
    spec = feature_specs.get(feature_name, {})
    kind = spec.get("kind", "categorical")
    current = sample[feature_name]

    if kind == "numeric":
        if "values" in spec:
            values = list(spec["values"])
            if current not in values:
                values.append(current)
            return [v for v in values if v != current]
        bounds = spec.get("bounds")
        values = [float(current)]
        if bounds is not None:
            values.extend([bounds[0], bounds[1], float(np.mean(bounds))])
        return sorted({float(v) for v in values})

    if kind in {"binary", "categorical", "ordinal"}:
        values = list(spec.get("values", []))
        if current not in values:
            values.append(current)
        return [v for v in values if v != current]

    raise ValueError(f"Unsupported feature kind: {kind!r}")


@dataclass
class AttackStep:
    step: int
    feature: str
    old_value: Any
    new_value: Any
    score_before: float
    score_after: float


def greedy_counterfactual_attack(
    sample: Any,
    score_fn: Callable[[Any], float],
    feature_specs: Mapping[str, Mapping[str, Any]],
    threshold: float = 50.0,
    max_steps: int = 8,
    minimize_score: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """Greedy single-feature counterfactual attack.

    The function repeatedly tries all allowed single-feature changes and
    applies the best improvement at each step. This keeps the perturbation
    small and easy to explain in a notebook.
    """
    current = _as_series(sample)
    score = _call_score_fn(score_fn, current)
    path: list[AttackStep] = []

    for step in range(1, max_steps + 1):
        if score <= threshold and minimize_score:
            break
        if score >= threshold and not minimize_score:
            break

        best = None
        best_score = score

        for feature in current.index:
            spec = feature_specs.get(feature, {})
            candidates = candidate_values_for_feature(feature, current, feature_specs)
            if not candidates:
                continue

            for candidate in candidates:
                trial = current.copy()
                trial[feature] = candidate
                trial_score = _call_score_fn(score_fn, trial)
                if minimize_score:
                    improved = trial_score < best_score
                else:
                    improved = trial_score > best_score
                if improved:
                    best = (feature, current[feature], candidate, trial, trial_score)
                    best_score = trial_score

        if best is None:
            break

        feature, old_value, new_value, current, score = best
        path.append(
            AttackStep(
                step=step,
                feature=feature,
                old_value=old_value,
                new_value=new_value,
                score_before=path[-1].score_after if path else _call_score_fn(score_fn, sample),
                score_after=score,
            )
        )

    if path:
        path[0].score_before = _call_score_fn(score_fn, sample)

    path_df = pd.DataFrame([step.__dict__ for step in path])
    return current, path_df


def brute_force_attack_search(
    sample: Any,
    score_fn: Callable[[Any], float],
    feature_specs: Mapping[str, Mapping[str, Any]],
    manipulable_features: Sequence[str],
    threshold: float = 50.0,
    max_changed_features: int = 3,
    minimize_score: bool = True,
) -> pd.DataFrame:
    """Small combinatorial search over a limited set of manipulable features.

    Useful when the greedy attack stalls and the feature space is small.
    """
    base = _as_series(sample)
    base_score = _call_score_fn(score_fn, base)
    rows = []

    features = list(manipulable_features)
    for k in range(1, min(max_changed_features, len(features)) + 1):
        for subset in combinations(features, k):
            # Cartesian expansion of candidate values for the chosen subset.
            candidate_lists = []
            for feat in subset:
                candidate_lists.append(candidate_values_for_feature(feat, base, feature_specs))
            if any(len(cands) == 0 for cands in candidate_lists):
                continue

            def _expand(idx: int, current: pd.Series):
                if idx == len(subset):
                    yield current
                    return
                feat = subset[idx]
                for cand in candidate_lists[idx]:
                    trial = current.copy()
                    trial[feat] = cand
                    yield from _expand(idx + 1, trial)

            seen = set()
            for trial in _expand(0, base.copy()):
                key = tuple(trial[feat] for feat in subset)
                if key in seen:
                    continue
                seen.add(key)
                score = _call_score_fn(score_fn, trial)
                rows.append(
                    {
                        "changed_features": subset,
                        "n_changed": len(subset),
                        "score_before": base_score,
                        "score_after": score,
                        "delta": score - base_score,
                        "meets_threshold": score <= threshold if minimize_score else score >= threshold,
                        "sample": trial,
                    }
                )

    result = pd.DataFrame(rows)
    if not result.empty:
        if minimize_score:
            result = result.sort_values(["meets_threshold", "score_after"], ascending=[False, True])
        else:
            result = result.sort_values(["meets_threshold", "score_after"], ascending=[False, False])
    return result.reset_index(drop=True)


def local_numeric_sensitivity(
    sample: Any,
    score_fn: Callable[[Any], float],
    numeric_features: Sequence[str],
    step_sizes: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Finite-difference style sensitivity for numeric features."""
    base = _as_series(sample)
    base_score = _call_score_fn(score_fn, base)
    rows = []
    step_sizes = dict(step_sizes or {})

    for feat in numeric_features:
        step = float(step_sizes.get(feat, 1.0))
        for direction in (-1.0, 1.0):
            trial = base.copy()
            trial[feat] = float(trial[feat]) + direction * step
            score = _call_score_fn(score_fn, trial)
            rows.append(
                {
                    "feature": feat,
                    "direction": "down" if direction < 0 else "up",
                    "step": step,
                    "score_before": base_score,
                    "score_after": score,
                    "delta": score - base_score,
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "delta"], ascending=[True, True])


def attack_summary_table(
    attacks: Mapping[str, pd.DataFrame],
    threshold: float = 50.0,
) -> pd.DataFrame:
    """Summarize attack outcomes across multiple models."""
    rows = []
    for model_name, df in attacks.items():
        if df is None or df.empty:
            rows.append(
                {
                    "model": model_name,
                    "best_score": np.nan,
                    "score_drop": np.nan,
                    "steps": 0,
                    "reached_threshold": False,
                }
            )
            continue
        last = df.iloc[-1]
        first = df.iloc[0]
        rows.append(
            {
                "model": model_name,
                "best_score": float(last["score_after"]),
                "score_drop": float(first["score_before"] - last["score_after"]),
                "steps": int(len(df)),
                "reached_threshold": float(last["score_after"]) < threshold,
            }
        )
    return pd.DataFrame(rows)


def plot_attack_paths(attack_tables: Mapping[str, pd.DataFrame], threshold: float = 50.0, ax: plt.Axes | None = None):
    """Plot score trajectories for one or more attack runs."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    for model_name, df in attack_tables.items():
        if df is None or df.empty:
            continue
        scores = [float(df.iloc[0]["score_before"])] + df["score_after"].astype(float).tolist()
        ax.plot(range(len(scores)), scores, marker="o", label=model_name)

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"threshold={threshold}")
    ax.set_xlabel("Attack step")
    ax.set_ylabel("Predicted score")
    ax.set_title("Prediction shift under greedy manipulation")
    ax.legend()
    return ax


# ---------------------------------------------------------------------
# Fairness / group metrics
# ---------------------------------------------------------------------


def group_metrics_table(
    y_true: Sequence[int | float],
    y_pred: Sequence[int | float],
    groups: Sequence[Any],
    positive_label: int | float = 1,
) -> pd.DataFrame:
    """Compute core metrics per group.

    The positive class is the support-worthy / at-risk class in the notebook.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups})
    rows = []
    for group_name, g in df.groupby("group", dropna=False):
        yt = g["y_true"].to_numpy()
        yp = g["y_pred"].to_numpy()
        yt_pos = yt == positive_label
        yp_pos = yp == positive_label
        tp = int(np.sum(yt_pos & yp_pos))
        tn = int(np.sum(~yt_pos & ~yp_pos))
        fp = int(np.sum(~yt_pos & yp_pos))
        fn = int(np.sum(yt_pos & ~yp_pos))
        rows.append(
            {
                "group": group_name,
                "support": int(len(g)),
                "accuracy": accuracy_score(yt, yp),
                "precision": precision_score(yt, yp, pos_label=positive_label, zero_division=0),
                "recall": recall_score(yt, yp, pos_label=positive_label, zero_division=0),
                "f1": f1_score(yt, yp, pos_label=positive_label, zero_division=0),
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else np.nan,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) else np.nan,
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def disparity_table(metrics_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute simple max-min disparity for a fairness metric."""
    if metric not in metrics_df.columns:
        raise KeyError(f"{metric!r} not present in metrics DataFrame.")
    values = metrics_df[metric].astype(float)
    return pd.DataFrame(
        {
            "metric": [metric],
            "min": [values.min()],
            "max": [values.max()],
            "range": [values.max() - values.min()],
            "std": [values.std(ddof=0)],
        }
    )


def plot_group_metrics(metrics_df: pd.DataFrame, metrics: Sequence[str] = ("accuracy", "recall", "false_negative_rate"), ax: plt.Axes | None = None):
    """Bar plot for several metrics by group."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    plot_df = metrics_df.melt(id_vars="group", value_vars=list(metrics), var_name="metric", value_name="value")
    for metric in metrics:
        subset = plot_df[plot_df["metric"] == metric]
        ax.bar(subset["group"].astype(str) + " / " + metric, subset["value"], label=metric, alpha=0.8)
    ax.set_ylabel("Score")
    ax.set_title("Fairness metrics by group")
    ax.tick_params(axis="x", rotation=45)
    return ax


# ---------------------------------------------------------------------
# OCR perturbation analysis
# ---------------------------------------------------------------------


try:
    from PIL import Image, ImageEnhance, ImageFilter
except Exception:  # pragma: no cover
    Image = None
    ImageEnhance = None
    ImageFilter = None


def _to_pil_gray(image: np.ndarray):
    if Image is None:
        raise ImportError("Pillow is required for OCR perturbation utilities.")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _to_float01(image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def perturb_image(image: np.ndarray, kind: str, severity: float = 0.3, seed: int | None = None) -> np.ndarray:
    """Apply a small OCR-style perturbation to a grayscale image.

    Supported kinds: rotation, shift, blur, noise, contrast.
    The output is returned in float format in [0, 1].
    """
    rng = np.random.default_rng(seed)
    pil = _to_pil_gray(_to_float01(image))

    if kind == "rotation":
        angle = float(severity) * 20.0
        pil = pil.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
    elif kind == "shift":
        dx = int(round(rng.uniform(-1.0, 1.0) * severity * 4))
        dy = int(round(rng.uniform(-1.0, 1.0) * severity * 4))
        canvas = Image.new("L", pil.size, 255)
        canvas.paste(pil, (dx, dy))
        pil = canvas
    elif kind == "blur":
        radius = max(0.1, float(severity) * 1.5)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    elif kind == "noise":
        arr = np.asarray(pil, dtype=np.float32)
        noise = rng.normal(0.0, severity * 20.0, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="L")
    elif kind == "contrast":
        factor = max(0.2, 1.0 - severity)
        pil = ImageEnhance.Contrast(pil).enhance(factor)
    else:
        raise ValueError(f"Unsupported perturbation kind: {kind!r}")

    return _to_float01(pil)


def confidence_sweep(
    model: Any,
    image: np.ndarray,
    transform_kinds: Sequence[str],
    severities: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    predict_fn: Callable[[Any, np.ndarray], np.ndarray] | None = None,
    class_idx: int | None = None,
) -> pd.DataFrame:
    """Measure model confidence under a set of perturbations."""
    if predict_fn is None:
        predict_fn = lambda m, x: m.predict(x[None, ...], verbose=0)[0]

    rows = []
    for kind in transform_kinds:
        for severity in severities:
            perturbed = perturb_image(image, kind=kind, severity=severity)
            proba = np.asarray(predict_fn(model, perturbed), dtype=np.float32)
            pred_idx = int(np.argmax(proba))
            rows.append(
                {
                    "kind": kind,
                    "severity": float(severity),
                    "pred_idx": pred_idx,
                    "confidence": float(proba[pred_idx]),
                    "target_confidence": float(proba[class_idx]) if class_idx is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def top_confusion_pairs(cm: np.ndarray, label_names: Sequence[str], top_n: int = 10) -> pd.DataFrame:
    """Return the strongest off-diagonal confusions."""
    rows = []
    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count > 0:
                rows.append(
                    {
                        "true_label": label_names[true_idx],
                        "pred_label": label_names[pred_idx],
                        "count": count,
                    }
                )
    out = pd.DataFrame(rows).sort_values("count", ascending=False)
    return out.head(top_n).reset_index(drop=True)


def per_class_recall_table(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> pd.DataFrame:
    """Per-class recall and support."""
    rows = []
    for idx, name in enumerate(class_names):
        mask = np.asarray(y_true) == idx
        support = int(mask.sum())
        if support == 0:
            recall = np.nan
        else:
            recall = recall_score(np.asarray(y_true), np.asarray(y_pred), labels=[idx], average="macro", zero_division=0)
        rows.append({"class": name, "support": support, "recall": recall})
    return pd.DataFrame(rows).sort_values(["recall", "support"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Explanation helpers
# ---------------------------------------------------------------------


def permutation_importance_table(perm_result, feature_names: Sequence[str]) -> pd.DataFrame:
    """Convert sklearn permutation_importance output into a tidy table."""
    return pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance_mean": np.asarray(perm_result.importances_mean),
            "importance_std": np.asarray(perm_result.importances_std),
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def local_contribution_table(feature_names: Sequence[str], values: Sequence[float], contributions: Sequence[float]) -> pd.DataFrame:
    """Pack local explanation values into a sortable table."""
    df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "value": list(values),
            "contribution": list(contributions),
        }
    )
    return df.reindex(df["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def maybe_import_shap():
    """Import SHAP lazily so notebooks can degrade gracefully if missing."""
    try:
        import shap  # type: ignore

        return shap
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "SHAP is not available in the current environment. "
            "Install it or skip the SHAP-specific notebook cells."
        ) from exc
