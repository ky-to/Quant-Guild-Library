from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM

from .config import StrategyConfig


def _clean_returns(returns: np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).reshape(-1, 1)
    return arr[np.isfinite(arr).all(axis=1)]


def _build_model(
    returns: np.ndarray,
    n_components: int,
    covariance_type: str,
    cfg: StrategyConfig,
    n_iter: int,
) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=cfg.random_state,
        min_covar=1e-6,
        tol=1e-4,
    )

    model.init_params = ""
    model.params = "stmc"

    model.startprob_ = np.full(n_components, 1.0 / n_components)
    model.transmat_ = np.full((n_components, n_components), 1.0 / n_components)

    quantiles = np.linspace(0.15, 0.85, n_components)
    seeded_means = np.quantile(returns[:, 0], quantiles).reshape(-1, 1)
    base_var = max(float(np.var(returns[:, 0])), 1e-6)

    model.means_ = seeded_means

    if covariance_type == "full":
        model.covars_ = np.array([[[base_var]] for _ in range(n_components)], dtype=float)
    else:
        model.covars_ = np.full((n_components, 1), base_var, dtype=float)

    return model


def _valid_model(model: GaussianHMM) -> bool:
    params = [model.startprob_, model.transmat_, model.means_, model.covars_]
    if any(not np.all(np.isfinite(param)) for param in params):
        return False

    row_sums = model.transmat_.sum(axis=1)
    if np.any(row_sums <= 0):
        return False

    return True


def fit_hmm(returns: np.ndarray, cfg: StrategyConfig, n_iter: Optional[int] = None) -> GaussianHMM:
    """Fit a Gaussian HMM to return data with fallbacks for unstable windows."""
    arr = _clean_returns(returns)
    if len(arr) < max(cfg.n_regimes * 20, 60):
        raise ValueError(f"Not enough clean samples for HMM fit: {len(arr)}")

    if float(np.std(arr[:, 0])) < 1e-12:
        raise ValueError("Return variance is too small for HMM fit")

    candidates = [(cfg.n_regimes, cfg.covariance_type)]
    if cfg.covariance_type != "diag":
        candidates.append((cfg.n_regimes, "diag"))
    if cfg.n_regimes > 2:
        candidates.append((2, "diag"))

    tried = []
    seen = set()

    for n_components, covariance_type in candidates:
        if (n_components, covariance_type) in seen:
            continue
        seen.add((n_components, covariance_type))

        try:
            model = _build_model(
                returns=arr,
                n_components=n_components,
                covariance_type=covariance_type,
                cfg=cfg,
                n_iter=n_iter or cfg.hmm_iter,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message="Some rows of transmat_ have zero sum.*")
                model.fit(arr)

            if not _valid_model(model):
                raise ValueError("Degenerate fitted HMM parameters")

            return model

        except Exception as exc:
            tried.append(f"{n_components} regimes / {covariance_type}: {exc}")

    raise ValueError("HMM fit failed after retries: " + " | ".join(tried))


def label_regimes(
    model: GaussianHMM,
    returns: np.ndarray,
) -> Tuple[Dict[str, int], List[Tuple[int, float, float]]]:
    arr = _clean_returns(returns)
    predictions = model.predict(arr)
    stats: List[Tuple[int, float, float]] = []

    for regime_id in range(model.n_components):
        mask = predictions == regime_id
        mean_ret = arr[mask].mean() if mask.sum() > 0 else 0.0
        vol = arr[mask].std() if mask.sum() > 1 else 0.0
        stats.append((regime_id, float(mean_ret), float(vol)))

    stats = sorted(stats, key=lambda item: item[1])

    labels: Dict[str, int] = {
        "bear": stats[0][0],
        "bull": stats[-1][0],
    }

    if len(stats) == 3:
        labels["neutral"] = stats[1][0]
    elif len(stats) > 3:
        for idx, (regime_id, _, _) in enumerate(stats[1:-1], start=1):
            labels[f"neutral_{idx}"] = regime_id

    return labels, stats


def regime_to_signal(regime: int, labels: Dict[str, int], cfg: StrategyConfig) -> float:
    if regime == labels["bull"]:
        return cfg.signal_bull
    if regime == labels.get("neutral"):
        return cfg.signal_neutral
    if regime == labels["bear"]:
        return cfg.signal_bear
    return cfg.signal_neutral