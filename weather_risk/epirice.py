"""
weather_risk/epirice.py
=======================
Python wrapper around run_epirice.R (epicrop / NASA POWER).

Public API
----------
run_epirice(lat, lon, establishment_date, cache_dir) -> pd.DataFrame
    Daily disease intensity, one column per EPIRICE disease, indexed by date.
    Calls ``Rscript run_epirice.R`` once per unique (lat, lon, date) triple;
    subsequent calls use a disk cache and skip R entirely.

risk_prior(curves, on_date, train_prior, eps, min_healthy, lam) -> np.ndarray
    Convert daily intensity curves into a class prior over CLASSES.
    The pure weather prior is computed from intensity curves, then blended
    with train_prior via lam: lam=1 gives pure weather, lam=0 gives
    train_prior unchanged.  Default lam=0.5.

prior_shift(probs, train_prior, new_prior) -> np.ndarray
    Reweight class posteriors from train_prior to new_prior
    (Saerens et al. 2002).  Handles a single sample (shape [C]) or a
    batch (shape [N, C]).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Class registry  (fixed order for the whole project)
# ---------------------------------------------------------------------------
CLASSES: list[str] = [
    "healthy",
    "bacterial_leaf_blight",
    "brown_spot",
    "leaf_blast",
    "sheath_blight",
    "tungro",
    "leaf_scald",
    "narrow_brown_spot",
    "rice_hispa",
]

# The five diseases that EPIRICE models
EPIRICE_DISEASES: list[str] = [
    "bacterial_blight",
    "brown_spot",
    "leaf_blast",
    "sheath_blight",
    "tungro",
]

# EPIRICE disease name → project CLASSES name
_EPIRICE_TO_CLASS: dict[str, str] = {
    "bacterial_blight": "bacterial_leaf_blight",
    "brown_spot":       "brown_spot",
    "leaf_blast":       "leaf_blast",
    "sheath_blight":    "sheath_blight",
    "tungro":           "tungro",
}

# Classes whose prior is held neutral (equal to training prior)
_NEUTRAL_CLASSES: set[str] = {"leaf_scald", "narrow_brown_spot", "rice_hispa"}

_R_SCRIPT = Path(__file__).parent / "run_epirice.R"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_key(lat: float, lon: float, establishment_date: str) -> str:
    tag = f"{lat:.6f}_{lon:.6f}_{establishment_date}"
    return hashlib.md5(tag.encode()).hexdigest()[:16]


def _load_curves(run_dir: Path) -> pd.DataFrame:
    """
    Read one CSV per EPIRICE disease from *run_dir* and merge into a DataFrame.

    Expected CSV columns: ``dates`` (parseable as dates), ``intensity`` or
    ``severity`` (float in [0, 1]).  Handles both column names produced by
    different epicrop versions.
    """
    run_dir = Path(run_dir)
    series: dict[str, pd.Series] = {}

    for disease in EPIRICE_DISEASES:
        path = run_dir / f"{disease}.csv"
        df = pd.read_csv(path, parse_dates=["dates"])

        if "intensity" in df.columns:
            col = "intensity"
        elif "severity" in df.columns:
            col = "severity"
        else:
            raise ValueError(
                f"No 'intensity' or 'severity' column in {path}. "
                f"Found: {list(df.columns)}"
            )

        series[disease] = df.set_index("dates")[col].rename(disease)

    return pd.DataFrame(series)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_epirice(
    lat: float,
    lon: float,
    establishment_date: str,
    cache_dir: str | Path = ".cache/epirice",
) -> pd.DataFrame:
    """
    Run the five EPIRICE rice disease models for a 120-day season.

    Parameters
    ----------
    lat, lon :
        Decimal-degree location.
    establishment_date :
        Crop emergence date, ``'YYYY-MM-DD'``.
    cache_dir :
        Root directory for on-disk caching.  Results are stored under a
        hash of ``(lat, lon, establishment_date)`` and reused on repeat calls.

    Returns
    -------
    pd.DataFrame
        Index: date (daily, 121 rows).
        Columns: ``EPIRICE_DISEASES`` — daily intensity in [0, 1].

    Raises
    ------
    RuntimeError
        If the R script exits with a non-zero status.
    """
    cache_dir = Path(cache_dir)
    key = _cache_key(lat, lon, establishment_date)
    run_dir = cache_dir / key

    # Cache hit: all five disease CSVs already present
    if all((run_dir / f"{d}.csv").exists() for d in EPIRICE_DISEASES):
        return _load_curves(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "Rscript",
            str(_R_SCRIPT),
            str(lat),
            str(lon),
            establishment_date,
            str(run_dir),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"run_epirice.R failed (exit {result.returncode}).\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return _load_curves(run_dir)


def risk_prior(
    curves: pd.DataFrame,
    on_date: str,
    train_prior: np.ndarray | list[float],
    eps: float = 0.01,
    min_healthy: float = 0.2,
    lam: float = 0.5,
) -> np.ndarray:
    """
    Build a class prior from EPIRICE intensity curves for a specific date.

    Algorithm
    ---------
    1. Compute the **pure weather prior** from intensity curves:

       * Neutral classes (leaf_scald, narrow_brown_spot, rice_hispa) keep their
         train_prior share unchanged.  Their combined mass is ``neutral_mass``;
         the remaining ``active_budget = 1 - neutral_mass`` is allocated to
         'healthy' and the five EPIRICE-mapped classes.

       * Within the active budget::

               healthy_frac = max(min_healthy, 1 - Σ intensities)   # ∈ [0, 1]
               epirice_frac = 1 - healthy_frac

         EPIRICE class shares: proportional to (intensity + eps), scaled to
         fill ``epirice_frac × active_budget``.

    2. **Blend** with the training prior::

           result = lam * weather_prior + (1 - lam) * train_prior

       ``lam=1`` gives the pure weather prior (maximum weather influence);
       ``lam=0`` returns *train_prior* unchanged (weather layer disabled).
       The default ``lam=0.5`` prevents near-zero EPIRICE classes at the start
       of the season from completely suppressing disease predictions in
       ``prior_shift``.

    Parameters
    ----------
    curves :
        DataFrame from :func:`run_epirice` (index=date, cols=EPIRICE_DISEASES).
    on_date :
        Target date string ``'YYYY-MM-DD'``.  Must lie within the simulated
        season; a :class:`ValueError` is raised otherwise.
    train_prior :
        Class prior from training data, shape ``(len(CLASSES),)``.
    eps :
        Small additive constant so that a zero-intensity disease still
        receives a non-zero weather-prior share.
    min_healthy :
        Minimum fraction of ``active_budget`` guaranteed to 'healthy' in the
        weather prior.
    lam :
        Blending weight in [0, 1].  ``lam=1`` → pure weather prior;
        ``lam=0`` → train_prior unchanged.  Ablate at 0, 0.25, 0.5, 0.75, 1.

    Returns
    -------
    np.ndarray shape ``(len(CLASSES),)``, sums to 1.
    """
    train_prior = np.asarray(train_prior, dtype=float)
    if train_prior.shape != (len(CLASSES),):
        raise ValueError(
            f"train_prior must have {len(CLASSES)} elements, "
            f"got shape {train_prior.shape}."
        )

    ts = pd.Timestamp(on_date)
    if ts not in curves.index:
        raise ValueError(
            f"Date {ts.date()} is outside the simulated season "
            f"({curves.index[0].date()} – {curves.index[-1].date()})."
        )

    intensities = curves.loc[ts]          # Series, index = EPIRICE_DISEASES

    # -- budget split ---------------------------------------------------------
    neutral_idx  = [CLASSES.index(c) for c in _NEUTRAL_CLASSES]
    neutral_mass = float(train_prior[neutral_idx].sum())
    active_budget = 1.0 - neutral_mass

    total_intensity = float(intensities.clip(lower=0).sum())
    healthy_frac    = max(min_healthy, 1.0 - total_intensity)   # fraction of [0,1]
    epirice_frac    = 1.0 - healthy_frac

    # Proportional weights for EPIRICE classes (softmax-like with eps floor)
    weights = np.array([float(intensities[d]) + eps for d in EPIRICE_DISEASES],
                       dtype=float)
    weights /= weights.sum()

    # -- assemble prior -------------------------------------------------------
    prior = np.zeros(len(CLASSES), dtype=float)

    for idx in neutral_idx:
        prior[idx] = train_prior[idx]

    prior[CLASSES.index("healthy")] = healthy_frac * active_budget

    for i, disease in enumerate(EPIRICE_DISEASES):
        cls = _EPIRICE_TO_CLASS[disease]
        prior[CLASSES.index(cls)] = weights[i] * epirice_frac * active_budget

    # Renormalise weather prior for floating-point safety, then blend
    prior /= prior.sum()
    return lam * prior + (1.0 - lam) * train_prior


def prior_shift(
    probs: np.ndarray,
    train_prior: np.ndarray | list[float],
    new_prior: np.ndarray | list[float],
) -> np.ndarray:
    """
    Reweight class posteriors from *train_prior* to *new_prior*.

    Implements the prior-shift correction of Saerens et al. (2002)::

        p_new(C|x)  ∝  p_train(C|x) × new_prior[C] / train_prior[C]

    Parameters
    ----------
    probs :
        Softmax probabilities, shape ``(C,)`` or ``(N, C)``.
    train_prior :
        Class prior used during model training, shape ``(C,)``.
    new_prior :
        Target class prior, shape ``(C,)``.

    Returns
    -------
    np.ndarray — same shape as *probs*, rows renormalised to 1.
    """
    probs       = np.asarray(probs,       dtype=float)
    train_prior = np.asarray(train_prior, dtype=float)
    new_prior   = np.asarray(new_prior,   dtype=float)

    # Avoid division by zero: if train_prior[c] == 0 the ratio is 0.
    # np.errstate suppresses the spurious RuntimeWarning from the masked branch.
    with np.errstate(divide="ignore", invalid="ignore"):
        raw   = np.where(train_prior > 0, new_prior / train_prior, 0.0)
    ratio = raw

    if probs.ndim == 1:
        shifted = probs * ratio
        return shifted / shifted.sum()

    # Batch: (N, C)
    shifted = probs * ratio[np.newaxis, :]
    return shifted / shifted.sum(axis=1, keepdims=True)
