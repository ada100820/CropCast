"""
tests/test_weather.py
=====================
Tests for weather_risk.epirice.  Runs entirely without R or network access:
fixture CSVs under tests/fixtures/ stand in for real epicrop output, and
subprocess calls are mocked where tested.

Fixture data covers 20 days (2023-07-15 – 2023-08-03):
  * 2023-07-15: all intensities = 0.0   → season start, zero-intensity case
  * 2023-07-25: total intensity ≈ 0.52  → healthy floor is NOT active
  * 2023-08-03: total intensity = 0.95  → healthy floor IS active

lam blending (tests 9-12):
  lam=0  → output == train_prior exactly
  lam=1  → output == pure weather prior (old behaviour)
  lam=0.5 is the midpoint, verified via linearity
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import pandas as pd

from weather_risk.epirice import (
    CLASSES,
    EPIRICE_DISEASES,
    _cache_key,
    _load_curves,
    prior_shift,
    risk_prior,
    run_epirice,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Training prior (sums to 1.0)
TRAIN_PRIOR = np.array([
    0.30,   # healthy
    0.10,   # bacterial_leaf_blight
    0.10,   # brown_spot
    0.10,   # leaf_blast
    0.10,   # sheath_blight
    0.10,   # tungro
    0.10,   # leaf_scald         ← neutral
    0.05,   # narrow_brown_spot  ← neutral
    0.05,   # rice_hispa         ← neutral
])

NEUTRAL_CLASSES = ["leaf_scald", "narrow_brown_spot", "rice_hispa"]


@pytest.fixture(scope="module")
def curves():
    """Load intensity curves from the fixture CSVs (no subprocess / R needed)."""
    return _load_curves(FIXTURES_DIR)


# ---------------------------------------------------------------------------
# 1. Prior sums to 1
# ---------------------------------------------------------------------------
def test_prior_sums_to_1(curves):
    prior = risk_prior(curves, on_date="2023-07-25", train_prior=TRAIN_PRIOR)
    assert abs(prior.sum() - 1.0) < 1e-9, f"prior sums to {prior.sum()}, expected 1.0"


# ---------------------------------------------------------------------------
# 2. Neutral classes are unchanged
# ---------------------------------------------------------------------------
def test_neutral_classes_unchanged(curves):
    prior = risk_prior(curves, on_date="2023-07-25", train_prior=TRAIN_PRIOR)
    for cls in NEUTRAL_CLASSES:
        idx = CLASSES.index(cls)
        # Because neutral_mass + active_budget = 1.0 exactly, renormalisation
        # leaves neutral entries numerically identical to their training values.
        assert abs(prior[idx] - TRAIN_PRIOR[idx]) < 1e-9, (
            f"{cls}: expected {TRAIN_PRIOR[idx]:.6f}, got {prior[idx]:.6f}"
        )


# ---------------------------------------------------------------------------
# 3. Healthy floor is respected
#    On 2023-08-03 total_intensity = 0.95, so (1 - total) = 0.05 < min_healthy=0.2
#    → healthy_frac is clamped to min_healthy.
#    Expected healthy share: 0.2 × active_budget (= 0.2 × 0.80 = 0.16).
# ---------------------------------------------------------------------------
def test_healthy_floor(curves):
    min_healthy   = 0.2
    neutral_mass  = sum(TRAIN_PRIOR[CLASSES.index(c)] for c in NEUTRAL_CLASSES)
    active_budget = 1.0 - neutral_mass   # = 0.80

    prior = risk_prior(
        curves, on_date="2023-08-03",
        train_prior=TRAIN_PRIOR,
        min_healthy=min_healthy,
    )

    healthy_idx = CLASSES.index("healthy")
    expected_floor = min_healthy * active_budget   # = 0.16
    assert prior[healthy_idx] >= expected_floor - 1e-9, (
        f"healthy={prior[healthy_idx]:.6f} is below floor {expected_floor:.6f}"
    )


# ---------------------------------------------------------------------------
# 4. Date outside simulated season raises ValueError
# ---------------------------------------------------------------------------
def test_date_outside_season_raises(curves):
    with pytest.raises(ValueError, match="outside the simulated season"):
        risk_prior(curves, on_date="2025-01-01", train_prior=TRAIN_PRIOR)


# ---------------------------------------------------------------------------
# 5. prior_shift with uniform train == uniform new  →  input unchanged
#    When both priors are identical, ratio = 1 everywhere, so the output
#    equals the (normalised) input.
# ---------------------------------------------------------------------------
def test_prior_shift_uniform_returns_input_unchanged():
    n       = len(CLASSES)
    uniform = np.full(n, 1.0 / n)
    # A valid probability vector (already sums to 1)
    probs   = np.array([0.10, 0.20, 0.15, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10])
    assert abs(probs.sum() - 1.0) < 1e-12, "fixture probs must sum to 1"

    result = prior_shift(probs, train_prior=uniform, new_prior=uniform)
    np.testing.assert_allclose(result, probs, atol=1e-12)


# ---------------------------------------------------------------------------
# 6. prior_shift preserves batch shape and each row sums to 1
# ---------------------------------------------------------------------------
def test_prior_shift_batch_shape_preserved():
    n       = len(CLASSES)
    uniform = np.full(n, 1.0 / n)
    rng     = np.random.default_rng(42)
    batch   = rng.dirichlet(np.ones(n), size=5)   # shape (5, 9), rows sum to 1

    result = prior_shift(batch, train_prior=uniform, new_prior=uniform)

    assert result.shape == (5, n), f"Expected shape (5, {n}), got {result.shape}"
    np.testing.assert_allclose(result.sum(axis=1), np.ones(5), atol=1e-12,
                               err_msg="Batch rows must sum to 1")


# ---------------------------------------------------------------------------
# 7. run_epirice uses disk cache and skips subprocess on repeat call
# ---------------------------------------------------------------------------
def test_run_epirice_uses_cache_skips_subprocess(tmp_path):
    lat, lon, estab = 21.25, 81.63, "2023-07-15"

    # Pre-populate the cache directory with the fixture CSVs
    key     = _cache_key(lat, lon, estab)
    run_dir = tmp_path / key
    run_dir.mkdir()
    for disease in EPIRICE_DISEASES:
        src = FIXTURES_DIR / f"{disease}.csv"
        (run_dir / f"{disease}.csv").write_bytes(src.read_bytes())

    # subprocess.run must NOT be called because the cache is already populated
    with patch("weather_risk.epirice.subprocess.run") as mock_run:
        result = run_epirice(lat, lon, estab, cache_dir=tmp_path)
        mock_run.assert_not_called()

    assert list(result.columns) == EPIRICE_DISEASES
    assert len(result) == 20   # fixture has 20 rows


# ---------------------------------------------------------------------------
# 8. prior_shift raises / handles zero train_prior entries without crashing
#    (ratio uses np.where so zero-prior classes get 0 in the output)
# ---------------------------------------------------------------------------
def test_prior_shift_zero_train_prior_class():
    n           = len(CLASSES)
    train_prior = np.full(n, 1.0 / n)
    new_prior   = np.full(n, 1.0 / n)
    # Set one training-prior entry to zero
    train_prior[3] = 0.0
    train_prior   /= train_prior.sum()

    probs         = np.full(n, 1.0 / n)
    result        = prior_shift(probs, train_prior, new_prior)
    # Should not raise; class 3 gets 0 weight
    assert result[3] == pytest.approx(0.0, abs=1e-12)
    assert abs(result.sum() - 1.0) < 1e-9


# ===========================================================================
# lam blending tests (9-12)
# ===========================================================================

# ---------------------------------------------------------------------------
# 9. lam=0 returns train_prior exactly
#    The weather signal should be completely ignored.
# ---------------------------------------------------------------------------
def test_lam_zero_returns_train_prior(curves):
    result = risk_prior(curves, on_date="2023-07-25", train_prior=TRAIN_PRIOR,
                        lam=0.0)
    np.testing.assert_allclose(result, TRAIN_PRIOR, atol=1e-12,
                               err_msg="lam=0 must return train_prior unchanged")


# ---------------------------------------------------------------------------
# 10. lam=1 gives the pure weather prior (old behaviour before the lam fix)
#     Verified via linearity: p(lam=0.5) must equal (p(lam=0) + p(lam=1)) / 2.
#     Also checks that lam=1 actually differs from train_prior on a date with
#     non-zero intensities (i.e. the weather signal is doing something).
# ---------------------------------------------------------------------------
def test_lam_one_is_pure_weather_prior(curves):
    kw = dict(curves=curves, on_date="2023-07-25", train_prior=TRAIN_PRIOR)

    p0  = risk_prior(**kw, lam=0.0)
    p1  = risk_prior(**kw, lam=1.0)
    p05 = risk_prior(**kw, lam=0.5)

    # lam=1 must differ from train_prior (non-zero intensities on this date)
    assert not np.allclose(p1, TRAIN_PRIOR, atol=1e-3), \
        "lam=1 should differ from train_prior when intensities are non-zero"

    # Linearity: lam=0.5 must be exactly the midpoint of lam=0 and lam=1
    np.testing.assert_allclose(p05, 0.5 * p0 + 0.5 * p1, atol=1e-12,
                               err_msg="lam=0.5 must equal (lam=0 + lam=1) / 2")


# ---------------------------------------------------------------------------
# 11. Zero intensities + lam=0.5 → every class prior > 0
#     This is the core fix: at season start all EPIRICE intensities are 0,
#     but blending with train_prior ensures no class is zeroed out, so
#     prior_shift cannot suppress any prediction entirely.
# ---------------------------------------------------------------------------
def test_zero_intensities_lam_half_all_classes_positive():
    dates      = pd.date_range("2023-07-15", periods=5, freq="D")
    zero_curves = pd.DataFrame(
        {d: np.zeros(5) for d in EPIRICE_DISEASES},
        index=dates,
    )
    result = risk_prior(zero_curves, on_date="2023-07-15",
                        train_prior=TRAIN_PRIOR, lam=0.5)

    assert np.all(result > 0), (
        f"All classes should be positive with lam=0.5 and zero intensities.\n"
        f"Got: {dict(zip(CLASSES, result))}"
    )
    assert abs(result.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 12. prior_shift on a confident leaf_blast image (p=0.9) with zero EPIRICE
#     intensities and lam=0.5 must still rank leaf_blast first.
#     Without the lam fix (lam=1), the weather prior sets leaf_blast≈0 at
#     season start and prior_shift suppresses the disease prediction.
# ---------------------------------------------------------------------------
def test_prior_shift_confident_leaf_blast_survives_zero_intensities():
    dates       = pd.date_range("2023-07-15", periods=5, freq="D")
    zero_curves = pd.DataFrame(
        {d: np.zeros(5) for d in EPIRICE_DISEASES},
        index=dates,
    )
    # Blended prior: lam=0.5 gives leaf_blast a non-zero share via train_prior
    prior = risk_prior(zero_curves, on_date="2023-07-15",
                       train_prior=TRAIN_PRIOR, lam=0.5)

    # Confident image prediction for leaf_blast
    leaf_blast_idx = CLASSES.index("leaf_blast")
    probs          = np.full(len(CLASSES), 0.01)
    probs[leaf_blast_idx] = 0.90
    probs /= probs.sum()

    adjusted = prior_shift(probs, train_prior=TRAIN_PRIOR, new_prior=prior)

    assert adjusted.argmax() == leaf_blast_idx, (
        f"Expected leaf_blast (idx={leaf_blast_idx}) to remain the top prediction.\n"
        f"Got idx={adjusted.argmax()} ({CLASSES[adjusted.argmax()]}), "
        f"adjusted={np.round(adjusted, 4)}"
    )
