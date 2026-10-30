"""
weather_risk/cli.py
===================
Command-line interface for the weather-risk layer.

Usage
-----
::

    python -m weather_risk.cli \\
        --lat 21.25 --lon 81.63 \\
        --establishment 2023-07-15 \\
        --date 2023-09-10

Prints:
  * Peak intensity per disease for the 120-day season.
  * The weather-risk prior over all classes on the requested date.

The script requires R and epicrop to be installed the first time it is run
for a given (lat, lon, establishment) triple; results are cached on disk
afterwards.  See weather_risk/README.md for install instructions.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .epirice import CLASSES, EPIRICE_DISEASES, risk_prior, run_epirice

# ---------------------------------------------------------------------------
# Placeholder training prior (uniform).
# Replace with real class frequencies computed from the loki4514 rice disease
# training split once the dataset audit is complete.  Do NOT use
# splits/crop_train.csv — that file belongs to the old crop-type classifier
# (Jute/Maize/Rice/Sugarcane/Wheat) and has nothing to do with the 9-class
# rice disease registry.
# ---------------------------------------------------------------------------
_UNIFORM_PRIOR = np.full(len(CLASSES), 1.0 / len(CLASSES))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Weather-risk prior via EPIRICE / NASA POWER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lat",           type=float, required=True,
                        help="Latitude in decimal degrees")
    parser.add_argument("--lon",           type=float, required=True,
                        help="Longitude in decimal degrees")
    parser.add_argument("--establishment", required=True,
                        help="Crop emergence date YYYY-MM-DD")
    parser.add_argument("--date",          required=True,
                        help="Target date for prior (YYYY-MM-DD)")
    parser.add_argument("--cache-dir",     default=".cache/epirice",
                        help="On-disk cache directory for R output")
    parser.add_argument("--lam",           type=float, default=0.5,
                        help="Blending weight: 1=pure weather prior, 0=train prior unchanged")
    args = parser.parse_args(argv)

    print(
        f"Running EPIRICE  lat={args.lat}  lon={args.lon}"
        f"  emergence={args.establishment}"
    )
    print("(First run fetches NASA POWER data via R; subsequent runs use cache.)\n")

    try:
        curves = run_epirice(
            args.lat, args.lon, args.establishment,
            cache_dir=args.cache_dir,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # -- peak intensity per disease ------------------------------------------
    col_w = max(len(d) for d in EPIRICE_DISEASES) + 2
    print("── Peak intensity per disease (120-day season) " + "─" * 20)
    for disease in EPIRICE_DISEASES:
        peak      = curves[disease].max()
        peak_date = curves[disease].idxmax().date()
        print(f"  {disease:<{col_w}}  peak = {peak:.4f}   (day of peak: {peak_date})")

    # -- weather-risk prior --------------------------------------------------
    try:
        prior = risk_prior(curves, on_date=args.date, train_prior=_UNIFORM_PRIOR,
                           lam=args.lam)
    except ValueError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n── Weather-risk prior on {args.date}  (lam={args.lam}) " + "─" * 20)
    col_w2 = max(len(c) for c in CLASSES) + 2
    for cls, p in zip(CLASSES, prior):
        bar = "█" * max(1, round(p * 40))
        print(f"  {cls:<{col_w2}}  {p:.4f}  {bar}")

    print(f"\n  total = {prior.sum():.8f}")


if __name__ == "__main__":
    main()
