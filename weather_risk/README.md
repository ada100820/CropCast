# weather_risk — EPIRICE Disease Risk Layer

The `weather_risk` package integrates the
[EPIRICE](https://doi.org/10.1016/j.cropro.2012.01.007) rice epidemic model
(Savary et al. 2012) with NASA POWER weather data to produce a daily
disease-risk prior over the nine project classes.  This prior is later used for
prior-shift fusion with the image classifier (Saerens et al. 2002).

---

## 1. Setup

### 1.1 R and epicrop

Install R ≥ 4.2 from <https://cran.r-project.org/>.

Then install **epicrop** from the R-universe (do **not** use CRAN — the
R-universe build is newer):

```r
install.packages(
  "epicrop",
  repos = c(
    "https://adamhsparks.r-universe.dev",
    "https://cloud.r-project.org"
  )
)
```

Verify:

```r
library(epicrop)
?get_wth
example(predict_bacterial_blight)
```

### 1.2 Python dependencies

```bash
pip install -r requirements.txt
```

The weather-risk layer itself only needs **numpy** and **pandas**; the full
`requirements.txt` also covers the image model and app.

---

## 2. Usage

### 2.1 Run via CLI

```bash
python -m weather_risk.cli \
    --lat 21.25 --lon 81.63 \
    --establishment 2023-07-15 \
    --date 2023-09-10 \
    --lam 0.5
```

The first run fetches NASA POWER data through R and caches results in
`.cache/epirice/`.  Subsequent runs with the same arguments are instant.

Sample output:

```
Running EPIRICE  lat=21.25  lon=81.63  emergence=2023-07-15

── Peak intensity per disease (120-day season) ──────────────────────
  bacterial_blight        peak = 0.1843   (day of peak: 2023-09-28)
  brown_spot              peak = 0.2156   ...
  ...

── Weather-risk prior on 2023-09-10 ─────────────────────────────────
  healthy                    0.2941  ████████████
  bacterial_leaf_blight      0.0823  ███
  ...
  total = 1.00000000
```

### 2.2 Use from Python

```python
from weather_risk.epirice import run_epirice, risk_prior, prior_shift
import numpy as np

# 1. Fetch / load cached intensity curves
curves = run_epirice(lat=21.25, lon=81.63, establishment_date="2023-07-15")

# 2. Build weather-risk prior for a specific date
train_prior = np.array([...])   # 9 values, one per class, from loki4514 training split
prior = risk_prior(curves, on_date="2023-09-10", train_prior=train_prior, lam=0.5)

# 3. Reweight image-model softmax output
image_probs = np.array([...])   # from EfficientNet + temperature scaling
adjusted    = prior_shift(image_probs, train_prior=train_prior, new_prior=prior)
```

---

## 3. Running the tests

```bash
pytest          # uses pytest.ini: testpaths = tests
```

All tests are offline — fixture CSVs under `tests/fixtures/` replace real
epicrop output, and subprocess calls are mocked.

---

## 4. How it works

```
NASA POWER API
      │  (HTTP, via epicrop::get_wth)
      ▼
run_epirice.R   ──►  bacterial_blight.csv
                     brown_spot.csv
                     leaf_blast.csv       }  one CSV per disease
                     sheath_blight.csv       (dates, intensity)
                     tungro.csv
      │
      ▼  (cached on disk; Python reads CSVs)
epirice.py :: run_epirice()  →  pd.DataFrame  (date index, 5 disease columns)
      │
      ▼
epirice.py :: risk_prior()   →  np.ndarray shape (9,)  prior over CLASSES
      │
      ▼
epirice.py :: prior_shift()  →  reweighted image-model posteriors
```

The five EPIRICE diseases map to project classes as follows:

| EPIRICE name       | Project class            |
|--------------------|--------------------------|
| bacterial_blight   | bacterial_leaf_blight    |
| brown_spot         | brown_spot               |
| leaf_blast         | leaf_blast               |
| sheath_blight      | sheath_blight            |
| tungro             | tungro                   |

The three remaining classes (`leaf_scald`, `narrow_brown_spot`, `rice_hispa`)
have no EPIRICE model and keep their training-set prior (neutral).

---

## 5. Design notes for the thesis

> **EPIRICE models potential epidemic risk, not observed incidence.**
> The model runs under "disease-conducive but unmanaged" assumptions (Savary et
> al. 2012): no fungicide applications, no resistant varieties.  The intensity
> values therefore represent a *ceiling* for field conditions, not a direct
> observation.  Results should be interpreted as relative risk rankings across
> dates and locations, not as probability-of-infection estimates.

> **The `lam` parameter controls how strongly the weather signal overrides the
> training prior.**  At the start of the growing season all EPIRICE intensities
> are near zero; a pure weather prior (`lam=1`) would therefore assign ~0
> probability to every disease class, and `prior_shift` would suppress any
> disease prediction from the image model regardless of how confident the image
> evidence is.  The blending parameter `lam` fixes this: `lam=0.5` gives each
> class at least half of its training-prior share as a floor.  The thesis
> ablation should compare `lam ∈ {0, 0.25, 0.5, 0.75, 1.0}` and report the
> effect on fusion accuracy across early, mid, and late season images.

> **The intensity-to-prior mapping is a design choice to be ablated.**
> Converting daily EPIRICE intensity into a class prior via the formula
> `healthy_frac = max(min_healthy, 1 − Σ intensities)` is one possible
> mapping; others (e.g. logistic scaling, learned temperature parameter) may
> perform better.  The thesis should include an ablation comparing at least:
> (a) `lam=0` (weather layer completely disabled),
> (b) this linear mapping at `lam=0.5`, and
> (c) a learned calibration of `lam`, `min_healthy`, and `eps` on a held-out
>     location.

---

## 6. File index

```
weather_risk/
  __init__.py      package marker
  run_epirice.R    R CLI script (Rscript run_epirice.R lat lon date out_dir)
  epirice.py       Python API: run_epirice, risk_prior, prior_shift
  cli.py           __main__ entry point

tests/
  test_weather.py  pytest suite (offline, no R required)
  fixtures/        small synthetic CSVs for testing
    bacterial_blight.csv
    brown_spot.csv
    leaf_blast.csv
    sheath_blight.csv
    tungro.csv
```
