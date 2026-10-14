# FlowCast — Data Quality Report

## 1. Overview

**Source data:** 3 raw CSV tables, provided per the FlowCast Data Dictionary:
- `traffic_sensor_log.csv` — 178,468 rows, 25 road segments, 30-minute
  windows, January–May 2025
- `weather_observations.csv` — 10,872 rows, 3 weather stations
  (WS-CENTRAL, WS-NORTH, WS-SOUTH), hourly readings
- `calendar_events.csv` — 151 rows, 1 per calendar date, with public
  holiday / event / roadwork flags

**Pipeline stages applied:** Ingestion & Validation (M1) → Cleaning &
Merging (M2) → Feature Engineering (M3) → EDA & Reporting (M4), per
the project PRD's module structure.

**Final output:** `data/processed/flowcast_features.parquet` —
176,701 rows (after deduplication), 71 columns (36 cleaned/merged +
35 engineered), formally validated via `src/utils.py`'s
`validate_final_dataset()`.

**Reproducibility:** the full pipeline can be regenerated from raw
data by running, in order:
```bash
python src/ingest.py
python src/clean.py
python src/features.py
python src/utils.py   # validation gate
```

---

## 2. Defects Found & Resolved

| # | Defect | Raw Count | Resolution Method | Result | Module |
|---|---|---|---|---|---|
| 1 | Key-level duplicate rows (road_id+date+time) — sensor retries | 1,767 | `drop_duplicates(keep='first')` | 178,468 → 176,701 rows | M2 |
| 2 | Negative `traffic_volume` | 241 (239 post-dedup) | Set to NaN (impossible, not dropped) | Later imputed | M2 |
| 3 | `avg_speed` > 200 km/h | 237 (236 post-dedup) | Set to NaN | Later imputed | M2 |
| 4 | `occupancy` > 100% | 234 (229 post-dedup) | Set to NaN | Later imputed | M2 |
| 5 | Missing `traffic_volume` (sensor dropout + #2 above) | 4,387 raw | Grouped median imputation (road_id + hour + day_of_week) | 4,587 values filled, 0 remaining | M2 |
| 6 | Missing `avg_speed` (sensor dropout + #3 above) | 4,382 raw | Grouped median imputation | 4,579 values filled, 0 remaining | M2 |
| 7 | Missing `occupancy` (sensor dropout + #4 above) | 4,383 raw | Grouped median imputation | 4,573 values filled, 0 remaining | M2 |
| 8 | Blank `congestion_level` (by design — meant to be derived) | 26,883 raw (15.1%) / 26,624 post-dedup | Derived via V/C ratio formula (traffic_volume / (road_capacity/2)), banded into Free-flow/Moderate/Heavy/Severe | 0 remaining | M2 |
| 9 | `weather_condition` label inconsistency (13 raw variants: `RAIN`/`rain`/`rainy`/etc.) | 10,872 rows affected | Strip + lowercase + controlled-vocabulary mapping; `Overcast` folded into `Cloudy` | 3,355 values normalized → 4 clean categories (Clear/Cloudy/Rain/Fog) | M2 |
| 10 | Missing `temperature` | 167 | Time-based interpolation, per weather station | 0 remaining | M2 (found/fixed during M4 validation — see Section 3) |
| 11 | Missing `visibility` | 111 | Time-based interpolation, per weather station | 0 remaining | M2 (found/fixed during M4 validation — see Section 3) |
| 12 | Statistical outliers — `traffic_volume`, `avg_speed`, `occupancy` (naturally unusual, not impossible) | 53 / 527 / 342 (grouped IQR, per congestion_level) | Flagged (`*_is_outlier` boolean columns), **not nulled** — conservative choice, see Section 4 | Flags retained as potential model features | M2 |

**Note on outlier counts:** an initial *global* IQR pass incorrectly
flagged 4,705 `avg_speed` outliers, 85% of which were genuine Severe
congestion, not sensor errors — grouping IQR bounds by
`congestion_level` corrected this to the 527 shown above. See Section
5 for the full reasoning.

---

## 3. Post-Hoc Fix: Weather Imputation Gap

**How it was found:** during M4's formal validation step
(`validate_final_dataset()` in `src/utils.py`), `temperature` and
`visibility` were discovered to have 2,708 and 1,806 nulls respectively
in the final merged dataset — despite M2's cleaning stage never having
explicitly addressed them. This was not caught earlier because M2's
imputation work focused on `traffic_volume`/`avg_speed`/`occupancy`
(the columns M1 had flagged as having the largest, most visible gaps);
the smaller weather gaps (167 missing temperature, 111 missing
visibility, out of 10,872 raw weather rows — under 2% each) were
present from the start but never routed through an imputation step.

**Why the count grew (167→2,708, 111→1,806):** each single hourly
weather reading is joined onto *multiple* traffic rows during the M2
merge (one weather station serves many road segments, and each hourly
reading covers two 30-minute traffic windows). A small raw gap
therefore multiplies substantially once propagated through the merge
— a useful general lesson: upstream data gaps should be fixed *before*
a one-to-many join, not after, both for correctness and to keep the
fix cheap (167/111 rows to fix pre-merge, vs. 2,708/1,806 post-merge).

**Resolution:** `impute_weather()` (added to `src/clean.py`) uses
time-based linear interpolation, computed separately per weather
station, run on the raw weather table *before* the merge:

```python
w[col] = w.groupby("station_id")[col].transform(
    lambda s: s.interpolate(method="linear", limit_direction="both")
)
```

**Why interpolation, not grouped-median (as used for traffic):**
weather varies smoothly and continuously over time (temperature drifts
gradually hour-to-hour), unlike traffic volume's strong, discontinuous
time-of-day pattern (rush hour vs. 3am are genuinely different
regimes, which is why grouped-median by hour/day made sense there).
Interpolating between the nearest real readings is a better physical
model for weather than borrowing a value from a different time-of-day
bucket. A per-station median fallback handles any edge case
interpolation can't reach (e.g. a gap at the very start of a station's
series, with no earlier point to interpolate from).

**Verification:** full pipeline (`ingest.py` → `clean.py` →
`features.py` → `utils.py`) was re-run end-to-end after the fix.
Imputed exactly 167 temperature and 111 visibility values — matching
the original raw defect counts precisely, confirming the fix targeted
exactly the documented gap, nothing more or less.
`validate_final_dataset()` confirms 0 nulls remain in both columns.

**Takeaway:** this gap existed silently through M1, M2, and M3 — it
was only caught because M4 introduced a formal, assert-based
validation gate rather than relying on scattered manual checks. Ad
hoc verification catches what you think to check; a systematic final
validation gate catches what you didn't think to check.

---

## 4. Feature Engineering Decisions (M3)

71 columns total in the final dataset: 36 from cleaning/merging + 35
engineered features, built per PRD Section 10.1's concept-to-application
map. All lag and rolling features were built with explicit leakage
guards — verified by hand against real data at each step.

### Leakage Prevention
- **Temporal leakage guard:** all lag/rolling features computed via
  `sort_values(["road_id", "timestamp"])` + `groupby("road_id")`,
  ensuring no value crosses between different roads' time series.
- **Future-leakage guard:** rolling window features apply `.shift(1)`
  *before* `.rolling()`, excluding each row's own value from its own
  window — otherwise the model would partially "see" the value it's
  meant to predict.

### Feature Categories
| Category | Features | Notes |
|---|---|---|
| Temporal | `hour_sin/cos`, `dow_sin/cos`, `is_weekend`, `is_rush_hour` | Cyclical encoding so 23:00≈00:00, Sun≈Mon |
| Lag | `traffic_volume`/`avg_speed` at t-1, t-2, t-48 | Per road_id; verified against real prior readings |
| Rolling | mean/std of `traffic_volume`/`avg_speed`, 4 & 8 step windows (2h/4h) | Shifted to exclude current row |
| Capacity | `v_c_ratio`, `capacity_headroom` | **Target leakage risk for traffic_volume regression — see Section 5** |
| Weather | `rain_flag`, `low_visibility_flag` (<1000m), `temperature_band` (Cold/Mild/Hot) | Thresholds chosen from actual data range (visibility 301–10,000m, temp 5.8–29.8°C) |
| Calendar | `holiday_peak_interaction`, `days_to_event` | Verified `days_to_event=0` on a confirmed real event date |
| Categorical | One-hot: weather_condition_clean, temperature_band; Label-encoded: road_id | One-hot for low-cardinality (≤4 categories); label encoding for road_id (25 categories) since tree models don't assume ordering |

### Output
176,701 rows, 71 columns → `data/processed/flowcast_features.parquet`

---

## 5. Key EDA Findings (M4)

Full analysis in `notebooks/02_eda.ipynb`. This section summarizes the
findings that materially affect M5/M6 modelling decisions.

### 5.1 Descriptive Statistics & Skew
| Column | Mean | Median | Skew Direction |
|---|---|---|---|
| traffic_volume | 431.50 | 362.00 | Right-skewed (congestion spikes) |
| avg_speed | 42.11 | 44.90 | Left-skewed (congestion floor) |
| occupancy | 31.17 | 27.00 | Right-skewed |
| visibility | 9,672.17 | 10,000.00 | Left-skewed (sensor ceiling at 10,000) |
| temperature | 17.98 | 18.00 | Near-symmetric |

A hypothesis that `traffic_volume`'s mode (149.0, appearing 566 times)
was a grouped-median-imputation artifact was tested and **rejected**:
the value is spread across all 25 road segments and 15 different
hours with no meaningful concentration (max ~10% in any single road),
consistent with it being a naturally common integer vehicle count
rather than a pipeline side effect.

### 5.2 Target Leakage Risks — Critical for M5

Two features must be **excluded** from the feature set when
`traffic_volume` is the regression target:

- **`v_c_ratio`** (r=0.948) and **`capacity_headroom`** (r=-0.836) —
  both are algebraically derived FROM traffic_volume in M3
  (`v_c_ratio = traffic_volume / (road_capacity/2)`). Including them
  would let a model trivially reconstruct the target via near-inverse
  arithmetic rather than learning genuine patterns.
- **`vehicle_count`** (r=0.998) — per the data dictionary, this is
  "raw sensor count before flow adjustment," i.e. an earlier-stage
  version of traffic_volume itself. Arguably the more severe leakage
  risk of the three, since it's closer to being a pre-adjustment copy
  of the target rather than a derived ratio.

These three remain valid features for *other* targets (e.g.
`congestion_level` classification), where they represent legitimate
derived signal rather than a restatement of that specific target.

### 5.3 Multicollinearity (VIF Analysis)

VIF (Variance Inflation Factor) was computed on 171,069 of 176,701
rows (96.8% coverage; remainder excluded due to expected lag/rolling
series-start NaNs). Severe multicollinearity (VIF > 10) found among:

| Feature | VIF |
|---|---|
| traffic_volume_lag_1 | 51.81 |
| traffic_volume_rolling_mean_4 | 45.19 |
| traffic_volume | 34.48 |
| avg_speed_rolling_mean_4 | 27.73 |
| avg_speed_lag_1 | 24.63 |
| occupancy | 16.91 |
| avg_speed | 16.44 |

This is expected — these features derive from the same underlying
traffic signal at different time offsets, and `avg_speed`/`occupancy`
are strongly physically linked (r=-0.94, confirmed via heatmap).

**Modelling implication:** tree-based models (Decision Tree, Random
Forest, XGBoost) are largely unaffected by multicollinearity and can
use the full feature set. Linear Regression and SVM will need either
a reduced feature subset (e.g. one lag + one rolling window, not all)
or regularization (Ridge/Lasso) for stable, interpretable coefficients.

### 5.4 Class Imbalance

**congestion_level** (classification target): Free-flow 61% ·
Moderate 24% · Heavy 9% · Severe 5.5% — moderate imbalance, worth
monitoring per-class precision/recall rather than relying on overall
accuracy.

**accident_count** (basis for accident-risk classification target):
**99.06%** of rows have 0 accidents, 0.93% have 1, 0.008% (15 rows)
have 2 — severe imbalance. A naive "always predict no accident"
classifier would score 99% accuracy while being completely
uninformative, which is precisely why the PRD specifies **ROC-AUC**
(not accuracy) for this target. Given only 15 rows have
accident_count=2, framing this target as **binary**
(`accident_flag`: 0 vs. 1+) rather than 3-class is likely more
tractable — a decision to carry into M5.

### 5.5 Normality Check

All 5 key sensor columns formally reject normality (D'Agostino-Pearson
test, p<0.001 for all): traffic_volume, avg_speed, occupancy,
travel_time, temperature. traffic_volume/occupancy are right-skewed;
avg_speed is left-skewed with a visible secondary low-speed cluster
(severe congestion) — visual confirmation of the bimodality that
motivated M2's use of **grouped IQR rather than z-score** for outlier
detection. travel_time is the most extreme (skew=3.65, kurtosis=20.5).

temperature is a notable edge case: skew≈0 and visually bell-shaped,
yet still formally rejects normality — attributable to the test's
high statistical power at n=176,701, where even minor deviations
become "significant." Illustrates the distinction between statistical
and practical significance.

**Conclusion:** M2's grouped-IQR choice over z-score is retroactively
validated — the underlying data is not normally distributed, so a
method assuming normality would have been statistically inappropriate.

---

## 6. Validated Guarantees

The following are formally asserted by `validate_final_dataset()` in
`src/utils.py`, run as the final gate after every pipeline execution
(`python src/utils.py`). Unlike the narrative findings in Sections 2–5,
these are automated checks that fail loudly (raise `AssertionError`)
if violated — not manual observations that could silently go stale as
the pipeline evolves.

| Guarantee | Check |
|---|---|
| Row count integrity | Exactly 176,701 rows (post-deduplication count established in M2) |
| No nulls in core modelling columns | `traffic_volume`, `avg_speed`, `occupancy`, `congestion_level`, `travel_time`, `accident_count`, `vehicle_count`, `road_capacity` |
| No nulls in weather columns | `temperature`, `rainfall`, `visibility`, `weather_condition_clean` (added after the Section 3 fix) |
| Value ranges respected | `traffic_volume ≥ 0`, `avg_speed` within 0–200, `occupancy` within 0–100, `visibility ≥ 0` |
| Lag/rolling nulls match expectation | Reported (not asserted as zero) — these columns have legitimate nulls at each road's time-series start, and the exact counts are checked against the known arithmetic (e.g. `lag_48` → 1,200 nulls = 48 steps × 25 roads) |

**What this does NOT guarantee:** `validate_final_dataset()` checks
structural integrity (shape, nulls, ranges) — it does not re-verify
the deeper correctness properties established manually during
development (e.g. that lag features are leakage-free, or that the
V/C-ratio congestion derivation is mathematically correct). Those were
verified by hand against real data during M2/M3 build-out and are
documented in Sections 2 and 4; they are stable because the
transformation logic itself hasn't changed since verification, not
because this automated gate re-checks them on every run. A future
improvement would be extending this gate with spot-check assertions
for those properties too (e.g. asserting a known lag value on a fixed
sample row).

---

## 7. Known Limitations

Honest caveats for anyone building on or evaluating this dataset:

**1. Sensor dropout creates non-contiguous time series.** Not every
30-minute window has a corresponding row — some windows are simply
absent (not null, genuinely missing as rows), confirmed directly
during M3's timestamp-alignment step (e.g. road NL-007 on 2025-02-26
jumps 00:00 → 01:30 → 02:00, skipping 00:30 and 01:00 entirely). This
means lag/rolling features computed via `.shift()`/`.rolling()`
sometimes reach back further in *real* elapsed time than their name
implies (e.g. `lag_1` may actually be 1+ hours old, not exactly 30
minutes, when a window was dropped). This was not corrected — doing so
would require reindexing onto a complete time grid and explicitly
marking gaps, a reasonable extension for M5/M6 if sequence models
(LSTM/GRU) prove sensitive to it.

**2. Severe class imbalance on two important targets.**
`congestion_level`'s Severe class is 5.5% of rows; `accident_count`'s
positive class is under 1%. Both require imbalance-aware evaluation
(precision/recall/F1, ROC-AUC) rather than raw accuracy — flagged in
Section 5, but worth restating as a standing limitation rather than a
one-time note, since it affects every future model built on this data.

**3. Imputed values are indistinguishable from real readings.**
Neither the traffic imputation (M2, grouped median) nor the weather
imputation (M2, interpolation) added a "was this value imputed" flag
column. Roughly 2.6% of traffic_volume/avg_speed/occupancy readings
and ~1.5–2.5% of weather readings are model-filled estimates, not
sensor truth, with no column marking which ones. A future iteration
could add `*_was_imputed` boolean flags, letting models weight
imputed values differently if needed.

**4. Dataset versioning is git-based, not a dedicated tool.**
Per FR-06 ("persist a versioned, analysis-ready dataset"), this
project tracks dataset versions through git commit history (the
parquet output files are committed alongside the pipeline code that
produced them) rather than a dedicated data-versioning tool like DVC.
Appropriate for this project's single-developer, single-corridor
scope, but means there's no built-in diffing or rollback mechanism
beyond standard git operations.

**5. Statistical outlier flags are informational, not enforced.**
`traffic_volume_is_outlier`, `avg_speed_is_outlier`,
`occupancy_is_outlier` (from M2's grouped-IQR pass) were deliberately
kept flag-only — the underlying values were never nulled or adjusted
(see Section 2, defect #12). This was a considered decision (grouped
IQR showed most flagged values were genuine extreme traffic
conditions, not errors — Section 5.5's conclusion reinforces this),
but it means M5/M6 models will train on these values as-is unless a
model explicitly uses the flag columns to treat them differently.

**6. Single corridor, single time period.** All data is from one
25-segment corridor over 5 months (Jan–May 2025). Findings about
distributions, correlations, and class balance are specific to this
corridor/period and should not be assumed to generalize to other
roads, cities, or seasons without re-validation.