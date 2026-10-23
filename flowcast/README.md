# FlowCast — AI-Driven Traffic Analytics Platform

An AI-driven traffic analytics platform that predicts traffic volume,
congestion, travel time, and accident risk using sensor, weather, and
calendar data — helping city traffic authorities make proactive traffic
management decisions.

Built on the Northline Corridor dataset: 25 road segments, 30-minute
sensor windows, January–May 2025 (~178K raw traffic readings).

---

## Pipeline Status

| Module | Description | Status |
|---|---|---|
| M1 | Ingestion & Validation | ✅ Complete |
| M2 | Cleaning & Merging | ✅ Complete |
| M3 | Feature Engineering | ✅ Complete |
| M4 | Classical ML Models | ✅ Complete |
| M5 | Deep Learning (LSTM) | ✅ Complete |
| M6 | Dashboard | ⏳ Not started |

---

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### Setup
```bash
git clone https://github.com/SAMARTH-SHARMA3/Flowcast-Ai_Driven_Traffic_Analytic_Platform-/tree
cd Flowcast-Ai_Driven_Traffic_Analytic_Platform-/flowcast
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the pipeline so far
```bash
# Stage 1: load and validate the 3 raw source tables
python src/ingest.py

# Stage 2: clean, merge, and save the analysis-ready dataset
python src/clean.py
```
Running `clean.py` produces `data/processed/flowcast_clean.parquet` —
the single merged, cleaned table (176,701 rows, 33 columns) that all
later stages (feature engineering, modeling, dashboard) will consume.

### Data
Raw source files (not all committed for size reasons — check
`data/raw/` and `docs/FlowCast_Data_Dictionary.docx` for full column
definitions):
- `traffic_sensor_log.csv` — 25 road segments × 30-min windows, ~178K rows
- `weather_observations.csv` — 3 stations × hourly readings
- `calendar_events.csv` — 1 row per date, holiday/event/roadwork flags

---

## Project Structure

![FlowCast project structure](data/docs/images/project_structure_tree.png)



---

## M1 — Data Ingestion & Validation

**Code:** `src/ingest.py` implements three loader functions —
`load_calendar()`, `load_weather()`, `load_traffic()` — each performing
schema validation, type/date parsing, and flagging (not fixing) every
documented data-quality defect: duplicates, missing values, impossible
values, and label inconsistency. Fixing is deliberately deferred to
`src/clean.py` (M2), so validation and remediation remain independently
auditable.

**Verification:** Running `python src/ingest.py` standalone reproduces
the exact defect counts found manually during EDA (see
`notebooks/01_raw_data_profiling.ipynb`) — this agreement between
independent exploration and the automated pipeline is the evidence that
the validation logic is correct, not merely that it executes without
error.

**Findings:**
- 1,767 key-level duplicate rows (road_id + date + time) — sensor retries
- 4,387 / 4,382 / 4,383 missing traffic_volume / avg_speed / occupancy (~2.5% each)
- 241 negative traffic_volume, 237 avg_speed > 200km/h, 234 occupancy > 100%
- 26,883 (15.1%) blank congestion_level — by design, derived later from volume/capacity ratio
- weather_condition: 13 raw label variants collapsing to 4 real categories
- 167 missing temperature, 111 missing visibility readings
- Cross-file key consistency confirmed: 100% overlap between traffic's
  weather_station_id and weather's station_id (no orphaned keys)

---

## M2 — Data Cleaning & Merging

`src/clean.py` takes the three validated raw tables from M1 and fixes
every documented defect, producing one clean, merged, analysis-ready
table. The 9 steps run in a strict order, since each one assumes the
previous is already done — e.g. duplicates must be removed before
computing statistics, impossible values must be nulled before outlier
detection, and both tables' dates must be parsed before merging.

### 1. Deduplication
Sensor retries created duplicate rows sharing the same
`road_id + date + time`. Removed via `drop_duplicates(keep='first')` —
retries were treated as the same physical event logged twice, not
independent readings, so averaging wasn't used.
**Result:** 178,468 → 176,701 rows (1,767 removed).

### 2. Fixing Impossible Values
Negative `traffic_volume`, `avg_speed` > 200 km/h, and `occupancy` >
100% are physically impossible, not just statistically unusual. These
were set to `NaN` (not dropped as whole rows) so the rest of each row
stays usable, and so imputation (step 4) can treat them the same as
genuine missing data.
**Result:** 239 / 236 / 229 values nulled (traffic_volume / avg_speed / occupancy).

### 3. Statistical Outlier Handling
Outliers were detected using IQR bounds computed *within* each
congestion_level group rather than globally. An initial global-IQR pass
flagged 4,705 avg_speed outliers, 85% of which fell under Severe
congestion — meaning the naive approach was mostly flagging real
slow-traffic conditions as "wrong," not detecting sensor errors.
Grouped IQR (per congestion_level) reduced this to 527, concentrated in
Free-flow conditions, where an unusually slow reading is genuinely
suspicious.

**Decision:** outliers are flagged (`*_is_outlier` boolean columns) but
not nulled or imputed. This is a conservative choice — no real sensor
readings are altered or fabricated — and the flags remain available as
potential model features (an "anomalous reading" signal may itself be
predictive).

### 4. Imputation
Missing values (~2.5% each for traffic_volume, avg_speed, occupancy —
combining original sensor dropout and nulled impossible values) were
filled using grouped median imputation: median value for the same
road_id + hour + day_of_week combination, falling back to a global
median for any group with no valid values. This preserves realistic
time-of-day and road-specific traffic patterns instead of flattening
gaps to a single corridor-wide average.

### 5. Weather Label Harmonization
`weather_condition` had 13 raw variants (case differences, trailing
whitespace, alternate spellings like 'rainy'/'foggy') representing only
4 real categories. Normalized via lowercase + strip, then mapped to the
data dictionary's controlled vocabulary (Clear/Cloudy/Rain/Fog).
Overcast was folded into Cloudy, since it isn't one of the 4 documented
target categories.
**Result:** Clear 8,168 (75%) · Cloudy 2,202 (20%) · Rain 426 (4%) ·
Fog 76 (0.7%) — total matches raw row count exactly, confirming no
data loss.

### 6. Vehicle Type Distribution Parsing
`vehicle_type_dist` (a JSON string per row, e.g.
`{"2W":0.37,"Car":0.43,...}`) was parsed into 4 separate numeric
columns: `pct_2w`, `pct_car`, `pct_lcv`, `pct_hcv`.
**Result:** all 176,701 rows parsed successfully with zero failures.
Row-wise sums range 0.99–1.01 (mean 0.9999, median 1.0), confirming
valid distributions with only expected floating-point rounding.

### 7. Timestamp Alignment
Traffic (30-min grain) and weather (hourly grain) timestamps were
reconciled by flooring each traffic window's timestamp to its
containing hour (`weather_join_hour`). Both the `:00` and `:30` windows
within an hour correctly map to the same hourly weather reading, per
the data dictionary's specification. (Note: `weather_station_id` vs.
`station_id`, and the DD/MM/YYYY vs. YYYY-MM-DD date format difference,
were already resolved at the **ingestion** stage in `ingest.py` — each
loader parses dates into real `datetime` objects with an explicit
`format=` argument, so no format ambiguity remains by the time
cleaning/merging happens.)

### 8. Merging
Weather (joined on station_id + aligned hour) and calendar (joined on
date) were left-merged onto the traffic table, which remains the base
grain throughout.
**Result:** 176,701 rows before and after merge (no row loss or
duplication), with 0 unmatched weather keys and 0 unmatched calendar
keys — confirming full key alignment across all three source tables.

### 9. Deriving Congestion Level
The ~15% blank congestion_level values were filled using the
volume/capacity ratio formula from the data dictionary
(`V/C = traffic_volume / (road_capacity/2)`, banded into
Free-flow/Moderate/Heavy/Severe). Existing real labels were never
overwritten — only null values were filled.
**Result:** filled 26,624 values, leaving 0 missing.

**Final class distribution:** Free-flow 108,544 (61%) · Moderate 42,103
(24%) · Heavy 16,308 (9%) · Severe 9,746 (5.5%) — a meaningful class
imbalance to account for during classification modeling (M4).

### Schema Cleanup & Saving
After merging, two genuinely redundant columns were dropped in a final
`finalize_columns()` step:
- `station_id` — exact duplicate of `weather_station_id` after the merge
- `vehicle_type_dist` — raw JSON string, superseded by the parsed
  `pct_2w` / `pct_car` / `pct_lcv` / `pct_hcv` columns

All other derived columns (`hour`, `day_of_week`, `timestamp`,
`weather_join_hour`, `*_is_outlier` flags) were kept — not part of the
original raw schema, but useful inputs for feature engineering (M3).

The final table (176,701 rows, 33 columns) is saved to
`data/processed/flowcast_clean.parquet`. Parquet was chosen over CSV to
preserve exact dtypes (datetime, bool, float) and for faster reload in
later pipeline stages.

---

## Roadmap
- **M3 — Feature Engineering:** 


Built per PRD Section 10.1's concept-to-application map. All lag and
rolling features are computed within each road segment in strict time
order (sorted by road_id + timestamp, grouped by road_id), with
rolling windows shifted before aggregation — this guarantees no
feature uses information from a future window relative to the
prediction time (verified by hand on real data at each step; see
commit history).

**Temporal:** hour/day_of_week cyclical (sin/cos) encodings, so models
treat time as circular rather than a flat scale. Plus `is_weekend` and
`is_rush_hour` (weekday 7-9am/5-7pm) flags.

**Lag features:** traffic_volume and avg_speed at t-1 (30min), t-2
(1h), t-48 (same window, previous day) — per road segment.

**Rolling statistics:** rolling mean/std of traffic_volume and
avg_speed over the previous 4 and 8 windows (2h/4h) — shifted to
exclude each row's own value from its own window.

**Capacity:** volume-to-capacity ratio (v_c_ratio) and
capacity_headroom (remaining capacity before saturation).

**Weather:** rain_flag, low_visibility_flag (<1000m, standard
meteorological fog threshold, confirmed against this dataset's actual
301-10,000m range), and temperature_band (Cold/Mild/Hot, based on this
dataset's actual 5.8-29.8°C range).

**Calendar interactions:** holiday_peak_interaction (public_holiday ×
is_rush_hour — captures that a holiday rush hour behaves differently
from a normal one) and days_to_event (distance in days to the nearest
event; 0 on the event day itself, verified against a real event date).

**Categorical encoding:** weather_condition_clean and temperature_band
one-hot encoded (cheap, no false ordering); road_id label-encoded
(single integer column, avoids 25 mostly-zero one-hot columns —
appropriate since M4/M5 will use tree-based models that don't assume
numeric ordering between categories).

**Output:** 176,701 rows, 71 columns, saved to
`data/processed/flowcast_features.parquet`.



### Post-Hoc Fix: Weather Imputation Gap

During M4's formal validation step, it was discovered that temperature
(2,708 nulls) and visibility (1,806 nulls) had never been imputed — M2
originally only built imputation for traffic_volume/avg_speed/occupancy.
These gaps were small in the raw weather file (167/111 out of 10,872
rows) but multiplied across the merge (each weather reading maps to
multiple traffic rows). Fixed via time-based interpolation per weather
station (more appropriate than grouped-median, since weather varies
smoothly hour-to-hour rather than following traffic's strong
time-of-day pattern). Full pipeline re-run and re-verified: 0 nulls
remain in temperature/visibility, confirmed by validate_final_dataset().



### Post-Hoc Fix: congestion_level Not One-Hot Encoded

Discovered while defining M5's feature sets: congestion_level (current
window) had never been one-hot encoded in M3, despite weather_condition
and temperature_band both being handled. Fixed by adding
congestion_Free-flow/Heavy/Moderate/Severe to add_categorical_features().
This is NOT a leakage risk when used as a feature for target_congestion_next
(forecasting next window's congestion from the CURRENT window's congestion
is legitimate, unlike v_c_ratio/capacity_headroom, which are mathematically
derived from the current traffic_volume itself). Shape: 71 → 75 columns.







## M4 — EDA & Reporting

Exploratory analysis of the feature-engineered dataset, covering all 5
PRD-required mathematical concepts (descriptive statistics, spread,
correlation, covariance/multicollinearity, normality) — full analysis
in `notebooks/02_eda.ipynb`.

Key findings: identified target-leakage risks (`v_c_ratio`,
`capacity_headroom`, `vehicle_count` must be excluded when
`traffic_volume` is the regression target), severe multicollinearity
among lag/rolling features (relevant for linear models in M5),
confirmed class imbalance on both classification targets
(`congestion_level`, `accident_count`), and retroactively validated
M2's choice of grouped IQR over z-score for outlier detection (the
underlying data is not normally distributed).

EDA also surfaced and fixed a real gap: `temperature`/`visibility`
had never been imputed (see `reports/data_quality.md` Section 3) —
caught by M4's formal validation step, not earlier manual checks.

**Full report:** [`reports/data_quality.md`](reports/data_quality.md)
— defect log, resolution methods, EDA findings, validated guarantees,
and known limitations, consolidated across M1–M4.




## M5 — Classical ML Engine

### Forecasting Target Construction
Per PRD 11.1, all 4 targets are FORWARD-LOOKING ("next window"), not
same-window estimations — this is genuine forecasting. Built via
shift(-1) grouped by road_id (mirroring M3's backward shift() for lag
features). accident_count was first binarized (accident_flag: 0 vs
1+, per M4's class-imbalance finding) before shifting.

Verified by hand: target_volume_next on row N exactly matches
traffic_volume on row N+1 for the same road; the last row of each
road's series correctly shows NaN (25 nulls per target = 1 per road,
expected — these rows are dropped before training).



### Time-Based Split
Per PRD 11.3, split chronologically (never randomly) to respect
temporal ordering and avoid leaking future information into training:
Train 70% (2025-01-01 to 2025-04-16, 123,691 rows), Validation 15%
(2025-04-16 to 2025-05-09, 26,483 rows), Test 15% (2025-05-09 to
2025-05-31, 26,502 rows). The 25 rows with no next-window target
(each road's final timestamp) were dropped before splitting.


### Linear Regression Baseline (target_volume_next)

sklearn LinearRegression: RMSE 77.85, MAE 55.59, MAPE 15.19%, R² 0.927
(PRD target: MAPE ≤12%, not met by this baseline).

Error analysis (see notebooks/03_model_diagnostics.ipynb) found MAPE
is concentrated in low-volume windows: 24.2% in the lowest quintile of
actual traffic_volume vs. 10.8% in the highest — consistent with M4's
finding that traffic_volume is right-skewed and non-normal. A
log-transformed target was tested (log1p(volume) as training target,
expm1 to de-transform predictions) but did not meaningfully improve
MAPE (14.48%) while degrading RMSE/R² (97.81 / 0.885) — indicating
the limitation is Linear Regression's model capacity (a single global
linear relationship can't capture both low-traffic and high-congestion
regimes well), not merely a scale/transform issue.

This is expected and consistent with the PRD's own framing of Linear
Regression as an "interpretable baseline," with tree-based models
(Decision Tree, Random Forest, XGBoost) expected to perform better on
this non-linear, non-normal data (per M4's normality findings).


### Linear Regression — From-Scratch NumPy (Build-it-once rule)

Implemented via batch gradient descent (MSE cost, analytic gradients),
no sklearn. First attempt (lr=0.01, 1000 iter) under-converged
(MAPE 16.69% vs sklearn's 15.19%) — cost was still visibly decreasing
at iteration 900, not yet at the optimum. A second attempt at lr=0.1
diverged entirely (cost grew to inf/NaN within ~500 iterations) — too
large a step size caused gradient descent to overshoot the cost
minimum and compound exponentially; this failure mode is now guarded
against (training stops early with a warning if cost becomes non-finite).

Final: lr=0.03, 3000 iterations — smooth, monotonic convergence to
RMSE 77.98, MAE 55.83, MAPE 15.35%, R² 0.9268 — within 0.2% of
sklearn's closed-form solution (RMSE 77.85, MAPE 15.19%, R² 0.9270)
on every metric, confirming the implementation correctly solves the
same optimization problem via a different method.



### Decision Tree — Feature Importance Investigation

The tuned tree (max_depth=10) showed traffic_volume at 90.4% feature
importance — investigated rather than accepted at face value. Retraining
without traffic_volume caused only a marginal MAPE increase (11.50% →
11.74%, R² 0.9378 → 0.9371), confirming this is REDUNDANCY, not
fragility: the removed signal is substitutable via other multicollinear
features (congestion_Free-flow 59.9% importance, occupancy 8.5%,
road_capacity 9.0% — consistent with M4's VIF findings). The model is
robust to losing its top feature, not dependent on it.



### Project Structure Note (M5)

To avoid re-running the full data pipeline on every model experiment,
M5 is split: `src/ml_models.py` holds shared functions (target
construction, time-based split, feature selection, scaling, metrics)
plus a __main__ block that does one-time data prep and saves the
train/val/test splits to `data/processed/model_data_volume.pkl`.
Per-model scripts (`src/train_decision_tree.py`,
`src/train_random_forest.py`, etc.) load this pickle rather than
reprocessing raw data each run. **Important:** rerun `ml_models.py`
first if anything upstream (M3 features, targets, split ratios)
changes, since the per-model scripts depend on the pickle being current.

### Decision Tree

Unconstrained tree overfit severely (train RMSE 0.00/R² 1.0, test
RMSE 93.50/MAPE 14.82%) — perfectly memorized training data. Tuned
max_depth via validation set (grid: 3/5/8/10/15/None); max_depth=10
was optimal (val MAPE 11.22%), confirming the expected bias-variance
U-curve (too shallow underfits, too deep overfits).

**Final (test):** RMSE 71.87, MAE 48.09, **MAPE 11.50%** (PRD target
≤12%: **PASS** — first model to meet it), R² 0.9378.

Feature importance showed traffic_volume at 90.4% — investigated
rather than accepted at face value. Retraining without it caused only
a marginal MAPE increase (11.50%→11.74%), confirming this reflects
REDUNDANCY (substitutable via congestion_Free-flow, occupancy,
road_capacity — all flagged as multicollinear with traffic_volume in
M4's VIF analysis), not fragile over-reliance on one feature.

### Random Forest

Grid search over n_estimators (50/100/200) × max_depth (10/15/None)
on validation. Best: n_estimators=200, max_depth=None (val MAPE
10.10%) — notably, unconstrained depth performed WELL here (unlike
the single Decision Tree), since averaging across 200 independently-
sampled trees cancels out individual-tree overfitting, which is the
core mechanism bagging is designed to exploit.

**Final (test):** RMSE 63.45, MAE 42.65, **MAPE 10.19%** (PRD target:
**PASS**), R² 0.9515 — best regression result so far, beating Decision
Tree and both Linear Regression variants on every metric.

### Scoreboard So Far (target_volume_next, test set)

| Model | RMSE | MAE | MAPE | R² | PRD Target (≤12%) |
|---|---|---|---|---|---|
| Linear Regression (sklearn) | 77.85 | 55.59 | 15.19% | 0.9270 | FAIL |
| Linear Regression (NumPy scratch) | 77.98 | 55.83 | 15.35% | 0.9268 | FAIL |
| Decision Tree (depth=10) | 71.87 | 48.09 | 11.50% | 0.9378 | PASS |
| Random Forest (200, None) | 63.45 | 42.65 | **10.19%** | **0.9515** | PASS |




### XGBoost

Hyperparameter search over max_depth (4/6/8) × learning_rate
(0.01/0.05/0.1), with early stopping (20 rounds patience) on the
validation set rather than a fixed boosting-round count — lets each
configuration find its own natural stopping point. A follow-up check
with a higher ceiling (3000 rounds) confirmed the lr=0.01 configs had
genuinely converged, not been artificially capped, validating the
original comparison.

Best: max_depth=8, learning_rate=0.1 (val MAPE 10.11%, converging in
just 114 rounds).

**Final (test):** RMSE 61.99, MAE 41.81, **MAPE 9.98%** (first model
under 10%), R² 0.9537 — best result on every metric, confirming the
PRD's prediction as the top classical performer.

Feature importance is notably more distributed than Decision
Tree/Random Forest: traffic_volume drops to 66.0% (vs. 86-90% for the
tree-based bagging models), with is_rush_hour (6.3%), hour_cos (4.7%),
and avg_speed_rolling_mean_4/8 (2.5%/2.3%) all contributing
meaningfully. This reflects boosting's core mechanism — each
successive tree targets the residual errors of previous trees, so
once early trees exhaust traffic_volume's explanatory power, later
trees are forced to extract signal from the engineered features
(rush-hour flags, rolling stats, calendar effects) — a concrete
demonstration that M3's feature engineering work pays off.

### Scoreboard So Far (target_volume_next, test set)

| Model | RMSE | MAE | MAPE | R² | PRD Target (≤12%) |
|---|---|---|---|---|---|
| Linear Regression (sklearn) | 77.85 | 55.59 | 15.19% | 0.9270 | FAIL |
| Linear Regression (NumPy scratch) | 77.98 | 55.83 | 15.35% | 0.9268 | FAIL |
| Decision Tree (depth=10) | 71.87 | 48.09 | 11.50% | 0.9378 | PASS |
| Random Forest (200, None) | 63.45 | 42.65 | 10.19% | 0.9515 | PASS |
| XGBoost (depth=8, lr=0.1) | 61.99 | 41.81 | **9.98%** | **0.9537** | PASS |
