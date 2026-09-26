# Flowcast-Ai_Driven_Traffic_Analytic_Platform-
An AI-driven traffic analytics platform that predicts traffic volume, congestion, travel time, and accident risk using sensor, weather, and calendar data, helping city traffic authorities make proactive traffic management decisions.


## Data Ingestion & Validation  M1

**Code:** `src/ingest.py` implements three loader functions — `load_calendar()`,
`load_weather()`, `load_traffic()` — each performing schema validation,
type/date parsing, and flagging (not fixing) every documented data-quality
defect: duplicates, missing values, impossible values, and label
inconsistency. Fixing is deliberately deferred to `src/clean.py` (M2), so
validation and remediation remain independently auditable.

**Verification:** Running `python src/ingest.py` standalone reproduces the
exact defect counts found manually during EDA (see `notebooks/01_raw_data_profiling.ipynb`) —
this agreement between independent exploration and the automated pipeline is
the evidence that the validation logic is correct, not merely that it executes
without error.

Raw data (3 CSVs, ~178K traffic rows across 25 road segments, Jan–May 2025)
is loaded and validated in `src/ingest.py`. This stage flags data-quality
issues without fixing them — fixes happen separately in `src/clean.py`,
keeping validation and remediation independently auditable.

**Findings:**
- 1,767 key-level duplicate rows (road_id + date + time) — sensor retries
- 4,387 / 4,382 / 4,383 missing traffic_volume / avg_speed / occupancy (~2.5% each)
- 241 negative traffic_volume, 237 avg_speed > 200km/h, 234 occupancy > 100%
- 26,883 (15.1%) blank congestion_level — by design, derived later from volume/capacity ratio
- weather_condition: 13 raw label variants collapsing to 4 real categories
- 167 missing temperature, 111 missing visibility readings
- Cross-file key consistency confirmed: 100% overlap between traffic's
  weather_station_id and weather's station_id (no orphaned keys)

### M2 cleaning and impuding each documentd data

### Statistical Outlier Handling

Outliers were detected using IQR bounds computed *within* each
congestion_level group rather than globally. An initial global-IQR pass
flagged 4,705 avg_speed outliers, 85% of which fell under Severe
congestion — meaning the naive approach was mostly flagging real
slow-traffic conditions as "wrong," not detecting sensor errors. Grouped
IQR (per congestion_level) reduced this to 527, concentrated in
Free-flow conditions where an unusually slow reading is genuinely
suspicious.

Decision: outliers are flagged (`*_is_outlier` boolean columns) but not
nulled or imputed. This is a conservative choice — no real sensor
readings are altered or fabricated — and the flags remain available as
potential model features (an "anomalous reading" signal may itself be
predictive).



### Imputation

Missing values (~2.5% each for traffic_volume, avg_speed, occupancy —
combining original sensor dropout and nulled impossible values) were
filled using grouped median imputation: median value for the same
road_id + hour + day_of_week combination, falling back to a global
median for any group with no valid values. This preserves realistic
time-of-day and road-specific traffic patterns instead of flattening
gaps to a single corridor-wide average.

### Weather Label Harmonization

weather_condition had 13 raw variants (case differences, trailing
whitespace, alternate spellings like 'rainy'/'foggy') representing only
4 real categories. Normalized via lowercase + strip, then mapped to the
data dictionary's controlled vocabulary (Clear/Cloudy/Rain/Fog).
Overcast was folded into Cloudy, since it isn't one of the 4 documented
target categories. Result: Clear 8,168 (75%) · Cloudy 2,202 (20%) ·
Rain 426 (4%) · Fog 76 (0.7%) — total matches raw row count exactly,
confirming no data loss.



### Vehicle Type Distribution Parsing

vehicle_type_dist (a JSON string per row, e.g. {"2W":0.37,"Car":0.43,...})
was parsed into 4 separate numeric columns: pct_2w, pct_car, pct_lcv,
pct_hcv. All 176,701 rows parsed successfully with zero failures.
Row-wise sums range 0.99–1.01 (mean 0.9999, median 1.0), confirming
valid distributions with only expected floating-point rounding.


