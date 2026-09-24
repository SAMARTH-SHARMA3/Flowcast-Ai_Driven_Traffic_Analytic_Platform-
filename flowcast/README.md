# Flowcast-Ai_Driven_Traffic_Analytic_Platform-
An AI-driven traffic analytics platform that predicts traffic volume, congestion, travel time, and accident risk using sensor, weather, and calendar data, helping city traffic authorities make proactive traffic management decisions.


## Data Ingestion & Validation

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


  